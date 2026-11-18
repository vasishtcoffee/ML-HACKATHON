"""Key-safe candidate score loading for PathSnap development decoding.

The trusted score files are produced by scripts/score_baseline.py. All scoring
paths use the exact same candidate rows and ordering for top-1 and Viterbi.
"""

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["split", "trip_id", "obs_id", "arc_id"]
ROOT = Path(__file__).resolve().parents[1]
SCORE_FILES = {
    "logistic_regression": ROOT / "experiments/baselines/logistic_regression_student_scores.parquet",
    "knn": ROOT / "experiments/baselines/knn_student_scores.parquet",
}
LIVE_MODELS = {
    # Each entry is a repository-controlled trained artifact, not a user path.
    "live_baseline_logistic_regression": ROOT / "experiments/baselines/logistic_regression.joblib",
    "live_baseline_knn": ROOT / "experiments/baselines/knn.joblib",
    "live_baseline_mlp": ROOT / "experiments/mlp/mlp.joblib",
    "live_baseline_xgboost": ROOT / "experiments/xgboost/xgboost.joblib",
    "live_logistic_regression": ROOT / "experiments/classifier_tuning/logistic_regression/logistic_regression.joblib",
    "live_knn": ROOT / "experiments/classifier_tuning/knn/knn.joblib",
    "live_mlp": ROOT / "experiments/classifier_tuning/mlp/mlp.joblib",
    "live_random_forest": ROOT / "experiments/random_forest/random_forest.joblib",
    "live_tuned_random_forest": ROOT / "experiments/random_forest_tuned/random_forest.joblib",
    "live_xgboost": ROOT / "experiments/xgboost_tuned/xgboost.joblib",
}
MOTION_MODELS = {
    "live_heading_logistic_regression": "logistic_regression",
    "live_motion_knn": "knn",
    "live_motion_mlp": "mlp",
}
LIVE_MODELS.update({
    model: ROOT / f"experiments/motion_features/{name}.joblib"
    for model, name in MOTION_MODELS.items()
})
FINAL_MODEL = os.environ.get("PATHSNAP_FINAL_MODEL", "live_motion_knn").strip()
if FINAL_MODEL not in LIVE_MODELS:
    raise ValueError("PATHSNAP_FINAL_MODEL must identify a supported live_... classifier")
DISPLAY_NAMES = {
    "distance": "Distance baseline (development only)",
    "logistic_regression": "Logistic Regression (precomputed baseline)",
    "knn": "KNN (precomputed baseline)",
    "live_baseline_logistic_regression": "Logistic Regression baseline (live)",
    "live_baseline_knn": "KNN baseline (live)",
    "live_baseline_mlp": "MLP baseline (live)",
    "live_baseline_xgboost": "XGBoost baseline (live)",
    "live_logistic_regression": "Tuned Logistic Regression (live model)",
    "live_knn": "Tuned KNN (live model)",
    "live_mlp": "Tuned MLP (live model)",
    "live_random_forest": "Random Forest (live model, untuned)",
    "live_tuned_random_forest": "Tuned Random Forest (live model)",
    "live_xgboost": "Tuned XGBoost (live model)",
    "live_heading_logistic_regression": "Logistic Regression + heading (live)",
    "live_motion_knn": "KNN + motion (live)",
    "live_motion_mlp": "MLP + motion (live)",
}
DISPLAY_NAMES["final"] = "Final model · " + DISPLAY_NAMES[FINAL_MODEL]


@lru_cache(maxsize=len(LIVE_MODELS))
def _trusted_model(name: str):
    """Load only a shipped, team-trained artifact (joblib is not safe for uploads)."""
    if name not in LIVE_MODELS:
        raise ValueError("Unknown trained model")
    path = LIVE_MODELS[name]
    if not path.is_file():
        raise FileNotFoundError(f"Trusted trained model missing: {path}")
    import joblib

    artifact = joblib.load(path)  # trusted repository-controlled bytes ONLY
    from scripts.score_baseline import FEATURES
    baseline_id = name.removeprefix("live_").removeprefix("tuned_").removeprefix("baseline_")
    expected = MOTION_MODELS.get(name, baseline_id)
    if not isinstance(artifact, dict) or artifact.get("id") != expected:
        raise ValueError(f"Trained artifact does not match requested model {expected}")
    features = FEATURES
    if name in MOTION_MODELS:
        from scripts.motion_features import HEADING, MOTION

        features = FEATURES + HEADING + (MOTION if expected != "logistic_regression" else [])
    if artifact.get("features") != features:
        raise ValueError("Saved model feature schema differs from candidate generator")
    if artifact.get("score_kind") != "candidate_positive_probability":
        raise ValueError("Unsupported trained model score semantics")
    return artifact


def live_model_scores(rows: pd.DataFrame, model: str, *, geometry=None) -> np.ndarray:
    """Score new candidate geometry using a previously fitted classifier.

    Never accepts model paths, labels, or an evaluator-specific score export.
    The original score_baseline handles tuned KNN's KDTree/voting semantics.
    """
    selected = FINAL_MODEL if model == "final" else model
    if selected not in LIVE_MODELS:
        raise ValueError("Select an available live trained classifier")
    from scripts.score_baseline import FEATURES, score_candidates as predict_probabilities
    from threadpoolctl import threadpool_limits

    required = KEY + FEATURES
    missing = set(required) - set(rows.columns)
    if missing:
        raise ValueError(f"Candidate rows missing fields: {sorted(missing)}")
    if rows[KEY].isna().any().any() or rows.duplicated(KEY).any():
        raise ValueError("Live candidates contain null or duplicated keys")
    if not rows["trip_id"].map(lambda x: isinstance(x, str)).all():
        raise ValueError("Trip IDs must remain zero-padded strings")
    if rows.empty:
        return np.empty(0, dtype=float)
    artifact = _trusted_model(selected)
    if selected in MOTION_MODELS:
        if geometry is None:
            raise ValueError("Heading and motion models require the GPS track and road graph")
        from scripts.motion_features import motion_features

        track, nodes, arcs = geometry
        extra = motion_features(track, rows, nodes, arcs)
        rows = rows.merge(extra, on=KEY, validate="one_to_one", sort=False)
    data = rows[artifact["features"]].to_numpy(dtype=float)
    if not np.isfinite(data).all():
        raise ValueError("Candidate geometry must be finite")
    with threadpool_limits(limits=4):
        return np.asarray(predict_probabilities(artifact, rows), dtype=float)

def match_exported_scores(candidates: pd.DataFrame, exported: pd.DataFrame) -> np.ndarray:
    """Join a score file to candidates by full key, returning candidate-row order.

    Reject missing/extra/duplicate candidates, absent values, and bad score
    ranges. Never infer score alignment from Parquet row order.
    """
    for label, frame, required in (
        ("candidates", candidates, KEY),
        ("score export", exported, KEY + ["score"]),
    ):
        missing = set(required) - set(frame.columns)
        if missing:
            raise ValueError(f"{label} missing required columns: {sorted(missing)}")
        if frame[KEY].isna().any().any():
            raise ValueError(f"{label} contains null candidate keys")
        if frame.duplicated(subset=KEY).any():
            raise ValueError(f"{label} contains duplicate candidate keys")
        if not frame["trip_id"].map(lambda v: isinstance(v, str)).all():
            raise ValueError(f"{label} trip_id must be a string (preserve leading zeros)")

    ordered = candidates[KEY].copy()
    ordered["__original_order"] = np.arange(len(ordered), dtype=np.int64)
    joined = ordered.merge(
        exported[KEY + ["score"]], on=KEY, how="outer",
        validate="one_to_one", indicator=True, sort=False,
    )
    if not joined["_merge"].eq("both").all():
        count = joined["_merge"].value_counts()
        raise ValueError(
            "Candidate/score key mismatch: "
            f"{int(count.get('left_only', 0))} missing scores, "
            f"{int(count.get('right_only', 0))} unexpected scores"
        )
    scores = joined.sort_values("__original_order")["score"].to_numpy(dtype=float)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Score export must contain finite P(match) values in [0, 1]")
    return scores


def score_candidates(rows: pd.DataFrame, *, model="distance", trip_id=None, score_files=None, geometry=None):
    """Return (name, scores) in exactly the input candidate-row order.

    Select a development distance baseline or a trusted, committed
    student-test probability export. An API caller cannot specify a file path.
    Source score files never contain ground-truth labels.
    """
    if model not in DISPLAY_NAMES:
        raise ValueError(f"Unknown scorer {model!r}")
    if model == "distance":
        score = np.exp(-0.5 * (rows["distance_to_arc_m"].to_numpy(dtype=float) / 20.0) ** 2)
    elif model in LIVE_MODELS or model == "final":
        score = live_model_scores(rows, model, geometry=geometry)
    else:
        if trip_id is None or not isinstance(trip_id, str):
            raise ValueError("A string trip_id is required for trained-model scores")
        if rows.empty or not rows["trip_id"].eq(trip_id).all() or not rows["split"].eq("student_test").all():
            raise ValueError("Candidate rows must belong to the selected student-test trip")
        files = SCORE_FILES if score_files is None else score_files
        path = Path(files[model])
        if not path.is_file():
            raise FileNotFoundError(f"Missing trained score export: {path}")
        exported = pd.read_parquet(path, columns=KEY + ["score"], filters=[("trip_id", "==", trip_id)])
        score = match_exported_scores(rows, exported)
    if score.shape != (len(rows),) or not np.isfinite(score).all() or ((score < 0) | (score > 1)).any():
        raise ValueError("Scorer must return one finite probability in [0, 1] per candidate")
    return DISPLAY_NAMES[model], score
