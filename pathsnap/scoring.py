"""Key-safe candidate score loading for PathSnap development decoding.

The trusted score files are produced by scripts/score_baseline.py. All scoring
paths use the exact same candidate rows and ordering for top-1 and Viterbi.
"""

from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["split", "trip_id", "obs_id", "arc_id"]
ROOT = Path(__file__).resolve().parents[1]
SCORE_FILES = {
    "logistic_regression": ROOT / "experiments/baselines/logistic_regression_student_scores.parquet",
    "knn": ROOT / "experiments/baselines/knn_student_scores.parquet",
    "random_forest": ROOT / "experiments/random_forest/random_forest_student_scores.parquet",
    "xgboost": ROOT / "experiments/xgboost/xgboost_student_scores.parquet",
    "xgboost_tuned": ROOT / "experiments/xgboost_tuned/xgboost_tuned_student_scores.parquet",
    "mlp": ROOT / "experiments/mlp/mlp_student_scores.parquet",
}
DISPLAY_NAMES = {
    "distance": "Distance baseline (development)",
    "logistic_regression": "Logistic Regression (trained)",
    "knn": "KNN (trained)",
    "random_forest": "Random Forest (trained)",
    "xgboost": "XGBoost (trained)",
    "xgboost_tuned": "Tuned XGBoost (trained)",
    "mlp": "MLP (trained)",
}


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


def score_candidates(rows: pd.DataFrame, *, model="distance", trip_id=None, score_files=None):
    """Return (name, scores) in exactly the input candidate-row order.

    Select a development distance baseline or a trusted, committed
    student-test probability export. An API caller cannot specify a file path.
    Source score files never contain ground-truth labels.
    """
    if model not in DISPLAY_NAMES:
        raise ValueError(f"Unknown scorer {model!r}")
    if model == "distance":
        score = np.exp(-0.5 * (rows["distance_to_arc_m"].to_numpy(dtype=float) / 20.0) ** 2)
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
