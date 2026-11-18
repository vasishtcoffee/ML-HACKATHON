"""Train and evaluate PathSnap's baseline classifiers."""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from generate_candidates import ROOT
from score_baseline import FEATURES, KEY, score_candidates


OBS_KEY = KEY[:-1]
SEED = 42
N_FOLDS = 5
THRESHOLD = 0.5
DEFAULT_MODELS = ("logistic_regression", "knn")
MODELS = {
    "random_forest": {"n_estimators": 100, "max_depth": 16, "min_samples_leaf": 10,
                      "max_features": "sqrt", "n_jobs": 4, "random_state": SEED},
    "logistic_regression": {"C": 1.0, "max_iter": 200, "solver": "lbfgs", "random_state": SEED},
    "knn": {"n_neighbors": 31, "weights": "uniform", "algorithm": "kd_tree", "metric": "euclidean"},
}


def load_split(processed, split):
    labels = pd.read_parquet(processed / f"reference_labels_{split}.parquet")
    obs = pd.read_csv(
        processed / "observations.csv.gz", dtype={"trip_id": str},
        usecols=OBS_KEY + ["timestamp_s", "gap_before", "row_break_before"],
    )
    timeline = obs.loc[obs["split"].eq(split)].merge(labels, on=OBS_KEY, validate="one_to_one")
    candidates = pd.read_parquet(processed / f"candidates_{split}.parquet", columns=KEY + FEATURES)
    candidates = candidates.merge(
        labels[OBS_KEY + ["reference_arc_id", "alignment_status", "reference_in_candidates",
                          "usable_for_training"]],
        on=OBS_KEY, how="left", validate="many_to_one",
    )
    if candidates["alignment_status"].isna().any():
        raise ValueError(f"Candidate rows without observation labels in {split}.")
    candidates["positive"] = candidates["arc_id"].eq(candidates["reference_arc_id"]).fillna(False)
    candidates["binary_eligible"] = (
        candidates["alignment_status"].eq("aligned") & candidates["reference_in_candidates"]
    )
    return timeline.sort_values(["trip_id", "timestamp_s", "obs_id"]), candidates


def load_arcs(processed, archive_path):
    files = pd.read_csv(processed / "archive_inventory.csv", dtype={"trip_id": str})
    graphs = {}
    with ZipFile(archive_path) as archive:
        for row in files.loc[files["kind"].eq("arcs")].itertuples(index=False):
            with archive.open(row.member) as source:
                graphs[(row.split, row.trip_id)] = np.loadtxt(source, dtype=np.int32, ndmin=2)
    return graphs


def fit_model(name, candidates):
    eligible = candidates.loc[candidates["usable_for_training"]]
    x = eligible[FEATURES].to_numpy(dtype="float64")
    y = eligible["positive"].to_numpy(dtype=np.int8)
    estimator = (
        LogisticRegression(**MODELS[name]) if name == "logistic_regression"
        else RandomForestClassifier(**MODELS[name]) if name == "random_forest"
        else KNeighborsClassifier(**MODELS[name])
    )
    pipeline = (make_pipeline(estimator) if name == "random_forest"
                else make_pipeline(StandardScaler(), estimator))
    start = perf_counter()
    pipeline.fit(x, y)
    return {
        "id": name, "features": FEATURES, "parameters": MODELS[name], "pipeline": pipeline,
        "score_kind": "candidate_positive_probability",
        "score_transform": "(31 * positive_vote_fraction + 1) / 33" if name == "knn" else "predict_proba",
    }, len(y), perf_counter() - start


def top_candidates(candidates, score):
    ranked = candidates[KEY + ["candidate_rank"]].assign(score=score)
    ranked = ranked.sort_values(["score", "candidate_rank", "arc_id"], ascending=[False, True, True])
    return ranked.drop_duplicates(OBS_KEY)[OBS_KEY + ["arc_id"]]


def evaluate(timeline, candidates, score, graphs):
    eligible = candidates["binary_eligible"].to_numpy()
    y = candidates.loc[eligible, "positive"].to_numpy(dtype=np.int8)
    p = score[eligible]
    binary = p >= THRESHOLD
    tn, fp, fn, tp = confusion_matrix(y, binary, labels=[0, 1]).ravel()
    predictions = timeline.merge(top_candidates(candidates, score), on=OBS_KEY, how="left",
                                 validate="one_to_one")
    aligned = predictions.loc[predictions["alignment_status"].eq("aligned")]
    correct = aligned["arc_id"].eq(aligned["reference_arc_id"]).fillna(False)
    continuous = transitions = 0
    for (split, trip), group in predictions.groupby(["split", "trip_id"], sort=False):
        if graphs is None:
            continue
        arcs = graphs[(split, trip)]
        predicted = group["arc_id"].fillna(-1).to_numpy(dtype=np.int64)
        pair = (
            (predicted[:-1] >= 0) & (predicted[1:] >= 0)
            & ~group["gap_before"].to_numpy()[1:] & ~group["row_break_before"].to_numpy()[1:]
        )
        prev, curr = predicted[:-1][pair], predicted[1:][pair]
        continuous += int(np.count_nonzero((prev == curr) | (arcs[prev, 1] == arcs[curr, 0])))
        transitions += len(prev)
    return {
        "aligned_observations": len(aligned),
        "ambiguous_observations": int(predictions["alignment_status"].eq("ambiguous").sum()),
        "unmatched_observations": int(predictions["reference_arc_id"].isna().sum()),
        "candidate_misses": int((~aligned["reference_in_candidates"]).sum()),
        "candidate_miss_rate": float((~aligned["reference_in_candidates"]).mean()),
        "candidate_accuracy": accuracy_score(y, binary),
        "positive_precision": precision_score(y, binary, zero_division=0),
        "positive_recall": recall_score(y, binary, zero_division=0),
        "positive_f1": f1_score(y, binary, zero_division=0),
        "macro_f1": f1_score(y, binary, labels=[0, 1], average="macro", zero_division=0),
        "candidate_log_loss": log_loss(y, p, labels=[0, 1]),
        "top1_accuracy": float(correct.mean()),
        "continuity": continuous / transitions if transitions else np.nan,
        "continuity_transitions": transitions,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def measure(model, timeline, candidates, graphs, phase, fold=None):
    start = perf_counter()
    score = score_candidates(model, candidates)
    elapsed = perf_counter() - start
    metrics = evaluate(timeline, candidates, score, graphs)
    return {
        "model": model["id"], "phase": phase, "fold": fold,
        "trips_evaluated": timeline["trip_id"].nunique(), "scored_candidates": len(score),
        "inference_seconds": elapsed, **metrics,
    }, score


def run(processed, archive, output, models=None):
    models = DEFAULT_MODELS if models is None else models
    output.mkdir(parents=True, exist_ok=True)
    train_timeline, train = load_split(processed, "train")
    test_timeline, student_test = load_split(processed, "student_test")
    graphs = load_arcs(processed, archive) if archive.is_file() else None
    if graphs is None:
        print("Raw archive unavailable: route continuity is not evaluated.", flush=True)
    train_trips = set(train_timeline["trip_id"])
    test_trips = set(test_timeline["trip_id"])
    if len(train_trips) != 80 or len(test_trips) != 10 or train_trips & test_trips:
        raise ValueError("Expected disjoint splits of 80 train trips and 10 student-test trips.")

    splits = GroupKFold(n_splits=N_FOLDS).split(train_timeline, groups=train_timeline["trip_id"])
    folds = []
    for fold, (fit, validate) in enumerate(splits, 1):
        fit_ids = set(train_timeline.iloc[fit]["trip_id"])
        val_ids = set(train_timeline.iloc[validate]["trip_id"])
        assert not (fit_ids & val_ids)
        folds.append((fold, fit_ids, val_ids))
    pd.DataFrame([
        {"trip_id": trip_id, "validation_fold": fold}
        for fold, _, validation in folds for trip_id in sorted(validation)
    ]).sort_values("trip_id").to_csv(output / "cv_folds.csv", index=False)
    config = {
        "features": FEATURES, "candidate_threshold": THRESHOLD, "seed": SEED, "folds": N_FOLDS,
        "configurations": {name: MODELS[name] for name in models},
        "continuity_available": graphs is not None,
        "versions": {name: version(name) for name in
                     ("scikit-learn", "numpy", "pandas", "scipy", "joblib", "pyarrow")},
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    results = []
    for name in models:
        for fold, fit_ids, val_ids in folds:
            model, n_fit, fit_seconds = fit_model(name, train.loc[train["trip_id"].isin(fit_ids)])
            val = train.loc[train["trip_id"].isin(val_ids)]
            metrics, _ = measure(model, train_timeline.loc[train_timeline["trip_id"].isin(val_ids)],
                                 val, graphs, "cv", fold)
            results.append({**metrics, "fit_seconds": fit_seconds, "fit_candidates": n_fit})
            print(f"{name} fold {fold}: macro F1={metrics['macro_f1']:.4f}, "
                  f"top-1={metrics['top1_accuracy']:.4f}", flush=True)
        model, n_fit, fit_seconds = fit_model(name, train)
        fitted_trips = train.loc[train["usable_for_training"], "trip_id"].nunique()
        print(f"{name} fit {n_fit:,} candidates from {fitted_trips} trips", flush=True)
        joblib.dump(model, output / f"{name}.joblib", compress=3)
        for phase, timeline, rows in (
            ("train_fit", train_timeline, train), ("student_test", test_timeline, student_test),
        ):
            metrics, score = measure(model, timeline, rows, graphs, phase)
            results.append({**metrics, "fit_seconds": fit_seconds, "fit_candidates": n_fit})
            if phase == "student_test":
                rows[KEY].assign(score=score).to_parquet(
                    output / f"{name}_student_scores.parquet", index=False, compression="zstd"
                )
            print(f"{name} {phase}: macro F1={metrics['macro_f1']:.4f}, "
                  f"top-1={metrics['top1_accuracy']:.4f}", flush=True)
        pd.DataFrame(results).to_csv(output / "metrics.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--archive", type=Path, default=Path(os.environ.get(
        "PATHSNAP_DATA_ZIP", ROOT / "data" / "raw" / "Student folder.zip",
    )))
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/baselines")
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=DEFAULT_MODELS)
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        run(args.processed_dir, args.archive, args.output, args.models)


if __name__ == "__main__":
    main()
