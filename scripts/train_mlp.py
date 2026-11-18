"""Train and evaluate the PathSnap MLP on the published trip folds."""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from train_baselines import FEATURES, KEY, ROOT, SEED, THRESHOLD, load_arcs, load_split, measure


EPOCHS = 10
PARAMETERS = {
    "hidden_layer_sizes": (32, 16),
    "activation": "relu",
    "solver": "adam",
    "alpha": 0.0001,
    "batch_size": 1024,
    "learning_rate_init": 0.001,
    "early_stopping": False,
    "random_state": SEED,
}


def fit_mlp(candidates, validation=None, *, fold=None):
    fit = candidates.loc[candidates["usable_for_training"]]
    x = fit[FEATURES].to_numpy(dtype=np.float64)
    y = fit["positive"].to_numpy(dtype=np.int8)
    if validation is not None:
        valid = validation.loc[validation["binary_eligible"]]
        val_x = valid[FEATURES].to_numpy(dtype=np.float64)
        val_y = valid["positive"].to_numpy(dtype=np.int8)
    scaler = StandardScaler()
    network = MLPClassifier(**PARAMETERS)
    start = perf_counter()
    x = scaler.fit_transform(x)
    fit_seconds = perf_counter() - start
    if validation is not None:
        val_x = scaler.transform(val_x)
    history = []
    for epoch in range(1, EPOCHS + 1):
        start = perf_counter()
        network.partial_fit(x, y, classes=[0, 1])
        elapsed = perf_counter() - start
        fit_seconds += elapsed
        history.append({
            "fold": fold,
            "epoch": epoch,
            "train_objective": float(network.loss_),
            "validation_log_loss": (
                log_loss(val_y, network.predict_proba(val_x)[:, 1], labels=[0, 1])
                if validation is not None else np.nan
            ),
            "epoch_fit_seconds": elapsed,
        })
        print(f"fold {fold or 'refit'} epoch {epoch}/{EPOCHS}: "
              f"objective={network.loss_:.5f}", flush=True)
    return {
        "id": "mlp", "features": FEATURES, "parameters": {**PARAMETERS, "epochs": EPOCHS},
        "pipeline": Pipeline([("standardscaler", scaler), ("mlpclassifier", network)]),
        "score_kind": "candidate_positive_probability",
        "score_transform": "predict_proba",
    }, len(y), fit_seconds, history


def run(processed, archive, fold_file, output):
    output.mkdir(parents=True, exist_ok=True)
    timeline, train = load_split(processed, "train")
    test_timeline, student_test = load_split(processed, "student_test")
    graphs = load_arcs(processed, archive)
    train_ids = set(timeline["trip_id"])
    test_ids = set(test_timeline["trip_id"])
    folds = pd.read_csv(fold_file, dtype={"trip_id": str})
    if (len(train_ids) != 80 or len(test_ids) != 10 or train_ids & test_ids
            or folds["trip_id"].duplicated().any() or set(folds["trip_id"]) != train_ids
            or set(folds["validation_fold"]) != set(range(1, 6))):
        raise ValueError("Expected the published five-fold assignment and disjoint 80/10 trip splits.")
    folds.to_csv(output / "cv_folds.csv", index=False)
    configuration = {
        "model": "mlp", "features": FEATURES, "candidate_threshold": THRESHOLD,
        "epochs": EPOCHS, "parameters": PARAMETERS,
        "versions": {name: version(name) for name in
                     ("scikit-learn", "numpy", "pandas", "scipy", "joblib", "pyarrow")},
    }
    (output / "config.json").write_text(json.dumps(configuration, indent=2) + "\n", encoding="utf-8")
    metrics, history = [], []
    for fold, trips in folds.groupby("validation_fold"):
        held_out = trips["trip_id"]
        keep = train["trip_id"].isin(held_out)
        validation = train.loc[keep]
        model, fitted, elapsed, epochs = fit_mlp(train.loc[~keep], validation, fold=int(fold))
        row, _ = measure(model, timeline.loc[timeline["trip_id"].isin(held_out)],
                         validation, graphs, "cv", int(fold))
        metrics.append({**row, "fit_seconds": elapsed, "fit_candidates": fitted})
        history.extend(epochs)
        print(f"fold {fold}: macro F1={row['macro_f1']:.4f}, top-1={row['top1_accuracy']:.4f}",
              flush=True)

    model, fitted, elapsed, epochs = fit_mlp(train)
    history.extend(epochs)
    joblib.dump(model, output / "mlp.joblib", compress=3)
    for phase, observations, candidates in (
        ("train_fit", timeline, train), ("student_test", test_timeline, student_test),
    ):
        row, score = measure(model, observations, candidates, graphs, phase)
        metrics.append({**row, "fit_seconds": elapsed, "fit_candidates": fitted})
        if phase == "student_test":
            candidates[KEY].assign(score=score).to_parquet(
                output / "mlp_student_scores.parquet", index=False, compression="zstd"
            )
        print(f"{phase}: macro F1={row['macro_f1']:.4f}, top-1={row['top1_accuracy']:.4f}",
              flush=True)
    pd.DataFrame(metrics).to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(history).to_csv(output / "epochs.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--archive", type=Path, default=Path(os.environ.get(
        "PATHSNAP_DATA_ZIP", ROOT / "data" / "raw" / "Student folder.zip",
    )))
    parser.add_argument("--folds", type=Path, default=ROOT / "experiments/baselines/cv_folds.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/mlp")
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        run(args.processed_dir, args.archive, args.folds, args.output)


if __name__ == "__main__":
    main()
