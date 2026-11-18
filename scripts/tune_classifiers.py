"""Bayesian tuning of PathSnap Logistic Regression, KNN, and MLP."""
import argparse
from dataclasses import dataclass
from importlib.metadata import version
import json
import os
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KDTree
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from score_baseline import neighbor_probabilities
from train_baselines import FEATURES, KEY, OBS_KEY, ROOT, SEED, load_arcs, load_split, measure


MODELS = ("logistic_regression", "knn", "mlp")
NEIGHBORS = [7, 15, 23, 31, 41, 51]
NEIGHBOR_POOL = max(NEIGHBORS)
SHAPES = {"16x8": (16, 8), "32x16": (32, 16), "48x24": (48, 24)}
MLP_EPOCHS = [3, 5, 7, 10]
BASELINE_TRIAL = {
    "logistic_regression": {"C": 1.0, "positive_weight": 1.0},
    "knn": {"n_neighbors": 31, "weights": "uniform"},
    "mlp": {"shape": "32x16", "epochs": 10, "alpha": 0.0001, "learning_rate": 0.001},
}


@dataclass
class Fold:
    x_fit: np.ndarray
    y_fit: np.ndarray
    x_val: np.ndarray
    positive: np.ndarray
    starts: np.ndarray
    total: int

    def top1(self, scores):
        sizes = np.diff(self.starts, append=len(scores))
        best = np.repeat(np.maximum.reduceat(scores, self.starts), sizes)
        indexes = np.arange(len(scores))
        winners = np.minimum.reduceat(np.where(scores == best, indexes, len(scores)), self.starts)
        return float(self.positive[winners].sum() / self.total)


def prepare_folds(timeline, rows):
    folds, trip_assignments = [], []
    cv = GroupKFold(3)
    for fold, (train_idx, val_idx) in enumerate(cv.split(timeline, groups=timeline["trip_id"]), 1):
        train_ids = timeline.iloc[train_idx]["trip_id"].unique()
        val = timeline.iloc[val_idx]
        val_ids = val["trip_id"].unique()
        fit = rows.loc[rows["trip_id"].isin(train_ids) & rows["usable_for_training"]]
        validation = rows.loc[
            rows["trip_id"].isin(val_ids) & rows["alignment_status"].eq("aligned")
        ].sort_values(OBS_KEY + ["candidate_rank", "arc_id"])
        start_mask = ~validation.duplicated(OBS_KEY)
        scaler = StandardScaler()
        folds.append(Fold(
            scaler.fit_transform(fit[FEATURES].to_numpy(dtype="float64")),
            fit["positive"].to_numpy(dtype=np.uint8),
            scaler.transform(validation[FEATURES].to_numpy(dtype="float64")),
            validation["positive"].to_numpy(dtype=bool),
            np.flatnonzero(start_mask),
            int(val["alignment_status"].eq("aligned").sum()),
        ))
        trip_assignments.extend((trip_id, fold) for trip_id in sorted(val_ids))
    return folds, pd.DataFrame(trip_assignments, columns=["trip_id", "validation_fold"])


def suggest(trial, name):
    if name == "logistic_regression":
        return {
            "C": trial.suggest_float("C", 0.001, 100.0, log=True),
            "positive_weight": trial.suggest_categorical("positive_weight", [1.0, 2.0, 4.0, 8.0]),
        }
    if name == "knn":
        return {
            "n_neighbors": trial.suggest_categorical("n_neighbors", NEIGHBORS),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        }
    return {
        "shape": trial.suggest_categorical("shape", list(SHAPES)),
        "epochs": trial.suggest_categorical("epochs", MLP_EPOCHS),
        "alpha": trial.suggest_float("alpha", 0.000001, 0.01, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.0003, 0.003, log=True),
    }


def fit_network_or_logistic(name, params, x, y):
    if name == "logistic_regression":
        return LogisticRegression(
            C=params["C"], class_weight={0: 1.0, 1: params["positive_weight"]},
            solver="lbfgs", max_iter=200, random_state=SEED,
        ).fit(x, y)
    model = MLPClassifier(
        hidden_layer_sizes=SHAPES[params["shape"]], activation="relu", solver="adam",
        alpha=params["alpha"], batch_size=1024, learning_rate_init=params["learning_rate"],
        early_stopping=False, random_state=SEED,
    )
    for _ in range(params["epochs"]):
        model.partial_fit(x, y, classes=[0, 1])
    return model


def knn_neighbors(folds):
    result = []
    for i, fold in enumerate(folds, 1):
        start = perf_counter()
        distances, indexes = KDTree(fold.x_fit, leaf_size=30).query(fold.x_val, k=NEIGHBOR_POOL)
        result.append((distances, fold.y_fit[indexes]))
        print(f"KNN fold {i}: indexed {len(fold.x_val):,} validation candidates in "
              f"{perf_counter() - start:.1f}s", flush=True)
    return result


def search(name, folds, output, trials):
    start = perf_counter()
    neighbors = knn_neighbors(folds) if name == "knn" else None
    cache = {}

    def objective(trial):
        params = suggest(trial, name)
        key = tuple(params.values())
        if key not in cache:
            scores = []
            for i, fold in enumerate(folds):
                if neighbors is None:
                    estimator = fit_network_or_logistic(name, params, fold.x_fit, fold.y_fit)
                    prediction = estimator.predict_proba(fold.x_val)[:, 1]
                else:
                    distances, targets = neighbors[i]
                    prediction = neighbor_probabilities(
                        distances, targets, params["n_neighbors"], params["weights"]
                    )
                scores.append(fold.top1(prediction))
            cache[key] = scores
            trial.set_user_attr("cached", False)
        else:
            trial.set_user_attr("cached", True)
        trial.set_user_attr("fold_top1", cache[key])
        return float(np.mean(cache[key]))

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=3)
    )
    study.enqueue_trial(BASELINE_TRIAL[name])
    study.optimize(objective, n_trials=trials, callbacks=[
        lambda study, trial: study.trials_dataframe().to_csv(output / "search_trials.csv", index=False)
    ])
    selection = {
        "selected_trial": study.best_trial.number,
        "selected_parameters": study.best_trial.params,
        "baseline_trial_cv_top1": study.trials[0].value,
        "selected_trial_cv_top1": study.best_value,
        "trials_completed": len(study.trials),
        "distinct_configurations": len(cache),
        "search_seconds": perf_counter() - start,
    }
    (output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    return selection["selected_parameters"]


def refit(name, params, rows):
    fit = rows.loc[rows["usable_for_training"]]
    x = fit[FEATURES].to_numpy(dtype="float64")
    y = fit["positive"].to_numpy(dtype=np.uint8)
    start = perf_counter()
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    artifact = {
        "id": name, "features": FEATURES, "parameters": params,
        "score_kind": "candidate_positive_probability",
    }
    if name == "knn":
        artifact.update(
            scaler=scaler, neighbor_index=KDTree(x, leaf_size=30), neighbor_targets=y,
            neighbor_pool=NEIGHBOR_POOL,
            score_transform="smoothed positive vote over the selected prefix of a fixed neighbor pool",
        )
    else:
        estimator = fit_network_or_logistic(name, params, x, y)
        artifact.update(pipeline=Pipeline([("standardscaler", scaler), ("classifier", estimator)]),
                        score_transform="predict_proba")
    return artifact, len(y), perf_counter() - start


def run(processed, archive, output, trials):
    output.mkdir(parents=True, exist_ok=True)
    timeline, training = load_split(processed, "train")
    if timeline["trip_id"].nunique() != 80:
        raise ValueError("Expected 80 training trips.")
    folds, assignments = prepare_folds(timeline, training)
    assignments.to_csv(output / "search_folds.csv", index=False)
    protocol = {
        "objective": "mean three-fold aligned observation top-1 with candidate misses counted",
        "sampler": "Optuna TPE", "startup_trials": 3, "trials_per_model": trials, "seed": SEED,
        "knn_fixed_neighbor_pool": NEIGHBOR_POOL, "features": FEATURES,
        "student_test_used_for_selection": False, "threads": 4,
        "dependencies": {name: version(name) for name in
                         ("optuna", "scikit-learn", "numpy", "pandas", "joblib", "pyarrow")},
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    selected = {}
    for name in MODELS:
        model_dir = output / name
        model_dir.mkdir(exist_ok=True)
        print(f"Optimizing {name}", flush=True)
        selected[name] = search(name, folds, model_dir, trials)
    del folds

    test_timeline, test = load_split(processed, "student_test")
    if test_timeline["trip_id"].nunique() != 10 or set(timeline["trip_id"]) & set(test_timeline["trip_id"]):
        raise ValueError("Expected 10 disjoint student-test trips.")
    arcs = load_arcs(processed, archive)
    comparison = []
    for name, params in selected.items():
        model_dir = output / name
        artifact, fitted, elapsed = refit(name, params, training)
        result, scores = measure(artifact, test_timeline, test, arcs, "student_test")
        result.update(fit_candidates=fitted, fit_seconds=elapsed)
        pd.DataFrame([result]).to_csv(model_dir / "metrics.csv", index=False)
        joblib.dump(artifact, model_dir / f"{name}.joblib", compress=3)
        test[KEY].assign(score=scores).to_parquet(
            model_dir / f"{name}_student_scores.parquet", index=False, compression="zstd"
        )
        baseline_dir = "mlp" if name == "mlp" else "baselines"
        baseline = pd.read_csv(ROOT / "experiments" / baseline_dir / "metrics.csv")
        untuned = baseline.loc[baseline["model"].eq(name) & baseline["phase"].eq("student_test")].iloc[0]
        for metrics, variant in ((untuned, "untuned"), (result, "tuned")):
            comparison.append({
                "model": name, "variant": variant, **{metric: metrics[metric] for metric in (
                    "top1_accuracy", "macro_f1", "candidate_accuracy", "positive_precision",
                    "positive_recall", "positive_f1", "candidate_log_loss", "continuity",
                    "candidate_misses", "fit_seconds", "inference_seconds",
                )},
            })
        print(f"{name} student top-1: {untuned['top1_accuracy']:.4f} -> "
              f"{result['top1_accuracy']:.4f}", flush=True)
    pd.DataFrame(comparison).to_csv(output / "student_test_comparison.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--archive", type=Path, default=Path(os.environ.get(
        "PATHSNAP_DATA_ZIP", ROOT / "data/raw/Student folder.zip",
    )))
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/classifier_tuning")
    args = parser.parse_args()
    if args.trials < 4:
        parser.error("--trials must be at least 4 to include a TPE trial after startup.")
    with threadpool_limits(limits=4):
        run(args.processed_dir, args.archive, args.output, args.trials)


if __name__ == "__main__":
    main()
