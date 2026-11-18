"""Compare heading and motion features using trip-grouped training validation."""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from motion_features import HEADING, MOTION, add_features
from score_baseline import FEATURES, KEY, neighbor_probabilities
from train_baselines import OBS_KEY, ROOT, SEED, load_arcs, load_split, measure
from tune_classifiers import MODELS, NEIGHBOR_POOL, SHAPES, Fold, fit_network_or_logistic


SETS = {"heading": FEATURES + HEADING, "motion": FEATURES + HEADING + MOTION}
KNN_OBSERVATIONS = 16000


def sample_knn(fitting):
    groups = fitting[OBS_KEY].drop_duplicates()
    # Ensure each fitting trip contributes observations to the capped index.
    anchors = groups.drop_duplicates(["split", "trip_id"])
    remaining = groups.drop(anchors.index)
    sample = remaining.sample(n=min(len(remaining), KNN_OBSERVATIONS - len(anchors)),
                              random_state=SEED)
    return fitting.merge(pd.concat([anchors, sample]), on=OBS_KEY, validate="many_to_one")


def configs(prior):
    return {
        "logistic_regression": [
            prior["logistic_regression"],
            {"C": 1.0, "positive_weight": 1.0},
            {"C": 1.0, "positive_weight": 2.0},
        ],
        "knn": [
            {"n_neighbors": k, "weights": weight}
            for k in (15, 31, 41, 51) for weight in ("uniform", "distance")
        ],
        "mlp": [{**prior["mlp"], "epochs": epochs} for epochs in (5, 10)],
    }


def evaluate_options(name, settings, features, fit, validation, total):
    if name == "knn":
        fit = sample_knn(fit)
    target = fit["positive"].to_numpy(dtype=np.uint8)
    scaler = StandardScaler()
    x_fit = scaler.fit_transform(fit[features].to_numpy(dtype=float))
    x_val = scaler.transform(validation[features].to_numpy(dtype=float))
    ranking = Fold(x_fit, target, x_val, validation["positive"].to_numpy(dtype=bool),
                   np.flatnonzero(~validation.duplicated(OBS_KEY)), total)
    if name == "knn":
        distances, indices = KDTree(x_fit, leaf_size=30).query(x_val, k=NEIGHBOR_POOL)
        votes = target[indices]
        return [
            ranking.top1(neighbor_probabilities(distances, votes, p["n_neighbors"], p["weights"]))
            for p in settings
        ]
    if name == "mlp":
        p = settings[-1]
        estimator = MLPClassifier(
            hidden_layer_sizes=SHAPES[p["shape"]], activation="relu", solver="adam",
            alpha=p["alpha"], batch_size=1024, learning_rate_init=p["learning_rate"],
            early_stopping=False, random_state=SEED,
        )
        scores = []
        milestones = {s["epochs"] for s in settings}
        for epoch in range(1, p["epochs"] + 1):
            estimator.partial_fit(x_fit, target, classes=[0, 1])
            if epoch in milestones:
                scores.append(ranking.top1(estimator.predict_proba(x_val)[:, 1]))
        return scores
    return [
        ranking.top1(fit_network_or_logistic(name, p, x_fit, target).predict_proba(x_val)[:, 1])
        for p in settings
    ]


def fit_selected(name, params, features, rows):
    fitting = rows.loc[rows["usable_for_training"]]
    if name == "knn":
        fitting = sample_knn(fitting)
    target = fitting["positive"].to_numpy(dtype=np.uint8)
    start = perf_counter()
    scaler = StandardScaler()
    x = scaler.fit_transform(fitting[features].to_numpy(dtype=float))
    model = {
        "id": name, "features": features, "parameters": params,
        "score_kind": "candidate_positive_probability",
    }
    if name == "knn":
        model.update(scaler=scaler, neighbor_index=KDTree(x, leaf_size=30),
                     neighbor_targets=target, neighbor_pool=NEIGHBOR_POOL,
                     score_transform="smoothed vote over selected fixed-pool neighbors")
    else:
        estimator = fit_network_or_logistic(name, params, x, target)
        model.update(pipeline=Pipeline([("standardscaler", scaler), ("classifier", estimator)]),
                     score_transform="predict_proba")
    return model, len(fitting), int(target.sum()), perf_counter() - start


def run(processed, archive, prior_dir, output):
    output.mkdir(parents=True, exist_ok=True)
    timeline, candidates = load_split(processed, "train")
    train = add_features(processed, archive, "train", candidates)
    del candidates
    assignments = pd.read_csv(prior_dir / "search_folds.csv", dtype={"trip_id": str})
    ids = set(timeline["trip_id"])
    if len(ids) != 80 or not assignments.trip_id.is_unique or set(assignments.trip_id) != ids:
        raise ValueError("Expected the original trip-disjoint 80-trip search assignment.")
    assignments.to_csv(output / "search_folds.csv", index=False)
    prior = {
        name: json.loads((prior_dir / name / "selection.json").read_text())["selected_parameters"]
        for name in MODELS
    }
    options = configs(prior)
    selected = {}
    heldout = []
    protocol = {
        "features": SETS, "settings": options, "seed": SEED,
        "knn_fit_observation_cap": KNN_OBSERVATIONS,
        "objective": "mean aligned observation top-1 over the prior three train folds",
        "versions": {p: version(p) for p in ("scikit-learn", "numpy", "pandas", "pyarrow", "joblib")},
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    for name, settings in options.items():
        for variant, features in SETS.items():
            for fold, trips in assignments.groupby("validation_fold"):
                hold = train["trip_id"].isin(trips.trip_id)
                fitting = train.loc[~hold & train["usable_for_training"]]
                validate = train.loc[hold & train["alignment_status"].eq("aligned")].sort_values(
                    OBS_KEY + ["candidate_rank", "arc_id"]
                )
                total = int((timeline["trip_id"].isin(trips.trip_id)
                             & timeline["alignment_status"].eq("aligned")).sum())
                scores = evaluate_options(name, settings, features, fitting, validate, total)
                heldout.extend({
                    "model": name, "features": variant, "configuration": config,
                    "fold": fold, "aligned": total, "top1": score,
                } for config, score in enumerate(scores))
                pd.DataFrame(heldout).to_csv(output / "validation_trials.csv", index=False)
                print(f"{name} {variant} fold {fold}: best top-1 {max(scores):.2%}", flush=True)
        summary = pd.DataFrame(heldout).query("model == @name").groupby(
            ["features", "configuration"]
        ).top1.mean()
        variant, slot = summary.idxmax()
        selected[name] = {
            "variant": variant, "parameters": settings[slot], "validation_top1": float(summary.max()),
        }
        print(f"{name} selected: {selected[name]}", flush=True)
    (output / "selection.json").write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")

    test_timeline, test = load_split(processed, "student_test")
    if test_timeline.trip_id.nunique() != 10 or ids & set(test_timeline.trip_id):
        raise ValueError("Expected ten held-out student trips.")
    test = add_features(processed, archive, "student_test", test)
    arcs = load_arcs(processed, archive)
    results = []
    for name, choice in selected.items():
        artifact, fitted, observations, elapsed = fit_selected(
            name, choice["parameters"], SETS[choice["variant"]], train
        )
        metrics, scores = measure(artifact, test_timeline, test, arcs, "student_test")
        results.append({
            **metrics, "feature_set": choice["variant"], "fit_candidates": fitted,
            "fit_observations": observations, "fit_seconds": elapsed,
        })
        joblib.dump(artifact, output / f"{name}.joblib", compress=3)
        test[KEY].assign(score=scores).to_parquet(
            output / f"{name}_student_scores.parquet", index=False, compression="zstd"
        )
        print(f"{name} student top-1: {metrics['top1_accuracy']:.2%}", flush=True)
    pd.DataFrame(results).to_csv(output / "metrics.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--archive", type=Path, default=Path(os.environ.get(
        "PATHSNAP_DATA_ZIP", ROOT / "data/raw/Student folder.zip"
    )))
    parser.add_argument("--prior", type=Path, default=ROOT / "experiments/classifier_tuning")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/motion_features")
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        run(args.processed_dir, args.archive, args.prior, args.output)


if __name__ == "__main__":
    main()
