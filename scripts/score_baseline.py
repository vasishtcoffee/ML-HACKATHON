"""Score a candidate table with a fitted PathSnap baseline."""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


KEY = ["split", "trip_id", "obs_id", "arc_id"]
FEATURES = [
    "distance_to_arc_m", "candidate_rank", "fraction_along_arc", "offset_from_arc_start_m",
]


def neighbor_probabilities(distances, labels, k, weighting):
    labels = labels[:, :k]
    if weighting == "uniform":
        positive_votes = labels.sum(axis=1)
    else:
        distances = distances[:, :k]
        zero = distances == 0
        weights = np.divide(1.0, distances, out=np.zeros_like(distances), where=~zero)
        weights[zero.any(axis=1)] = zero[zero.any(axis=1)]
        positive_votes = k * (weights * labels).sum(axis=1) / weights.sum(axis=1)
    return (positive_votes + 1) / (k + 2)


def score_candidates(model, candidates):
    x = candidates.loc[:, model["features"]].to_numpy(dtype="float64")
    if "neighbor_index" in model:
        distances, indexes = model["neighbor_index"].query(
            model["scaler"].transform(x), k=model["neighbor_pool"]
        )
        params = model["parameters"]
        return neighbor_probabilities(
            distances, model["neighbor_targets"][indexes], params["n_neighbors"], params["weights"]
        )
    score = model["pipeline"].predict_proba(x)[:, 1]
    if model["id"] == "knn":
        k = model["parameters"]["n_neighbors"]
        score = (k * score + 1) / (k + 2)
    return score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Trusted PathSnap joblib model")
    parser.add_argument("candidates", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    model = joblib.load(args.model)
    candidates = pd.read_parquet(args.candidates, columns=KEY + model["features"])
    with threadpool_limits(limits=4):
        score = score_candidates(model, candidates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    candidates[KEY].assign(score=score).to_parquet(args.output, index=False, compression="zstd")


if __name__ == "__main__":
    main()
