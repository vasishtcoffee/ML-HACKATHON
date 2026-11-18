"""Compare the team's saved PathSnap variants using shared student-test protocol."""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pathsnap.decoder import CALIBRATED_DECODER
from pathsnap.scoring import LIVE_MODELS
from scripts.benchmark_trained_decoding import one_trip, summarize, write_csv
from scripts.view_trip import ROOT


SOURCES = {
    "live_heading_logistic_regression": "logistic_regression",
    "live_motion_knn": "knn",
    "live_motion_mlp": "mlp",
    "live_random_forest": "random_forest",
    "live_xgboost": "xgboost",
    "live_logistic_regression": "logistic_regression",
    "live_knn": "knn",
    "live_mlp": "mlp",
    "live_tuned_random_forest": "random_forest",
    "live_baseline_logistic_regression": "logistic_regression",
    "live_baseline_knn": "knn",
    "live_baseline_mlp": "mlp",
    "live_baseline_xgboost": "xgboost",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoder", choices=("calibrated", "original"), default="calibrated")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/final_viterbi_comparison")
    parser.add_argument("--archive", type=Path, default=Path(os.environ.get(
        "PATHSNAP_DATA_ZIP", ROOT / "data/raw/Student folder.zip",
    )))
    parser.add_argument("--models", nargs="+", choices=SOURCES, default=list(SOURCES))
    args = parser.parse_args()
    data = ROOT / "data/processed"
    args.output.mkdir(parents=True, exist_ok=True)
    trip_table = pd.read_csv(data / "trip_summary.csv", dtype={"trip_id": str})
    ids = sorted(trip_table.loc[trip_table.split.eq("student_test"), "trip_id"].unique())
    if len(ids) != 10:
        raise ValueError("Expected ten student-test trips")
    decoder = CALIBRATED_DECODER if args.decoder == "calibrated" else {}

    per_trip = []
    for model in args.models:
        family = SOURCES[model]
        export = LIVE_MODELS[model].with_name(f"{family}_student_scores.parquet")
        for trip_id in ids:
            row = one_trip(
                model, trip_id, data, args.archive,
                data / "candidates_student_test.parquet",
                data / "reference_labels_student_test.parquet",
                export, decoder_kwargs=decoder, cache_size=20000,
                live=True, compare_route_pairs=True,
            )
            per_trip.append(row)
            print(f"{model} {trip_id}: top-1={row['top1_accuracy']:.2%}, "
                  f"Viterbi={row['viterbi_accuracy']:.2%}, scorer={row['score_seconds']:.2f}s", flush=True)

    aggregate = summarize(per_trip, args.models)
    for row in aggregate:
        sample = [r for r in per_trip if r["model"] == row["model"]]
        row["classifier"] = SOURCES[row["model"]]
        row["beta_m"] = decoder.get("beta_m", 30.0)
        row["gap_seconds"] = decoder.get("gap_seconds", 5.0)
        reference_pairs = sum(r["reference_route_pairs"] for r in sample)
        for name in ("top1", "viterbi"):
            predicted = sum(r[f"{name}_route_pairs"] for r in sample)
            correct = sum(r[f"{name}_route_pairs_correct"] for r in sample)
            row[f"{name}_route_pair_f1"] = 2 * correct / (predicted + reference_pairs)
    aggregate.sort(key=lambda r: r["viterbi_accuracy"], reverse=True)
    write_csv(args.output / "per_trip.csv", per_trip)
    write_csv(args.output / "aggregate.csv", aggregate)
    for row in aggregate:
        print(f"{row['model']}: {row['viterbi_correct']}/{row['aligned']} "
              f"decoded ({row['viterbi_accuracy']:.2%})", flush=True)


if __name__ == "__main__":
    main()
