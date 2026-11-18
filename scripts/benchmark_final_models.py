"""Fair comparison of five existing saved model score exports AFTER Viterbi.

Does not select the final classifier automatically: reports all ten
student-test trips, one shared graph/candidate layout and unchanged
decoder settings. A 20,000-entry per-trip routing cache changes speed,
not outputs. No evaluator data is read or used.
"""
import csv
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.benchmark_trained_decoding import one_trip, write_csv
from scripts.view_trip import ROOT

SOURCES = {
    "tuned_logistic_regression": ROOT / "experiments/classifier_tuning/logistic_regression/logistic_regression_student_scores.parquet",
    "tuned_knn": ROOT / "experiments/classifier_tuning/knn/knn_student_scores.parquet",
    "tuned_mlp": ROOT / "experiments/classifier_tuning/mlp/mlp_student_scores.parquet",
    "random_forest_baseline": ROOT / "experiments/random_forest/random_forest_student_scores.parquet",
    "tuned_xgboost": ROOT / "experiments/xgboost_tuned/xgboost_student_scores.parquet",
}


def main():
    data = ROOT / "data/processed"
    archive = ROOT / "data/raw/Student folder.zip"
    output = ROOT / "experiments/final_viterbi_comparison"
    output.mkdir(parents=True, exist_ok=True)
    trip_table = pd.read_csv(data / "trip_summary.csv", dtype={"trip_id": str})
    ids = sorted(trip_table.loc[trip_table.split.eq("student_test"), "trip_id"].unique().tolist())
    if len(ids) != 10:
        raise ValueError("Ten student-test trips are required")
    for path in [archive, *SOURCES.values()]:
        if not path.is_file():
            raise FileNotFoundError(path)

    per_trip, aggregate = [], []
    for name, score_file in SOURCES.items():
        current = []
        print("\nMODEL:", name, flush=True)
        for trip_id in ids:
            r = one_trip(
                name, trip_id, data, archive,
                data / "candidates_student_test.parquet",
                data / "reference_labels_student_test.parquet",
                score_file,
                cache_size=20000,
            )
            per_trip.append(r)
            current.append(r)
            print(f"{trip_id}: Top-1 {r['top1_accuracy']:.2%} => Viterbi {r['viterbi_accuracy']:.2%}, "
                  f"breaks {r['route_breaks']}, runtime {r['decode_seconds']:.2f}s", flush=True)
        aligned = sum(r["aligned"] for r in current)
        tp = sum(r["top1_correct"] for r in current)
        vp = sum(r["viterbi_correct"] for r in current)
        top_pairs = sum(r["top1_pairs"] for r in current)
        dec_pairs = sum(r["viterbi_pairs"] for r in current)
        aggregate.append({
            "model": name, "aligned": aligned,
            "top1_correct": tp, "viterbi_correct": vp,
            "top1_accuracy": tp / aligned,
            "viterbi_accuracy": vp / aligned,
            "gain_pp": 100 * (vp - tp) / aligned,
            "top1_continuity": sum(r["top1_continuous"] for r in current) / top_pairs,
            "viterbi_continuity": sum(r["viterbi_continuous"] for r in current) / dec_pairs,
            "route_breaks": sum(r["route_breaks"] for r in current),
            "decoder_seconds": sum(r["decode_seconds"] for r in current),
            "end_to_end_seconds_excluding_ml_inference": sum(r["end_to_end_seconds"] for r in current),
        })
        write_csv(output / "per_trip.csv", per_trip)
        write_csv(output / "aggregate.csv", aggregate)
        print("AGGREGATE", json.dumps(aggregate[-1]), flush=True)
    print("Completed five-model, fifty-trip comparison; metrics include student-test reference alignment only.",flush=True)


if __name__ == "__main__":
    main()
