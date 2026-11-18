"""Evaluate both the prior and enhanced classifier exports with one Viterbi policy."""
import argparse
import os
from pathlib import Path

import pandas as pd

from benchmark_trained_decoding import ROOT, one_trip, summarize


MODEL_NAMES = ("logistic_regression", "knn", "mlp")
# Fixed in the existing train-only decoder calibration.
DECODER = {"beta_m": 60.0, "gap_seconds": 10.0}


def run_variant(variant, processed, archive, prior, output):
    meta = pd.read_csv(processed / "trip_summary.csv", dtype={"trip_id": str})
    trip_ids = sorted(meta.loc[meta["split"].eq("student_test"), "trip_id"].unique())
    if len(trip_ids) != 10:
        raise ValueError("Expected the ten student-test trips.")
    results = []
    for name in MODEL_NAMES:
        score_dir = prior / name if variant == "prior" else output
        score_file = score_dir / f"{name}_student_scores.parquet"
        for trip_id in trip_ids:
            row = one_trip(
                name, trip_id, processed, archive, processed / "candidates_student_test.parquet",
                processed / "reference_labels_student_test.parquet", score_file,
                decoder_kwargs=DECODER, cache_size=20000,
            )
            results.append(row)
            print(f"{variant} {name} {trip_id}: top-1={row['top1_accuracy']:.2%}, "
                  f"Viterbi={row['viterbi_accuracy']:.2%}", flush=True)
    pd.DataFrame(results).to_csv(output / f"viterbi_{variant}_per_trip.csv", index=False)
    pd.DataFrame(summarize(results, MODEL_NAMES)).to_csv(
        output / f"viterbi_{variant}_summary.csv", index=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--archive", type=Path, default=Path(os.environ.get(
        "PATHSNAP_DATA_ZIP", ROOT / "data/raw/Student folder.zip"
    )))
    parser.add_argument("--prior", type=Path, default=ROOT / "experiments/classifier_tuning")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/motion_features")
    parser.add_argument("--variant", choices=("prior", "enhanced", "both"), default="both")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    variants = ("prior", "enhanced") if args.variant == "both" else (args.variant,)
    for variant in variants:
        run_variant(variant, args.processed_dir, args.archive, args.prior, args.output)


if __name__ == "__main__":
    main()
