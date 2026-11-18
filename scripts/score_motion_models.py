"""Regenerate direction and motion features and score a split without labels."""
import argparse
import os
from pathlib import Path

import joblib
import pandas as pd
from threadpoolctl import threadpool_limits

from score_baseline import FEATURES, KEY, score_candidates
from generate_candidates import ROOT
from motion_features import add_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Trusted joblib artifact from train_motion_models.py")
    parser.add_argument("output", type=Path)
    parser.add_argument("--split", choices=("train", "student_test"), default="student_test")
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--archive", type=Path, default=Path(os.environ.get(
        "PATHSNAP_DATA_ZIP", ROOT / "data/raw/Student folder.zip",
    )))
    args = parser.parse_args()
    model = joblib.load(args.model)
    candidates = pd.read_parquet(
        args.processed_dir / f"candidates_{args.split}.parquet", columns=KEY + FEATURES,
    )
    enriched = add_features(args.processed_dir, args.archive, args.split, candidates)
    with threadpool_limits(limits=4):
        scores = score_candidates(model, enriched)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    enriched[KEY].assign(score=scores).to_parquet(args.output, index=False, compression="zstd")


if __name__ == "__main__":
    main()
