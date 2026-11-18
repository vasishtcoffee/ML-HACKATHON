"""Focused regression checks for the classifier-score handoff.

Run from the repository root: python -m unittest discover -s tests -v
"""
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from pathsnap.scoring import match_exported_scores, score_candidates


class KeyedScoreTests(unittest.TestCase):
    def setUp(self):
        self.candidates = pd.DataFrame({
            "split": ["student_test"] * 3,
            "trip_id": ["00000052"] * 3,
            "obs_id": [10, 10, 11],
            "arc_id": [1, 2, 1],
            "distance_to_arc_m": [10, 20, 30],
        })
        self.exported = self.candidates[["split", "trip_id", "obs_id", "arc_id"]].copy()
        self.exported["score"] = [0.25, 0.75, 0.5]

    def test_full_key_join_independent_of_row_order(self):
        shuffled = self.exported.iloc[[2, 0, 1]].reset_index(drop=True)
        np.testing.assert_allclose(match_exported_scores(self.candidates, shuffled), [0.25, 0.75, 0.5])

    def test_duplicate_score_key_rejected(self):
        duplicate = pd.concat([self.exported, self.exported.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            match_exported_scores(self.candidates, duplicate)

    def test_missing_score_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing scores"):
            match_exported_scores(self.candidates, self.exported.iloc[:2])

    def test_unexpected_score_rejected(self):
        extra = self.exported.iloc[[0]].copy()
        extra["arc_id"] = 99
        with self.assertRaisesRegex(ValueError, "unexpected scores"):
            match_exported_scores(self.candidates, pd.concat([self.exported, extra]))

    def test_invalid_probability_rejected(self):
        for invalid in [np.nan, np.inf, -0.1, 1.1]:
            bad = self.exported.copy()
            bad.loc[0, "score"] = invalid
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "finite P"):
                match_exported_scores(self.candidates, bad)

    def test_key_confusion_rejected(self):
        bad = self.exported.copy()
        bad.loc[0, "trip_id"] = "0000052"
        with self.assertRaisesRegex(ValueError, "key mismatch"):
            match_exported_scores(self.candidates, bad)

    def test_trained_scorer_uses_full_key(self):
        with tempfile.NamedTemporaryFile() as mock_file:
            with patch("pathsnap.scoring.pd.read_parquet", return_value=self.exported.iloc[::-1]) as read:
                name, values = score_candidates(
                    self.candidates, model="knn", trip_id="00000052",
                    score_files={"knn": mock_file.name},
                )
                self.assertIn("KNN", name)
                np.testing.assert_allclose(values, [0.25, 0.75, 0.5])
                self.assertEqual(read.call_args.kwargs["filters"], [("trip_id", "==", "00000052")])

    def test_distance_scorer_remains_selectable(self):
        name, values = score_candidates(self.candidates, model="distance")
        self.assertIn("Distance", name)
        self.assertEqual(len(values), 3)


if __name__ == "__main__":
    unittest.main()
