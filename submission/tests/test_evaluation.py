"""Regression tests for exactly matching the baseline top-1 ranking policy."""

import unittest
import pandas as pd

from pathsnap.decoder import Candidate
from pathsnap.evaluation import evaluate
from pathsnap.routing import Position


def option(arc, score, rank):
    return Candidate(Position(arc, 0.5), score, rank)


class EvaluationTests(unittest.TestCase):
    def test_tied_probability_prefers_candidate_rank_over_arc_id(self):
        # Highest arc ID wins because it has the lower (better) candidate rank.
        labels = pd.DataFrame({
            "obs_id": [0], "alignment_status": ["aligned"], "reference_arc_id": [9],
        })
        candidates = {0: [option(2, 0.75, 2), option(9, 0.75, 1)]}
        result = evaluate(labels, candidates, {0: 9})
        self.assertEqual(result["top1_correct"], 1)
        self.assertEqual(result["viterbi_correct"], 1)

    def test_tied_score_and_rank_prefers_smaller_arc_id(self):
        labels = pd.DataFrame({
            "obs_id": [0], "alignment_status": ["aligned"], "reference_arc_id": [2],
        })
        result = evaluate(
            labels, {0: [option(9, 0.75, 1), option(2, 0.75, 1)]}, {0: 2},
        )
        self.assertEqual(result["top1_correct"], 1)

    def test_missed_reference_counts_against_both(self):
        labels = pd.DataFrame({
            "obs_id": [0, 1, 2, 3],
            "alignment_status": ["aligned", "aligned", "ambiguous", "unmatched_distance"],
            "reference_arc_id": [2, 9, None, None],
        })
        result = evaluate(
            labels,
            {0: [option(2, 0.9, 1)], 1: [option(3, 0.9, 1)]},
            {0: 2, 1: 3},
        )
        self.assertEqual(result["aligned"], 2)
        self.assertEqual(result["candidate_misses"], 1)
        self.assertEqual(result["top1_correct"], 1)
        self.assertEqual(result["viterbi_correct"], 1)
        self.assertEqual(result["top1_accuracy"], 0.5)
        self.assertEqual(result["viterbi_accuracy"], 0.5)
        self.assertEqual(result["ambiguous_excluded"], 1)
        self.assertEqual(result["unmatched_excluded"], 1)


if __name__ == "__main__":
    unittest.main()
