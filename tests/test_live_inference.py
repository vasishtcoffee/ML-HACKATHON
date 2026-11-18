"""Live final-model inference and evaluator trip smoke tests.

These run against the trusted saved joblib artifacts and a synthetic
evaluator archive with NO reference file. We never inspect evaluator GT.
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from fastapi.testclient import TestClient

from pathsnap import api
from pathsnap.scoring import live_model_scores, match_exported_scores


class LiveScorerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = pq.read_table(
            api.DATA / "processed/candidates_student_test.parquet",
            filters=[("split", "=", "student_test"), ("trip_id", "=", "00000052")],
        ).to_pandas()

    def test_live_tuned_models_reproduce_committed_scores(self):
        examples = {
            "live_logistic_regression": "experiments/classifier_tuning/logistic_regression/logistic_regression_student_scores.parquet",
            "live_knn": "experiments/classifier_tuning/knn/knn_student_scores.parquet",
            "live_mlp": "experiments/classifier_tuning/mlp/mlp_student_scores.parquet",
            "live_xgboost": "experiments/xgboost_tuned/xgboost_student_scores.parquet",
            "live_random_forest": "experiments/random_forest/random_forest_student_scores.parquet",
        }
        for model, path in examples.items():
            with self.subTest(model=model):
                actual = live_model_scores(self.rows, model)
                exported = pq.read_table(
                    api.DATA.parent / path,
                    filters=[("trip_id", "=", "00000052")],
                    columns=["split", "trip_id", "obs_id", "arc_id", "score"],
                ).to_pandas()
                expected = match_exported_scores(self.rows, exported)
                np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-7)

    def test_disallows_untrusted_model_names(self):
        with self.assertRaises(ValueError):
            live_model_scores(self.rows, "/tmp/untrusted.joblib")

    def test_invalid_candidate_features_fail_closed(self):
        invalid = self.rows.drop(columns=["candidate_rank"])
        with self.assertRaisesRegex(ValueError, "Candidate rows missing fields"):
            live_model_scores(invalid, "live_mlp")


class EvaluatorAPITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(api.app)

    def test_fake_evaluator_archive_no_labels_needed(self):
        with tempfile.TemporaryDirectory() as td:
            archive_path = Path(td) / "evaluator.zip"
            with ZipFile(archive_path, "w") as z:
                base = "evaluator_batch/test123/test123"
                z.writestr(base + ".track", "0.0002 0.0 0\n0.0008 0.0 1\n")
                z.writestr(base + ".nodes", "0.0 0.0\n0.001 0.0\n")
                z.writestr(base + ".arcs", "0 1\n")
                z.writestr(base + ".route", "THIS MUST NEVER BE READ")
            with patch.object(api, "EVALUATOR_ARCHIVE", str(archive_path)):
                listed = self.client.get("/trips")
                self.assertEqual(listed.status_code, 200, listed.text[:400])
                self.assertEqual(listed.json()["evaluator_trip_ids"], ["test123"])
                refused = self.client.post("/predict", json={
                    "split": "evaluator", "trip_id": "test123", "model": "knn",
                })
                self.assertEqual(refused.status_code, 422)
                prediction = self.client.post("/predict", json={
                    "split": "evaluator", "trip_id": "test123", "model": "live_mlp",
                })
                self.assertEqual(prediction.status_code, 200, prediction.text[:1200])
                result = prediction.json()
                self.assertEqual(result["split"], "evaluator")
                self.assertEqual(result["scoring_model"], "live_mlp")
                self.assertEqual(result["observations"], 2)
                self.assertIsNone(result["evaluation"])
                self.assertIsNone(result["accuracy"])
                self.assertIn("Decoded route", result["map_html"])
                self.assertNotIn("THIS MUST NEVER BE READ", str(result))
                self.assertGreater(result["timings"]["map_render_seconds"], 0)

    def test_unconfigured_evaluator_is_not_exposed(self):
        with patch.object(api, "EVALUATOR_ARCHIVE", ""):
            response = self.client.get("/trips")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["evaluator_trip_ids"], [])
            reply = self.client.post("/predict", json={
                "split": "evaluator", "trip_id": "123", "model": "live_mlp",
            })
            self.assertEqual(reply.status_code, 503)


if __name__ == "__main__":
    unittest.main()
