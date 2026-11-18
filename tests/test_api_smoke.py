"""Local FastAPI development smoke test with actual committed student-test trip.

Run: python -m unittest discover -s tests -v
Requires fastapi and httpx in addition to the processing dependencies.
"""

import unittest
from fastapi.testclient import TestClient
from pathsnap.api import app, SHORTEST_CACHE_SIZE


class FastAPIDemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_models_and_ten_student_trip_ids(self):
        response = self.client.get("/trips")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data["trip_ids"]), 10)
        self.assertEqual(len(set(data["trip_ids"])), 10)
        self.assertIn("knn", data["models"])
        self.assertIn("logistic_regression", data["models"])

    def test_knn_prediction_and_separate_timing(self):
        response = self.client.post("/predict", json={"trip_id": "00000052", "model": "knn"})
        self.assertEqual(response.status_code, 200, response.text[:400])
        data = response.json()
        self.assertEqual(data["scoring_model"], "knn")
        self.assertEqual(data["observations"], 262)
        self.assertEqual(data["evaluation"]["aligned"], 203)
        self.assertEqual(data["evaluation"]["viterbi_accuracy"], 1.0)
        self.assertGreater(len(data["map_html"]), 1000)
        self.assertIn("Decoded route", data["map_html"])
        self.assertEqual(data["timings"]["cache_size"], SHORTEST_CACHE_SIZE)
        self.assertGreater(data["timings"]["road_routing_seconds"], 0)
        self.assertGreater(data["timings"]["map_render_seconds"], 0)
        self.assertGreaterEqual(data["runtime_seconds"], data["timings"]["map_render_seconds"])

    def test_invalid_trip_rejected(self):
        response = self.client.post("/predict", json={"trip_id": "fake", "model": "knn"})
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
