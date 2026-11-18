"""Regression coverage for directed road geometry and Viterbi decoding.

Run: python -m unittest discover -s tests -v

Tiny synthetic graphs exercise route choices; no reference labels or hidden
evaluator trips are used.
"""

import unittest

from pathsnap.decoder import Candidate, Observation, decode
from pathsnap.routing import Position, RoadGraph, distance_m


class DirectedRoutingTests(unittest.TestCase):
    def setUp(self):
        # Two components: 0 -> 1 -> 2 -> 3, and 4 -> 5.
        self.nodes = [(0.001 * i, 0.0) for i in range(6)]
        self.graph = RoadGraph(self.nodes, [(0, 1), (1, 2), (2, 3), (4, 5)])

    def test_forward_arc_and_distance(self):
        route = self.graph.route(Position(0, 0.25), Position(0, 0.75))
        self.assertEqual(route.arc_ids, (0,))
        self.assertAlmostEqual(route.distance_m, self.graph.lengths[0] * 0.5)

    def test_directed_shortest_path_includes_intermediate_arc(self):
        route = self.graph.route(Position(0, 0.5), Position(2, 0.5))
        self.assertEqual(route.arc_ids, (0, 1, 2))
        self.assertAlmostEqual(
            route.distance_m,
            0.5 * self.graph.lengths[0] + self.graph.lengths[1]
            + 0.5 * self.graph.lengths[2],
        )

    def test_reverse_direction_is_not_invented(self):
        self.assertIsNone(self.graph.route(Position(2, 0.5), Position(0, 0.5)))
        self.assertIsNone(self.graph.route(Position(0, 0.75), Position(0, 0.25)))

    def test_disconnected_and_distance_limited_paths(self):
        self.assertIsNone(self.graph.route(Position(0, 0.5), Position(3, 0.5)))
        self.assertIsNone(self.graph.route(Position(0, 0.5), Position(2, 0.5), 10.0))

    def test_rejects_invalid_arc_and_fraction(self):
        with self.assertRaisesRegex(ValueError, "Invalid arc"):
            self.graph.route(Position(99, 0), Position(0, 0))
        with self.assertRaisesRegex(ValueError, "fraction"):
            self.graph.route(Position(0, -0.1), Position(0, 0))

    def test_haversine_distance_is_metric(self):
        self.assertAlmostEqual(distance_m(self.nodes[0], self.nodes[1]), 111.195, delta=0.1)


class DecoderTests(unittest.TestCase):
    def setUp(self):
        self.nodes = [(0.001 * i, 0.0) for i in range(6)]
        self.graph = RoadGraph(self.nodes, [(0, 1), (1, 2), (2, 3), (4, 5)])

    def obs(self, index, timestamp, *candidates, lon=None):
        return Observation(
            index, float(timestamp),
            (0.001 * index if lon is None else lon, 0.0),
            tuple(Candidate(Position(arc, 0.5), probability, rank)
                  for arc, probability, rank in candidates),
        )

    def test_viterbi_prefers_connected_route_over_higher_single_score(self):
        # The high-scoring option on arc 3 is unreachable from arc 1.
        observations = [
            self.obs(0, 0, (0, 0.9, 1), (3, 0.1, 2), lon=0.0005),
            self.obs(1, 1, (3, 0.99, 1), (1, 0.8, 2), lon=0.0015),
        ]
        result = decode(self.graph, observations)
        self.assertEqual(result.arcs_by_observation, {0: 0, 1: 1})
        self.assertEqual(result.segments, ((0, 1),))

    def test_large_timestamp_gap_splits_sequences(self):
        result = decode(self.graph, [
            self.obs(0, 0, (0, 0.9, 1)),
            self.obs(1, 10, (1, 0.8, 1)),
        ])
        self.assertEqual(result.arcs_by_observation, {0: 0, 1: 1})
        self.assertEqual(result.segments, ((0,), (1,)))

    def test_missing_candidates_create_missing_prediction(self):
        result = decode(self.graph, [
            self.obs(0, 0, (0, 0.9, 1)),
            self.obs(1, 1),
            self.obs(2, 2, (2, 0.9, 1)),
        ])
        self.assertEqual(result.arcs_by_observation, {0: 0, 1: None, 2: 2})
        self.assertEqual(result.segments, ((0,), (2,)))

    def test_disconnected_transition_starts_new_segment(self):
        result = decode(self.graph, [
            self.obs(0, 0, (0, 0.8, 1)),
            self.obs(1, 1, (3, 0.9, 1)),
        ])
        self.assertEqual(result.segments, ((0,), (3,)))
        self.assertEqual(result.arcs_by_observation, {0: 0, 1: 3})

    def test_non_monotonic_timestamp_rejected(self):
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            decode(self.graph, [
                self.obs(0, 5, (0, 0.7, 1)),
                self.obs(1, 5, (1, 0.8, 1)),
            ])

    def test_bad_probability_rejected(self):
        with self.assertRaisesRegex(ValueError, "probabilities"):
            decode(self.graph, [self.obs(0, 0, (0, 1.1, 1))])


if __name__ == "__main__":
    unittest.main()
