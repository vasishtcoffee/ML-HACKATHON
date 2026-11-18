"""Regression tests for trip-local bounded shortest-path cache and exact route equality."""
import unittest
from pathsnap.routing import RoadGraph, Position
from pathsnap.decoder import Candidate, Observation, decode


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.nodes = [(0.001*i, 0.0) for i in range(5)]
        self.arcs = [(0,1),(1,2),(2,3),(3,4)]
        self.raw = RoadGraph(self.nodes,self.arcs)
        self.cached = RoadGraph(self.nodes,self.arcs,shortest_cache_size=10)
        self.start=Position(0,0.5)
        self.end=Position(3,0.5)

    def test_failure_does_not_poison_larger_bound(self):
        short = self.cached.route(self.start,self.end,20)
        self.assertIsNone(short)
        actual=self.cached.route(self.start,self.end,10000)
        expected=self.raw.route(self.start,self.end,10000)
        self.assertEqual(actual,expected)
        self.assertIsNotNone(actual)

    def test_cache_success_checks_new_distance_limit(self):
        route=self.cached.route(self.start,self.end,10000)
        assert route is not None
        self.assertIsNone(self.cached.route(self.start,self.end,route.distance_m-0.1))
        self.assertEqual(self.cached.route(self.start,self.end,route.distance_m+1),route)
        self.assertGreaterEqual(self.cached.shortest_cache_hits,1)

    def test_smaller_failed_bounds_reuse_proof(self):
        self.cached.route(self.start,self.end,220)
        before=self.cached.shortest_searches
        self.cached.route(self.start,self.end,180)
        self.assertEqual(before,self.cached.shortest_searches)
        self.assertGreater(self.cached.shortest_cache_hits,0)

    def test_decoder_outputs_identical_with_cache(self):
        points = [
            Observation(0,0,(0.0005,0.0),(Candidate(Position(0,.5),.9,1),Candidate(Position(3,.5),.1,2))),
            Observation(1,1,(0.0015,0.0),(Candidate(Position(1,.5),.8,1),Candidate(Position(3,.5),.95,2))),
            Observation(2,2,(0.0025,0.0),(Candidate(Position(2,.5),.8,1),Candidate(Position(0,.5),.9,2))),
        ]
        self.assertEqual(decode(self.raw,points),decode(self.cached,points))


if __name__=="__main__":
    unittest.main()
