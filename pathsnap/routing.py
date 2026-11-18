"""Directed shortest paths between positions on numbered road arcs."""

from dataclasses import dataclass
from heapq import heappop, heappush
from math import asin, cos, inf, isfinite, radians, sin, sqrt


@dataclass(frozen=True)
class Position:
    arc_id: int
    fraction: float


@dataclass(frozen=True)
class Route:
    distance_m: float
    arc_ids: tuple[int, ...]


def distance_m(a, b):
    lon1, lat1 = map(radians, a)
    lon2, lat2 = map(radians, b)
    h = sin((lat2 - lat1) / 2)**2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2)**2
    return 2 * 6_371_008.8 * asin(sqrt(min(1.0, max(0.0, h))))


class RoadGraph:
    """Nodes are (longitude, latitude), and arc IDs are original row indices."""

    def __init__(self, nodes, arcs):
        self.arcs = tuple((int(u), int(v)) for u, v in arcs)
        self.outgoing = [[] for _ in nodes]
        self.lengths = []
        for arc_id, (u, v) in enumerate(self.arcs):
            if not (0 <= u < len(nodes) and 0 <= v < len(nodes)):
                raise ValueError("Arc endpoint outside node table")
            self.lengths.append(distance_m(nodes[u], nodes[v]))
            self.outgoing[u].append((v, arc_id))

    def _shortest(self, source, target, limit):
        best = {source: 0.0}
        previous = {}
        queue = [(0.0, source)]
        while queue:
            cost, node = heappop(queue)
            if cost != best[node]:
                continue
            if node == target:
                arcs = []
                while node != source:
                    node, arc_id = previous[node]
                    arcs.append(arc_id)
                return Route(cost, tuple(reversed(arcs)))
            for neighbor, arc_id in self.outgoing[node]:
                candidate = cost + self.lengths[arc_id]
                if candidate <= limit and candidate < best.get(neighbor, inf):
                    best[neighbor] = candidate
                    previous[neighbor] = node, arc_id
                    heappush(queue, (candidate, neighbor))
        return None

    def route(self, start, end, max_distance_m=inf):
        for position in (start, end):
            if not 0 <= position.arc_id < len(self.arcs):
                raise ValueError("Invalid arc ID")
            if not isfinite(position.fraction) or not 0 <= position.fraction <= 1:
                raise ValueError("Arc fraction must be between 0 and 1")
        a, b = start.arc_id, end.arc_id
        if a == b and start.fraction <= end.fraction:
            distance = (end.fraction - start.fraction) * self.lengths[a]
            return Route(distance, (a,)) if distance <= max_distance_m else None

        tails = (1 - start.fraction) * self.lengths[a] + end.fraction * self.lengths[b]
        if tails > max_distance_m:
            return None
        middle = self._shortest(self.arcs[a][1], self.arcs[b][0], max_distance_m - tails)
        if middle is None:
            return None
        return Route(tails + middle.distance_m, (a, *middle.arc_ids, b))
