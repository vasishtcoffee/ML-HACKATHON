"""Viterbi decoding over candidates scored by a road classifier."""

from dataclasses import dataclass
from math import isfinite, log

from .routing import Position, Route, distance_m


@dataclass(frozen=True)
class Candidate:
    position: Position
    probability: float
    candidate_rank: int  # one-based proximity rank from the original candidate table


@dataclass(frozen=True)
class Observation:
    obs_id: int
    timestamp_s: float
    lon_lat: tuple[float, float]
    candidates: tuple[Candidate, ...]
    break_before: bool = False


@dataclass(frozen=True)
class DecodedRoute:
    arcs_by_observation: dict[int, int | None]
    segments: tuple[tuple[int, ...], ...]


def decode(graph, observations, *, gap_seconds=5.0, beta_m=30.0,
           max_speed_mps=65.0, routing_slack_m=100.0, jitter_m=15.0):
    """Decode timestamp-ordered observations. Scores are P(match) in [0, 1].

    The transition penalizes the absolute difference between road distance and
    GPS displacement. Gaps, empty candidate sets, and unreachable transitions
    end a segment. The distance bound limits path searches and implausible jumps.
    """
    for value in (gap_seconds, beta_m, max_speed_mps, routing_slack_m):
        if not isfinite(value) or value <= 0:
            raise ValueError("Decoder parameters must be positive and finite")

    if not isfinite(jitter_m) or jitter_m < 0:
        raise ValueError("Jitter tolerance must be nonnegative and finite")

    predictions = {}
    segments = []
    layers = []

    def finish_segment():
        if not layers:
            return
        chosen = max(layers[-1][1], key=lambda candidate: layers[-1][1][candidate][0])
        path_parts = []
        for observation, states in reversed(layers):
            predictions[observation.obs_id] = chosen.position.arc_id
            _, previous, route = states[chosen]
            if previous is None:
                first_arc = chosen.position.arc_id
            else:
                path_parts.append(route.arc_ids[1:])
            chosen = previous
        segments.append((first_arc, *(arc for part in reversed(path_parts) for arc in part)))
        layers.clear()

    previous_observation = None
    for observation in observations:
        if observation.obs_id in predictions:
            raise ValueError("Observation IDs must be unique")
        if not isfinite(observation.timestamp_s):
            raise ValueError("Observation timestamps must be finite")
        dt = (observation.timestamp_s - previous_observation.timestamp_s
              if previous_observation is not None else 0.0)
        if previous_observation is not None and dt <= 0:
            raise ValueError("Observations must have strictly increasing timestamps")
        predictions[observation.obs_id] = None
        emissions = {}
        for candidate in observation.candidates:
            if not 0 <= candidate.probability <= 1:
                raise ValueError("Candidate probabilities must be in [0, 1]")
            # Validate singleton candidates even when they never need a transition.
            graph.route(candidate.position, candidate.position)
            emissions[candidate] = log(max(1e-12, candidate.probability))

        if not emissions or dt > gap_seconds or observation.break_before:
            finish_segment()
        if emissions:
            states = {}
            if layers:
                displacement = distance_m(previous_observation.lon_lat, observation.lon_lat)
                limit = max(displacement, dt * max_speed_mps) + routing_slack_m
                for candidate, emission in emissions.items():
                    for previous, (score, _, _) in layers[-1][1].items():
                        a, b = previous.position, candidate.position
                        # Projected positions carry GPS noise. Small same-arc
                        # retreats represent stationary/forward travel uncertainty,
                        # never an invented reverse directed road path.
                        retreat = ((a.fraction - b.fraction) * graph.lengths[a.arc_id]
                                   if a.arc_id == b.arc_id else float("inf"))
                        if 0 < retreat <= jitter_m:
                            route = Route(0.0, (a.arc_id,))
                        else:
                            route = graph.route(a, b, limit)
                        if route is None:
                            continue
                        total = score + emission - abs(route.distance_m - displacement) / beta_m
                        if candidate not in states or total > states[candidate][0]:
                            states[candidate] = (total, previous, route)
            if not states:
                finish_segment()
                states = {candidate: (score, None, None) for candidate, score in emissions.items()}
            layers.append((observation, states))
        previous_observation = observation

    finish_segment()
    return DecodedRoute(predictions, tuple(segments))
