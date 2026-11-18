"""Development map matching API. Run with uvicorn pathsnap.api:app."""

import logging
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pathsnap.decoder import Candidate, Observation, decode
from pathsnap.evaluation import evaluate
from pathsnap.scoring import DISPLAY_NAMES, score_candidates
from pathsnap.routing import Position, RoadGraph
from scripts.view_trip import load_trip, render_trip

DATA = Path(__file__).resolve().parents[1] / "data"
OBSERVATIONS = DATA / "processed/observations.csv.gz"
ARCHIVE = Path(os.environ.get("PATHSNAP_DATA_ZIP", DATA / "raw/Student folder.zip")).expanduser()
SPLIT = "student_test"
# Per-trip bounded cache; 0 disables it to reproduce the uncached baseline.
SHORTEST_CACHE_SIZE = int(os.environ.get("PATHSNAP_SHORTEST_CACHE_SIZE", "20000"))
if SHORTEST_CACHE_SIZE < 0:
    raise ValueError("PATHSNAP_SHORTEST_CACHE_SIZE must be nonnegative")
logger = logging.getLogger("uvicorn.error")
app = FastAPI(title="PathSnap development API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.environ.get("PATHSNAP_UI_ORIGIN", "http://localhost:8001")],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class PredictionRequest(BaseModel):
    trip_id: str
    model: str = "distance"


def trip_ids():
    trips = pd.read_csv(DATA / "processed/trip_summary.csv", dtype={"trip_id": str})
    return sorted(trips.loc[trips["split"].eq(SPLIT), "trip_id"].unique().tolist())


@app.get("/trips")
def list_trips():
    return {"trip_ids": trip_ids(), "models": DISPLAY_NAMES}


@app.post("/predict")
def predict(request: PredictionRequest):
    trip_id = request.trip_id
    if trip_id not in trip_ids():
        raise HTTPException(404, "Unknown student-test trip")
    if request.model not in DISPLAY_NAMES:
        raise HTTPException(422, "Unknown scoring model")
    logger.info("Decoding development trip %s with %s", trip_id, request.model)
    start = perf_counter()

    try:
        track, nodes, arcs = load_trip(OBSERVATIONS, ARCHIVE, SPLIT, trip_id)
        rows = pq.read_table(
            DATA / "processed/candidates_student_test.parquet",
            filters=[("trip_id", "=", trip_id)],
        ).to_pandas()
    except (OSError, ValueError):
        logger.exception("Could not load inputs for trip %s", trip_id)
        raise HTTPException(503, "Trip data could not be loaded. Check the server setup.")

    score_start = perf_counter()
    try:
        score_name, scores = score_candidates(rows, model=request.model, trip_id=trip_id)
    except (ValueError, FileNotFoundError, OSError) as exc:
        logger.exception("Invalid or unavailable candidate score export for %s", trip_id)
        raise HTTPException(503, "Candidate score export unavailable or invalid: " + str(exc)) from exc
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (len(rows),) or not np.all((scores >= 0) & (scores <= 1)):
        raise HTTPException(500, "Scorer must return one finite score in [0, 1] per candidate")

    score_seconds = perf_counter() - score_start
    candidates = {}
    for row, score in zip(rows.itertuples(index=False), scores, strict=True):
        candidates.setdefault(row.obs_id, []).append(
            Candidate(Position(row.arc_id, row.fraction_along_arc), score, int(row.candidate_rank))
        )
    observations = [
        Observation(row.obs_id, row.timestamp_s, (row.longitude, row.latitude),
                    tuple(candidates.get(row.obs_id, ())))
        for row in track.itertuples(index=False)
    ]
    graph = RoadGraph(nodes, arcs, shortest_cache_size=SHORTEST_CACHE_SIZE)
    decode_start = perf_counter()
    result = decode(graph, observations)
    decode_seconds = perf_counter() - decode_start
    try:
        labels = pq.read_table(
            DATA / "processed/reference_labels_student_test.parquet",
            filters=[("trip_id", "=", trip_id)],
            columns=["obs_id", "alignment_status", "reference_arc_id"],
        ).to_pandas()
    except (OSError, ValueError):
        logger.exception("Could not load evaluation labels for trip %s", trip_id)
        raise HTTPException(503, "Evaluation labels could not be loaded.")
    if not labels.obs_id.is_unique or set(labels.obs_id) != set(track.obs_id):
        raise HTTPException(503, "Evaluation labels do not match the trip observations.")
    evaluation = evaluate(labels, candidates, result.arcs_by_observation)
    map_start = perf_counter()
    trip_map, _ = render_trip(track, nodes, arcs, SPLIT, trip_id,
                              route_segments=result.segments)
    map_html = trip_map.get_root().render()
    map_seconds = perf_counter() - map_start
    runtime = round(perf_counter() - start, 3)
    logger.info("Trip %s model=%s decoded in %.3fs (routing %.3fs, cache hits %d, map %.3fs)",
                trip_id, request.model, decode_seconds, graph.route_seconds,
                graph.shortest_cache_hits, map_seconds)

    return {
        "trip_id": trip_id,
        "scoring": score_name,
        "scoring_model": request.model,
        "arcs_by_observation": result.arcs_by_observation,
        "route_segments": result.segments,
        "matched_observations": sum(arc is not None for arc in result.arcs_by_observation.values()),
        "observations": len(observations),
        "runtime_seconds": runtime,
        "timings": {
            "score_load_seconds": round(score_seconds, 4),
            "decode_seconds": round(decode_seconds, 4),
            "road_routing_seconds": round(graph.route_seconds, 4),
            "dijkstra_seconds": round(graph.shortest_search_seconds, 4),
            "map_render_seconds": round(map_seconds, 4),
            "total_server_seconds": runtime,
            "shortest_path_searches": graph.shortest_searches,
            "shortest_cache_hits": graph.shortest_cache_hits,
            "cache_size": SHORTEST_CACHE_SIZE,
        },
        "accuracy": evaluation["viterbi_accuracy"],
        "evaluation": evaluation,
        "map_html": map_html,
    }
