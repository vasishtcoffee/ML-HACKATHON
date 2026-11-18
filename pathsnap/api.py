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
from pathsnap.evaluator import available_evaluator_trips, load_evaluator_trip
from pathsnap.evaluation import evaluate
from pathsnap.scoring import DISPLAY_NAMES, score_candidates
from pathsnap.routing import Position, RoadGraph
from scripts.view_trip import load_trip, render_trip

DATA = Path(__file__).resolve().parents[1] / "data"
OBSERVATIONS = DATA / "processed/observations.csv.gz"
ARCHIVE = Path(os.environ.get("PATHSNAP_DATA_ZIP", DATA / "raw/Student folder.zip")).expanduser()
SPLIT = "student_test"
EVALUATOR_ARCHIVE = os.environ.get("PATHSNAP_EVALUATOR_ZIP", "").strip()
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
    split: str = "student_test"


def trip_ids():
    trips = pd.read_csv(DATA / "processed/trip_summary.csv", dtype={"trip_id": str})
    return sorted(trips.loc[trips["split"].eq(SPLIT), "trip_id"].unique().tolist())


@app.get("/trips")
def list_trips():
    return {
        "trip_ids": trip_ids(), "evaluator_trip_ids": (
            available_evaluator_trips(EVALUATOR_ARCHIVE) if EVALUATOR_ARCHIVE else []
        ),
        "final_model_configured": "final" in DISPLAY_NAMES,
        "models": DISPLAY_NAMES,
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    trip_id = request.trip_id
    if request.split not in ("student_test", "evaluator"):
        raise HTTPException(422, "Unknown split; select student_test or evaluator")
    if request.model not in DISPLAY_NAMES:
        raise HTTPException(422, "Unknown scoring model")
    if request.split == "student_test":
        if trip_id not in trip_ids():
            raise HTTPException(404, "Unknown student-test trip")
    else:
        if not EVALUATOR_ARCHIVE:
            raise HTTPException(503, "PATHSNAP_EVALUATOR_ZIP must be configured on Laptop 1")
        if request.model in ("distance", "logistic_regression", "knn"):
            raise HTTPException(422, "Evaluator trips require live trained-model inference")
    logger.info("Decoding trip %s split=%s with %s", trip_id, request.split, request.model)
    start = perf_counter()

    try:
        if request.split == "student_test":
            track, nodes, arcs = load_trip(OBSERVATIONS, ARCHIVE, SPLIT, trip_id)
            rows = pq.read_table(
                DATA / "processed/candidates_student_test.parquet",
                filters=[("split", "=", SPLIT), ("trip_id", "=", trip_id)],
            ).to_pandas()
        else:
            track, nodes, arcs, rows = load_evaluator_trip(EVALUATOR_ARCHIVE, trip_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except (OSError, ValueError, EOFError) as exc:
        logger.exception("Could not load trip %s in split %s", trip_id, request.split)
        raise HTTPException(503, "Trip data invalid or unavailable: " + str(exc)) from exc

    score_start = perf_counter()
    try:
        score_name, scores = score_candidates(rows, model=request.model, trip_id=trip_id)
    except (ValueError, FileNotFoundError, OSError, ImportError, TypeError) as exc:
        logger.exception("Invalid or unavailable candidate score export for %s", trip_id)
        raise HTTPException(503, "Trained scorer unavailable or invalid: " + str(exc)) from exc
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
    evaluation = None
    if request.split == "student_test":
        try:
            labels = pq.read_table(
                DATA / "processed/reference_labels_student_test.parquet",
                filters=[("split", "=", SPLIT), ("trip_id", "=", trip_id)],
                columns=["obs_id", "alignment_status", "reference_arc_id"],
            ).to_pandas()
        except (OSError, ValueError) as exc:
            logger.exception("Could not load student-test labels for %s", trip_id)
            raise HTTPException(503, "Student-test evaluation labels unavailable") from exc
        if not labels.obs_id.is_unique or set(labels.obs_id) != set(track.obs_id):
            raise HTTPException(503, "Evaluation labels do not match trip observations")
        evaluation = evaluate(labels, candidates, result.arcs_by_observation)
    map_start = perf_counter()
    trip_map, _ = render_trip(track, nodes, arcs, request.split, trip_id,
                              route_segments=result.segments)
    map_html = trip_map.get_root().render()
    map_seconds = perf_counter() - map_start
    runtime = round(perf_counter() - start, 3)
    logger.info("Trip %s model=%s decoded in %.3fs (routing %.3fs, cache hits %d, map %.3fs)",
                trip_id, request.model, decode_seconds, graph.route_seconds,
                graph.shortest_cache_hits, map_seconds)

    return {
        "trip_id": trip_id,
        "split": request.split,
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
        "accuracy": evaluation["viterbi_accuracy"] if evaluation is not None else None,
        "evaluation": evaluation,
        "map_html": map_html,
    }
