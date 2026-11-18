"""Development map matching API. Run with uvicorn pathsnap.api:app."""

import logging
import os
from math import exp
from pathlib import Path
from time import perf_counter

import pandas as pd
import pyarrow.parquet as pq
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pathsnap.decoder import Candidate, Observation, decode
from pathsnap.routing import Position, RoadGraph
from scripts.view_trip import load_trip, render_trip

DATA = Path(__file__).resolve().parents[1] / "data"
OBSERVATIONS = DATA / "processed/observations.csv.gz"
ARCHIVE = Path(os.environ.get("PATHSNAP_DATA_ZIP", DATA / "raw/Student folder.zip")).expanduser()
SPLIT = "student_test"
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


def trip_ids():
    trips = pd.read_csv(DATA / "processed/trip_summary.csv", dtype={"trip_id": str})
    return sorted(trips.loc[trips["split"].eq(SPLIT), "trip_id"].unique().tolist())


@app.get("/trips")
def list_trips():
    return {"trip_ids": trip_ids()}


@app.post("/predict")
def predict(request: PredictionRequest):
    trip_id = request.trip_id
    if trip_id not in trip_ids():
        raise HTTPException(404, "Unknown student-test trip")
    logger.info("Decoding development trip %s", trip_id)
    start = perf_counter()

    try:
        track, nodes, arcs = load_trip(OBSERVATIONS, ARCHIVE, SPLIT, trip_id)
        rows = pq.read_table(
            DATA / "processed/candidates_student_test.parquet",
            filters=[("trip_id", "=", trip_id)],
            columns=["obs_id", "arc_id", "fraction_along_arc", "distance_to_arc_m"],
        ).to_pandas()
    except (OSError, ValueError):
        logger.exception("Could not load inputs for trip %s", trip_id)
        raise HTTPException(503, "Trip data could not be loaded. Check the server setup.")

    candidates = {}
    for row in rows.itertuples(index=False):
        # Replace this development score with the classifier probability at integration.
        score = exp(-0.5 * (row.distance_to_arc_m / 20.0) ** 2)
        candidates.setdefault(row.obs_id, []).append(
            Candidate(Position(row.arc_id, row.fraction_along_arc), score)
        )
    observations = [
        Observation(row.obs_id, row.timestamp_s, (row.longitude, row.latitude),
                    tuple(candidates.get(row.obs_id, ())))
        for row in track.itertuples(index=False)
    ]
    result = decode(RoadGraph(nodes, arcs), observations)
    runtime = round(perf_counter() - start, 3)
    trip_map, _ = render_trip(track, nodes, arcs, SPLIT, trip_id,
                              route_segments=result.segments)

    return {
        "trip_id": trip_id,
        "scoring": "distance_baseline",
        "arcs_by_observation": result.arcs_by_observation,
        "route_segments": result.segments,
        "matched_observations": sum(arc is not None for arc in result.arcs_by_observation.values()),
        "observations": len(observations),
        "runtime_seconds": runtime,
        "accuracy": None,
        "map_html": trip_map.get_root().render(),
    }
