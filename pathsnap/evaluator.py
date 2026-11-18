"""Read a trusted evaluator ZIP without inspecting or requiring reference labels.

The evaluator's ZIP is configured on Laptop 1 via PATHSNAP_EVALUATOR_ZIP,
not uploaded via the browser or supplied as an API parameter. Expected members:
  <optional prefix>/<trip_id>/<trip_id>.track
  <optional prefix>/<trip_id>/<trip_id>.nodes
  <optional prefix>/<trip_id>/<trip_id>.arcs

The outer directory can have any name. Each .track has longitude latitude
timestamp_s; .nodes has longitude latitude; .arcs has directed 0-based node
indices. This matches the student-data geometry conventions.
"""

from pathlib import Path
import re
from zipfile import ZipFile

import numpy as np
import pandas as pd

from scripts.generate_candidates import project_candidates

_SUFFIX = re.compile(r"(?:^|/)([A-Za-z0-9_-]+)/\1\.(track|nodes|arcs)$")
_MAX_MEMBERS = 300
_MAX_ARCHIVE_MEMBER_BYTES = 75_000_000
_MAX_NODES = 400_000
_MAX_ARCS = 800_000
_MAX_OBSERVATIONS = 100_000


def _members(archive):
    if len(archive.infolist()) > _MAX_MEMBERS:
        raise ValueError("Too many archive entries")
    found = {}
    for item in archive.infolist():
        if item.file_size > _MAX_ARCHIVE_MEMBER_BYTES:
            raise ValueError("Evaluator archive member exceeds size limit")
        match = _SUFFIX.search(item.filename)
        if not match:
            continue  # .route and other ground-truth files are NEVER loaded
        trip_id, kind = match.groups()
        if trip_id in found and kind in found[trip_id]:
            raise ValueError(f"Duplicate evaluator file for {trip_id}/{kind}")
        found.setdefault(trip_id, {})[kind] = item.filename
    found = {k: v for k, v in found.items() if all(t in v for t in ("track", "nodes", "arcs"))}
    return found


def available_evaluator_trips(archive_path):
    path = Path(archive_path)
    if not path.is_file():
        raise FileNotFoundError(f"Evaluator archive not found at configured path: {path}")
    with ZipFile(path) as archive:
        return sorted(_members(archive))


def _load_whitespace(archive, name, width, dtype, limit):
    with archive.open(name) as source:
        array = np.loadtxt(source, ndmin=2, dtype=dtype)
    if array.ndim != 2 or array.shape[1] != width or not 0 < len(array) <= limit:
        raise ValueError(f"Invalid evaluator table shape: {name}")
    if not np.isfinite(array).all():
        raise ValueError(f"Evaluator table contains non-finite entries: {name}")
    return array


def load_evaluator_trip(archive_path, trip_id):
    """Return (track, nodes, arcs, candidate_rows) without ground truth.

    Candidate features are computed using exactly the training generator's
    50 m radius and geometry conventions; original node/arc indices remain.
    """
    if not isinstance(trip_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", trip_id):
        raise ValueError("Invalid evaluator trip ID")
    with ZipFile(Path(archive_path)) as archive:
        entries = _members(archive)
        if trip_id not in entries:
            raise KeyError(f"Unknown evaluator trip ID: {trip_id}")
        paths = entries[trip_id]
        raw = _load_whitespace(archive, paths["track"], 3, float, _MAX_OBSERVATIONS)
        nodes = _load_whitespace(archive, paths["nodes"], 2, float, _MAX_NODES)
        arcs = _load_whitespace(archive, paths["arcs"], 2, np.int64, _MAX_ARCS)
    if not (np.abs(nodes[:, 0]) <= 180).all() or not (np.abs(nodes[:, 1]) <= 90).all():
        raise ValueError("Invalid evaluator node coordinates")
    if (arcs < 0).any() or (arcs >= len(nodes)).any():
        raise ValueError("Evaluator arc endpoints are outside the graph")
    gps = pd.DataFrame(raw, columns=["longitude", "latitude", "timestamp_s"])
    good = (gps.longitude.between(-180, 180) &
            gps.latitude.between(-90, 90) & np.isfinite(gps.timestamp_s))
    track = gps.loc[good].sort_values("timestamp_s", kind="stable")
    track = track.drop_duplicates("timestamp_s", keep="first").copy()
    if track.empty:
        raise ValueError("No usable evaluator GPS observations")
    track.index.name = "obs_id"
    track = track.reset_index()
    track["split"] = "evaluator"
    track["trip_id"] = trip_id
    track["gap_before"] = track.timestamp_s.diff().gt(5).fillna(False)
    track["row_break_before"] = track.obs_id.diff().ne(1) & track.timestamp_s.diff().notna()
    candidates = project_candidates(track, nodes, arcs, radius_m=50.0)
    candidates["split"] = "evaluator"
    candidates["trip_id"] = trip_id
    # No reference labels are read or made available to inference or UI.
    return track, nodes, arcs, candidates
