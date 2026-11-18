"""Build causal GPS direction and motion features for directed road candidates."""
from zipfile import ZipFile

import numpy as np
import pandas as pd

if __package__:
    from .generate_candidates import EARTH_RADIUS_M, longitude_delta, read_graph
    from .score_baseline import KEY
else:
    from generate_candidates import EARTH_RADIUS_M, longitude_delta, read_graph
    from score_baseline import KEY


HEADING = ["heading_agreement"]
MOTION = ["projected_progress", "log_speed"]


def motion_features(track, candidates, nodes, arcs):
    if len(track) == 0:
        raise ValueError("Cannot compute features for an empty GPS track.")
    gps = track[["longitude", "latitude"]].to_numpy(dtype=float)
    lon_scale = EARTH_RADIUS_M * np.cos(np.deg2rad(np.median(gps[:, 1])))

    def metric(coords):
        return np.deg2rad(np.column_stack((
            longitude_delta(coords[:, 0], gps[0, 0]), coords[:, 1] - gps[0, 1]
        ))) * [lon_scale, EARTH_RADIUS_M]

    xy = metric(gps)
    indexes = np.arange(len(track))
    boundaries = track["gap_before"].to_numpy() | track["row_break_before"].to_numpy()
    segment_start = np.maximum.accumulate(np.where(boundaries, indexes, 0))
    anchor = np.maximum(segment_start, indexes - 3)
    delta = xy - xy[anchor]
    distance = np.linalg.norm(delta, axis=1)
    direction = delta / np.maximum(distance, 12.0)[:, None]
    previous = np.maximum(segment_start, indexes - 1)
    step = np.linalg.norm(xy - xy[previous], axis=1)

    obs_index = pd.Index(track["obs_id"]).get_indexer(candidates["obs_id"])
    if (obs_index < 0).any():
        raise ValueError("Candidate refers to an unknown GPS observation.")
    endpoints = metric(nodes)[arcs[candidates["arc_id"].to_numpy()]]
    vectors = endpoints[:, 1] - endpoints[:, 0]
    length = np.linalg.norm(vectors, axis=1)
    unit = vectors / np.maximum(length, np.finfo(float).eps)[:, None]
    heading = np.einsum("ij,ij->i", unit, direction[obs_index])

    old_offset = np.einsum("ij,ij->i", xy[previous[obs_index]] - endpoints[:, 0], unit)
    old_offset = np.clip(old_offset, 0, length)
    new_offset = candidates["fraction_along_arc"].to_numpy() * length
    progress = (new_offset - old_offset) / np.maximum(step[obs_index], 8.0)
    progress[previous[obs_index] == obs_index] = 0
    speed = track["speed_mps"].fillna(0).clip(0, 65).to_numpy()
    return candidates[KEY].assign(
        heading_agreement=heading.clip(-1, 1),
        projected_progress=progress.clip(-1, 1),
        log_speed=np.log1p(speed[obs_index]),
    )


def add_features(processed, archive_path, split, candidates):
    obs = pd.read_csv(processed / "observations.csv.gz", dtype={"trip_id": str})
    obs = obs.loc[obs["split"].eq(split)].sort_values(["trip_id", "timestamp_s", "obs_id"])
    tracks = dict(tuple(obs.groupby("trip_id")))
    inventory = pd.read_csv(processed / "archive_inventory.csv", dtype={"trip_id": str})
    members = inventory.pivot(index=["split", "trip_id"], columns="kind", values="member")
    features = []
    with ZipFile(archive_path) as archive:
        for trip, rows in candidates.groupby("trip_id", sort=False):
            nodes, arcs = read_graph(archive, members.loc[(split, trip)])
            features.append(motion_features(tracks[trip], rows, nodes, arcs))
    return candidates.merge(pd.concat(features), on=KEY, validate="one_to_one")
