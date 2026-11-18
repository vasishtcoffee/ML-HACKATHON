import argparse
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
EARTH_RADIUS_M = 6_371_008.8
OBSERVATION_KEY = ["split", "trip_id", "obs_id"]


def longitude_delta(longitude, origin):
    return (longitude - origin + 180) % 360 - 180


def project_candidates(track, nodes, arcs, radius_m):
    gps = track[["longitude", "latitude"]].to_numpy()
    origin_lon, origin_lat = gps[0]
    x_scale = EARTH_RADIUS_M * np.cos(np.deg2rad(np.median(gps[:, 1])))
    if x_scale < 1:
        raise ValueError("Cannot use a local equirectangular projection at the pole.")

    def to_metres(coords):
        return np.column_stack((
            x_scale * np.deg2rad(longitude_delta(coords[:, 0], origin_lon)),
            EARTH_RADIUS_M * np.deg2rad(coords[:, 1] - origin_lat),
        ))

    points = to_metres(gps)
    endpoints = to_metres(nodes)[arcs]
    start = endpoints[:, 0]
    vectors = endpoints[:, 1] - start
    arc_length_squared = np.einsum("ij,ij->i", vectors, vectors)
    midpoints = start + vectors / 2
    half_lengths = np.sqrt(arc_length_squared) / 2
    length_bands = np.floor(np.log2(np.maximum(half_lengths, 1)))

    matches = []
    for band in np.unique(length_bands):
        arc_ids = np.flatnonzero(length_bands == band)
        tree = cKDTree(midpoints[arc_ids])
        midpoint_radius = radius_m + half_lengths[arc_ids].max()
        # The triangle inequality bounds every segment within the search radius.
        neighbours = tree.query_ball_point(points, midpoint_radius + 1e-8)
        counts = np.fromiter(map(len, neighbours), dtype=np.int64, count=len(points))
        if not counts.any():
            continue

        obs_index = np.repeat(np.arange(len(points)), counts)
        arc_index = arc_ids[np.concatenate(neighbours[counts > 0], dtype=np.int64)]
        from_start = points[obs_index] - start[arc_index]
        squared = arc_length_squared[arc_index]
        fraction = np.clip(np.divide(
            np.einsum("ij,ij->i", from_start, vectors[arc_index]),
            squared,
            out=np.zeros(len(obs_index)),
            where=squared > 0,
        ), 0, 1)
        projected = start[arc_index] + fraction[:, None] * vectors[arc_index]
        distance = np.linalg.norm(points[obs_index] - projected, axis=1)
        nearby = distance <= radius_m
        matches.append(pd.DataFrame({
            "obs_id": track["obs_id"].to_numpy()[obs_index[nearby]],
            "arc_id": arc_index[nearby],
            "fraction_along_arc": fraction[nearby],
            "distance_to_arc_m": distance[nearby],
            "projected_longitude": longitude_delta(
                origin_lon + np.rad2deg(projected[nearby, 0] / x_scale), 0
            ).round(8),
            "projected_latitude": (
                origin_lat + np.rad2deg(projected[nearby, 1] / EARTH_RADIUS_M)
            ).round(8),
        }))

    columns = {
        "obs_id": "int64", "arc_id": "int64", "fraction_along_arc": "float64",
        "distance_to_arc_m": "float64", "projected_longitude": "float64",
        "projected_latitude": "float64",
    }
    candidates = (
        pd.concat(matches, ignore_index=True)
        if matches else pd.DataFrame({name: pd.Series(dtype=dtype) for name, dtype in columns.items()})
    )
    candidates.sort_values(["obs_id", "distance_to_arc_m", "arc_id"], inplace=True)
    candidates["candidate_rank"] = candidates.groupby("obs_id").cumcount() + 1

    arc_ids = candidates["arc_id"].to_numpy()
    lon1, lat1 = np.deg2rad(nodes[arcs[arc_ids, 0]]).T
    lon2, lat2 = np.deg2rad(nodes[arcs[arc_ids, 1]]).T
    hav = np.sin((lat2 - lat1) / 2) ** 2 + (
        np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    lengths = 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(hav, 0, 1)))
    candidates["offset_from_arc_start_m"] = candidates["fraction_along_arc"] * lengths
    return candidates


def read_graph(archive, members):
    with archive.open(members["nodes"]) as source:
        nodes = np.loadtxt(source, dtype=float, ndmin=2)
    with archive.open(members["arcs"]) as source:
        arcs = np.loadtxt(source, dtype=np.int64, ndmin=2)
    if nodes.shape[1] != 2 or not np.isfinite(nodes).all():
        raise ValueError(f"Invalid node coordinates in {members['nodes']}")
    if arcs.shape[1] != 2 or (arcs < 0).any() or (arcs >= len(nodes)).any():
        raise ValueError(f"Invalid directed arc endpoints in {members['arcs']}")
    return nodes, arcs


def export_candidates(archive_path, processed_dir, radius_m):
    observations = pd.read_csv(
        processed_dir / "observations.csv.gz", usecols=OBSERVATION_KEY + ["longitude", "latitude"],
        dtype={"trip_id": str},
    )
    inventory = pd.read_csv(processed_dir / "archive_inventory.csv", dtype={"trip_id": str})
    graphs = inventory.loc[inventory["kind"].isin(["nodes", "arcs"])].pivot(
        index=["split", "trip_id"], columns="kind", values="member",
    )

    coverage = []
    with ZipFile(archive_path) as archive, TemporaryDirectory(dir=processed_dir) as temp:
        stage = Path(temp)
        for split in ("train", "student_test"):
            split_tracks = observations.loc[observations["split"].eq(split)]
            schema = pa.schema([
                ("split", pa.string()), ("trip_id", pa.string()),
                ("obs_id", pa.int64()), ("arc_id", pa.int64()), ("candidate_rank", pa.int64()),
                ("fraction_along_arc", pa.float64()), ("offset_from_arc_start_m", pa.float64()),
                ("projected_longitude", pa.float64()), ("projected_latitude", pa.float64()),
                ("distance_to_arc_m", pa.float64()),
            ], metadata={b"search_radius_m": str(radius_m).encode()})
            with pq.ParquetWriter(stage / f"candidates_{split}.parquet", schema,
                                  compression="zstd") as writer:
                for trip_id, track in split_tracks.groupby("trip_id", sort=True):
                    nodes, arcs = read_graph(archive, graphs.loc[(split, trip_id)])
                    rows = project_candidates(track, nodes, arcs, radius_m)
                    rows["split"] = split
                    rows["trip_id"] = trip_id
                    writer.write_table(pa.Table.from_pandas(rows, schema=schema, preserve_index=False))
                    coverage.append({
                        "split": split,
                        "trip_id": trip_id,
                        "search_radius_m": radius_m,
                        "observations": len(track),
                        "observations_without_candidates": len(track) - rows["obs_id"].nunique(),
                        "candidate_rows": len(rows),
                    })
        report = pd.DataFrame(coverage)
        report.to_csv(stage / "candidate_coverage.csv", index=False, lineterminator="\n")
        for artifact in stage.iterdir():
            artifact.replace(processed_dir / artifact.name)
    print(report.groupby("split")[[
        "observations", "observations_without_candidates", "candidate_rows"
    ]].sum().to_string())


def main():
    parser = argparse.ArgumentParser(description="Generate directed road arc candidates.")
    parser.add_argument("--archive", type=Path, default=Path(os.environ.get(
        "PATHSNAP_DATA_ZIP", ROOT / "data" / "raw" / "Student folder.zip",
    )))
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--radius-m", type=float, default=50.0)
    args = parser.parse_args()
    if not np.isfinite(args.radius_m) or args.radius_m <= 0:
        parser.error("--radius-m must be positive and finite.")
    export_candidates(args.archive, args.processed_dir, args.radius_m)


if __name__ == "__main__":
    main()
