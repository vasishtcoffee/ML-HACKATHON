import argparse
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import pandas as pd

import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.generate_candidates import OBSERVATION_KEY, ROOT, project_candidates, read_graph


MAX_REFERENCE_DISTANCE_M = 75.0
AMBIGUITY_MARGIN_M = 3.0
SPLITS = ("train", "student_test")


def ordered_matches(options, observation_ids, route_size):
    states = np.arange(route_size + 1)
    cost = np.full(route_size + 1, np.inf)
    cost[0] = 0
    updates = []

    for obs_id in observation_ids:
        next_cost = cost + MAX_REFERENCE_DISTANCE_M
        steps, distances = options.get(obs_id, (np.array([], dtype=int), np.array([])))
        target = steps + 1
        if len(target):
            prefix_cost = np.minimum.accumulate(cost)
            prefix_state = np.maximum.accumulate(np.where(cost == prefix_cost, states, 0))
            match_cost = prefix_cost[target] + distances
            improved = match_cost < next_cost[target]
            next_cost[target[improved]] = match_cost[improved]
            updates.append(dict(zip(target[improved], prefix_state[target[improved]])))
        else:
            updates.append({})
        cost = next_cost

    state = int(np.argmin(cost))
    matched = np.full(len(observation_ids), -1, dtype=int)
    for i in range(len(updates) - 1, -1, -1):
        if state in updates[i]:
            matched[i] = state - 1
            state = updates[i][state]
    return matched


def align_trip(track, nodes, arcs, route):
    # The candidate generator's local arc IDs now identify route occurrences.
    nearby = project_candidates(track, nodes, arcs[route], MAX_REFERENCE_DISTANCE_M)
    options = {
        obs_id: (rows["arc_id"].to_numpy(), rows["distance_to_arc_m"].to_numpy())
        for obs_id, rows in nearby.groupby("obs_id", sort=False)
    }
    obs_ids = track["obs_id"].to_numpy()
    selected = ordered_matches(options, obs_ids, len(route))
    matched_rows = np.flatnonzero(selected >= 0)

    status = np.full(len(track), "unmatched_no_nearby_reference", dtype=object)
    status[np.isin(obs_ids, list(options))] = "unmatched_route_order"
    distance = np.full(len(track), np.nan)
    crosses_break = np.zeros(len(track), dtype=bool)
    route_arcs = arcs[route]
    breaks = np.r_[0, route_arcs[:-1, 1] != route_arcs[1:, 0]].cumsum()

    for i, row in enumerate(matched_rows):
        step = selected[row]
        steps, distances = options[obs_ids[row]]
        distance[row] = distances[steps == step].item()
        lower = selected[matched_rows[i - 1]] if i else 0
        upper = selected[matched_rows[i + 1]] if i + 1 < len(matched_rows) else len(route) - 1
        alternatives = (steps >= lower) & (steps <= upper) & (route[steps] != route[step])
        ambiguous = np.any(distances[alternatives] <= distance[row] + AMBIGUITY_MARGIN_M)
        crosses_break[row] = bool(i and breaks[step] > breaks[lower])
        status[row] = "ambiguous" if ambiguous else "aligned"

    route_step = pd.array(selected, dtype="Int64")
    arc_id = pd.array(route[selected], dtype="Int64")
    route_step[selected < 0] = pd.NA
    arc_id[selected < 0] = pd.NA
    return track[OBSERVATION_KEY].assign(
        reference_arc_id=arc_id,
        reference_route_step=route_step,
        alignment_status=status,
        alignment_distance_m=distance,
        crosses_reference_break=crosses_break,
    )


def export_labels(archive_path, processed_dir):
    observations = pd.read_csv(processed_dir / "observations.csv.gz", dtype={"trip_id": str})
    inventory = pd.read_csv(processed_dir / "archive_inventory.csv", dtype={"trip_id": str})
    members = inventory.pivot(index=["split", "trip_id"], columns="kind", values="member")
    coverage = []
    with ZipFile(archive_path) as archive, TemporaryDirectory(dir=processed_dir) as temp:
        stage = Path(temp)
        for split in SPLITS:
            trips = []
            for trip_id, track in observations.loc[observations["split"].eq(split)].groupby("trip_id"):
                entries = members.loc[(split, trip_id)]
                nodes, arcs = read_graph(archive, entries)
                with archive.open(entries["route"]) as source:
                    route = np.loadtxt(source, dtype=np.int64, ndmin=2)
                if route.shape[1] != 1 or not len(route) or (route < 0).any() or (route >= len(arcs)).any():
                    raise ValueError(f"Invalid reference arc sequence in {entries['route']}")
                trips.append(align_trip(track, nodes, arcs, route[:, 0]))

            labels = pd.concat(trips, ignore_index=True)
            candidates = pd.read_parquet(
                processed_dir / f"candidates_{split}.parquet", columns=OBSERVATION_KEY + ["arc_id"]
            )
            label_keys = pd.MultiIndex.from_frame(labels[OBSERVATION_KEY + ["reference_arc_id"]])
            candidate_keys = pd.MultiIndex.from_frame(candidates[OBSERVATION_KEY + ["arc_id"]])
            labels["reference_in_candidates"] = label_keys.isin(candidate_keys)
            labels["has_candidates"] = pd.MultiIndex.from_frame(labels[OBSERVATION_KEY]).isin(
                candidate_keys.droplevel("arc_id")
            )
            labels["usable_for_training"] = (
                labels["split"].eq("train")
                & labels["alignment_status"].eq("aligned")
                & labels["reference_in_candidates"]
            )
            labels.to_parquet(stage / f"reference_labels_{split}.parquet", index=False, compression="zstd")
            audited = labels.assign(
                aligned=labels["alignment_status"].eq("aligned"),
                ambiguous=labels["alignment_status"].eq("ambiguous"),
                unmatched=labels["reference_arc_id"].isna(),
                zero_candidates=~labels["has_candidates"],
                aligned_arc_missing=labels["alignment_status"].eq("aligned") & ~labels["reference_in_candidates"],
                aligned_with_other_candidates=(
                    labels["alignment_status"].eq("aligned") & labels["has_candidates"]
                    & ~labels["reference_in_candidates"]
                ),
            )
            coverage.append(audited.groupby(["split", "trip_id"]).agg(
                observations=("obs_id", "size"), aligned=("aligned", "sum"),
                ambiguous=("ambiguous", "sum"), unmatched=("unmatched", "sum"),
                zero_candidates=("zero_candidates", "sum"),
                aligned_arc_missing=("aligned_arc_missing", "sum"),
                aligned_with_other_candidates=("aligned_with_other_candidates", "sum"),
                usable_for_training=("usable_for_training", "sum"),
            ))

        audit = pd.concat(coverage)
        audit.to_csv(stage / "reference_label_coverage.csv", lineterminator="\n")
        for artifact in stage.iterdir():
            artifact.replace(processed_dir / artifact.name)
    print(audit.groupby(level="split").sum().to_string())


def main():
    parser = argparse.ArgumentParser(description="Align GPS observations to ordered reference arcs.")
    parser.add_argument("--archive", type=Path, default=Path(os.environ.get(
        "PATHSNAP_DATA_ZIP", ROOT / "data" / "raw" / "Student folder.zip",
    )))
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data" / "processed")
    args = parser.parse_args()
    export_labels(args.archive, args.processed_dir)


if __name__ == "__main__":
    main()
