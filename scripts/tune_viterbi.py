"""Select Viterbi parameters using TRAIN trips only, never student-test labels.

Execute from repository root:
    python scripts/tune_viterbi.py

This intentionally uses distance-emission scores on training trips because
the available trained Logistic Regression and KNN exports are student-test only.
The selected transition parameters are then evaluated once, separately,
on student-test model scores with benchmark_trained_decoding.py.
"""

import csv
import json
import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathsnap.decoder import decode
from pathsnap.evaluation import evaluate
from pathsnap.routing import RoadGraph
from scripts.view_trip import ROOT, load_trip
from scripts.benchmark_trained_decoding import assemble, continuity

DATA = ROOT / "data/processed"
ARCHIVE = Path(os.environ.get("PATHSNAP_DATA_ZIP", ROOT / "data/raw/Student folder.zip"))
OUT = ROOT / "experiments/viterbi_tuning"


def pick_spread(frame, count):
    """Select fixed training-trip IDs across lengths, without using labels."""
    frame = frame.sort_values(["gps", "trip_id"]).reset_index(drop=True)
    idx = [min(len(frame) - 1, int((i + 0.5) * len(frame) / count)) for i in range(count)]
    selected = frame.iloc[idx].trip_id.tolist()
    if len(set(selected)) != count:
        raise ValueError("Not enough unique training trips")
    return selected


def calibration_ids():
    meta = pd.read_csv(DATA / "trip_summary.csv", dtype={"trip_id": str})
    eligible = meta.loc[
        meta["split"].eq("train")
        & meta["gps"].between(300, 4000)
        & meta["arcs"].lt(50000)
    ]
    gapped = pick_spread(eligible.loc[eligible.gaps.gt(0)], 6)
    regular = pick_spread(eligible.loc[eligible.gaps.eq(0)], 4)
    # Separate, group-disjoint tuning and validation subsets, selected without
    # looking at accuracy. Gapped trips make the gap parameter meaningful.
    tuning = sorted(gapped[:3] + regular[:3])
    validation = sorted(gapped[3:] + regular[3:])
    if len(tuning) != 6 or len(validation) != 4 or set(tuning) & set(validation):
        raise AssertionError("Invalid train trip split")
    return tuning, validation


def load_training_trip(trip_id):
    track, nodes, arcs = load_trip(DATA / "observations.csv.gz", ARCHIVE, "train", trip_id)
    rows = pq.read_table(DATA / "candidates_train.parquet",
                         filters=[("split", "=", "train"), ("trip_id", "=", trip_id)]).to_pandas()
    labels = pq.read_table(DATA / "reference_labels_train.parquet",
                           filters=[("split", "=", "train"), ("trip_id", "=", trip_id)],
                           columns=["obs_id", "alignment_status", "reference_arc_id"]).to_pandas()
    if not labels.obs_id.is_unique or set(labels.obs_id) != set(track.obs_id):
        raise ValueError("Mismatched labels in training trip " + trip_id)
    if rows.empty or not rows.trip_id.eq(trip_id).all():
        raise ValueError("Missing candidates in training trip " + trip_id)
    distances = rows["distance_to_arc_m"].to_numpy(dtype=float)
    scores = np.exp(-0.5 * (distances / 20.0)**2)
    obs, grouped = assemble(track, rows, scores)
    return track, arcs, labels, obs, grouped, nodes


def measure(trip_id, values, params):
    track, arcs, labels, obs, grouped, nodes = values
    graph = RoadGraph(nodes, arcs, shortest_cache_size=20000)
    start = perf_counter()
    decoded = decode(graph, obs, **params)
    decode_time = perf_counter() - start
    report = evaluate(labels, grouped, decoded.arcs_by_observation)
    good, pairs = continuity(track, arcs, decoded.arcs_by_observation)
    return {
        "trip_id": trip_id,
        "beta_m": params["beta_m"],
        "gap_seconds": params["gap_seconds"],
        "aligned": report["aligned"],
        "correct": report["viterbi_correct"],
        "accuracy": report["viterbi_accuracy"],
        "continuous": good,
        "adjacent_pairs": pairs,
        "continuity": good / pairs if pairs else None,
        "segments": len(decoded.segments),
        "route_breaks": max(0, len(decoded.segments)-1),
        "gps_gaps": int(track.gap_before.fillna(False).sum()),
        "decode_seconds": decode_time,
        "routing_seconds": graph.route_seconds,
        "dijkstra_seconds": graph.shortest_search_seconds,
        "searches": graph.shortest_searches,
        "cache_hits": graph.shortest_cache_hits,
    }


def aggregate(rows):
    aligned = sum(r["aligned"] for r in rows)
    pairs = sum(r["adjacent_pairs"] for r in rows)
    return {
        "beta_m": rows[0]["beta_m"],
        "gap_seconds": rows[0]["gap_seconds"],
        "trips": len(rows),
        "aligned": aligned,
        "correct": sum(r["correct"] for r in rows),
        "accuracy": sum(r["correct"] for r in rows) / aligned,
        "continuity": sum(r["continuous"] for r in rows)/pairs if pairs else None,
        "route_breaks": sum(r["route_breaks"] for r in rows),
        "decode_seconds": sum(r["decode_seconds"] for r in rows),
    }


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        out = csv.DictWriter(file, fieldnames=list(rows[0]))
        out.writeheader()
        out.writerows(rows)


def main():
    if not ARCHIVE.is_file():
        raise FileNotFoundError(f"Missing road network archive: {ARCHIVE}")
    OUT.mkdir(parents=True, exist_ok=True)
    calibration, validation = calibration_ids()
    print("Training calibration:", calibration, flush=True)
    print("Training holdout:", validation, flush=True)
    (OUT / "split.json").write_text(json.dumps({
        "calibration_train_ids": calibration,
        "holdout_train_ids": validation,
        "selection": "Trip-length spread with 6 gapped and 4 gap-free training trips; graph arcs under 50000, GPS 300..4000",
        "emissions": "Development distance baseline only (exp(-0.5*(distance_m/20)^2))",
    }, indent=2)+"\n")

    # Reuse raw inputs but not reference labels in inference: labels enter
    # evaluate() only after each route has been decoded.
    cache = {trip_id: load_training_trip(trip_id) for trip_id in calibration}
    grid = [{"beta_m": beta, "gap_seconds": gap}
            for beta in (15.0, 30.0, 60.0)
            for gap in (3.0, 5.0, 10.0)]
    details, summaries = [], []
    for params in grid:
        trials = []
        for trip_id in calibration:
            result = measure(trip_id, cache[trip_id], params)
            details.append(result)
            trials.append(result)
        agg = aggregate(trials)
        summaries.append(agg)
        write_csv(OUT / "calibration_per_trip.csv", details)
        write_csv(OUT / "calibration_grid.csv", summaries)
        print(f"beta={params['beta_m']:.0f} gap={params['gap_seconds']:.0f}: "
              f"accuracy={agg['accuracy']:.4%}, continuity={agg['continuity']:.4%}, "
              f"breaks={agg['route_breaks']}, decode={agg['decode_seconds']:.1f}s", flush=True)

    # Calibration accuracy first; continuity and lower route breaks as tiebreaks.
    # Use no student-test information to choose the parameters.
    best = max(summaries, key=lambda x: (
        x["accuracy"], x["continuity"] if x["continuity"] is not None else -1,
        -x["route_breaks"], -x["decode_seconds"],
    ))
    chosen = {"beta_m": best["beta_m"], "gap_seconds": best["gap_seconds"]}
    (OUT / "chosen.json").write_text(json.dumps(chosen, indent=2)+"\n")
    print("CHOSEN FROM TRAINING ONLY", chosen, flush=True)

    holdout_data = {trip_id: load_training_trip(trip_id) for trip_id in validation}
    holdout = []
    for label, params in [("default", {"beta_m": 30.0, "gap_seconds": 5.0}),
                          ("training_selected", chosen)]:
        cases = []
        for trip_id in validation:
            row = measure(trip_id, holdout_data[trip_id], params)
            row["settings"] = label
            holdout.append(row)
            cases.append(row)
        agg = aggregate(cases)
        print("TRAINING HOLDOUT", label, agg, flush=True)
    write_csv(OUT / "training_holdout.csv", holdout)
    (OUT / "README.md").write_text(
        "# Training-only Viterbi parameter selection\n\n"
        "The calibration and holdout groups contain **only training trips**. "
        "Selected score parameters use a distance-emission scorer because trained "
        "Logistic Regression/KNN training-score files were not available. "
        "No student-test results informed this search.\n\n"
        "The grid tests beta_m = 15, 30, 60 metres and gap_seconds = 3, 5, 10. "
        "The selected setting maximizes pooled aligned-point accuracy on six "
        "calibration trips, then continuity, then fewer breaks. A distinct "
        "four-trip **training** holdout is reported for comparison with default "
        "beta_m=30, gap_seconds=5. The selected decoder can be evaluated on "
        "student-test trained-model scores only after this selection.\n\n"
        "Settings and exact metrics are in chosen.json and the CSV files. "
        "These are development parameters and may not generalize to evaluator data.\n"
    )
    print("Training-only tuning completed.", flush=True)


if __name__ == "__main__":
    main()
