"""Evaluate trained PathSnap candidate scores with and without Viterbi.

Run from repository root:
    python scripts/benchmark_trained_decoding.py

Requires the student-test observations/candidates/labels, the road-network ZIP,
and Person 2's two committed student-test score exports. Uses no evaluator split,
no ground-truth leakage into predictions, and no fabricated results.

Writes per-trip, aggregate and methodology CSV/Markdown into
experiments/route_comparison/ .
"""

import argparse
import csv
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import sys

# Running `python scripts/benchmark_trained_decoding.py` keeps scripts/ on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pathsnap.decoder import Candidate, Observation, decode
from pathsnap.evaluation import evaluate
from pathsnap.routing import Position, RoadGraph
from pathsnap.scoring import match_exported_scores, SCORE_FILES, DISPLAY_NAMES
from scripts.view_trip import ROOT, load_trip

KEY = ["split", "trip_id", "obs_id", "arc_id"]
SPLIT = "student_test"
MODELS = ("logistic_regression", "knn")


def continuity(track, arcs, predictions):
    """Match scripts/train_baselines.py's direct-arc continuity definition.

    Eligible pairs have two predicted arcs and no GPS gap or row break.
    Same arc or adjacent directed endpoints count as continuous. Valid
    multi-arc routes are not necessarily counted by this strict metric.
    """
    ids = track.obs_id.tolist()
    p = [predictions.get(int(obs_id)) for obs_id in ids]
    gap = track.gap_before.fillna(False).astype(bool).to_numpy()
    row_break = track.row_break_before.fillna(False).astype(bool).to_numpy()
    timestamps = track.timestamp_s.to_numpy(dtype=float)
    matches = 0
    total = 0
    for i in range(1, len(p)):
        if gap[i] or row_break[i] or timestamps[i] - timestamps[i - 1] > 5.0:
            continue
        first, second = p[i - 1], p[i]
        if first is None or second is None:
            continue
        total += 1
        matches += int(first == second or arcs[first, 1] == arcs[second, 0])
    return matches, total


def assemble(track, candidates, scores):
    """Build the *complete* observation timeline, including missing candidates."""
    if len(candidates) != len(scores):
        raise ValueError("Score/candidate length mismatch")
    by_obs = {}
    for row, probability in zip(candidates.itertuples(index=False), scores, strict=True):
        by_obs.setdefault(int(row.obs_id), []).append(
            Candidate(Position(int(row.arc_id), float(row.fraction_along_arc)),
                      float(probability), int(row.candidate_rank))
        )
    timeline = [
        Observation(int(row.obs_id), float(row.timestamp_s),
                    (float(row.longitude), float(row.latitude)),
                    tuple(by_obs.get(int(row.obs_id), ())))
        for row in track.itertuples(index=False)
    ]
    return timeline, by_obs


def one_trip(model, trip_id, data, archive, candidates_file, labels_file, score_file, *, decoder_kwargs=None, cache_size=0):
    t0 = perf_counter()
    track, nodes, arcs = load_trip(data / "observations.csv.gz", archive, SPLIT, trip_id)
    rows = pq.read_table(candidates_file, filters=[("split", "=", SPLIT), ("trip_id", "=", trip_id)]).to_pandas()
    labels = pq.read_table(
        labels_file, filters=[("split", "=", SPLIT), ("trip_id", "=", trip_id)],
        columns=["obs_id", "alignment_status", "reference_arc_id"],
    ).to_pandas()
    if not labels.obs_id.is_unique or set(labels.obs_id) != set(track.obs_id):
        raise ValueError(f"Reference/observation ID mismatch for {trip_id}")

    t_score = perf_counter()
    exported = pq.read_table(
        score_file, filters=[("split", "=", SPLIT), ("trip_id", "=", trip_id)],
        columns=KEY + ["score"],
    ).to_pandas()
    scores = match_exported_scores(rows, exported)
    score_seconds = perf_counter() - t_score
    if rows.empty or not rows["trip_id"].eq(trip_id).all():
        raise ValueError(f"Wrong or empty candidate trip {trip_id}")
    obs, by_obs = assemble(track, rows, scores)
    graph = RoadGraph(nodes, arcs, shortest_cache_size=cache_size)

    t_decode = perf_counter()
    reconstructed = decode(graph, obs, **(decoder_kwargs or {}))
    decode_seconds = perf_counter() - t_decode

    report = evaluate(labels, by_obs, reconstructed.arcs_by_observation)
    top1 = {
        int(obs_id): min(options, key=lambda c: (-c.probability, c.candidate_rank, c.position.arc_id)).position.arc_id
        for obs_id, options in by_obs.items() if options
    }
    if sum(int(top1.get(row.obs_id) == row.reference_arc_id)
           for row in labels.itertuples(index=False) if row.alignment_status == "aligned") != report["top1_correct"]:
        raise AssertionError("Top-1 ranking disagrees with evaluation.py")

    top1_good, top1_pairs = continuity(track, arcs, top1)
    decoded_good, decoded_pairs = continuity(track, arcs, reconstructed.arcs_by_observation)
    return {
        "model": model,
        "trip_id": trip_id,
        "observations": len(track),
        "candidates": len(rows),
        "aligned": report["aligned"],
        "ambiguous_excluded": report["ambiguous_excluded"],
        "unmatched_excluded": report["unmatched_excluded"],
        "candidate_misses": report["candidate_misses"],
        "top1_correct": report["top1_correct"],
        "viterbi_correct": report["viterbi_correct"],
        "top1_accuracy": report["top1_accuracy"],
        "viterbi_accuracy": report["viterbi_accuracy"],
        "accuracy_gain_pp": 100 * (report["viterbi_accuracy"] - report["top1_accuracy"]),
        "top1_continuous": top1_good,
        "top1_pairs": top1_pairs,
        "top1_continuity": top1_good / top1_pairs if top1_pairs else None,
        "viterbi_continuous": decoded_good,
        "viterbi_pairs": decoded_pairs,
        "viterbi_continuity": decoded_good / decoded_pairs if decoded_pairs else None,
        "route_segments": len(reconstructed.segments),
        "route_breaks": max(0, len(reconstructed.segments) - 1),
        "gps_gaps": int(track.gap_before.fillna(False).sum()),
        "unmatched_prediction_observations": sum(v is None for v in reconstructed.arcs_by_observation.values()),
        "matched_observations": sum(v is not None for v in reconstructed.arcs_by_observation.values()),
        "routing_seconds": graph.route_seconds,
        "dijkstra_seconds": graph.shortest_search_seconds,
        "shortest_searches": graph.shortest_searches,
        "cache_hits": graph.shortest_cache_hits,
        "score_seconds": score_seconds,
        "decode_seconds": decode_seconds,
        "end_to_end_seconds": perf_counter() - t0,
    }


def summarize(rows):
    combined = []
    for model in MODELS:
        sample = [r for r in rows if r["model"] == model]
        assert len(sample) == 10 and len({r["trip_id"] for r in sample}) == 10
        sums = lambda col: sum(x[col] for x in sample)
        aligned = sums("aligned")
        top = sums("top1_correct") / aligned
        vit = sums("viterbi_correct") / aligned
        a_pairs = sums("top1_pairs")
        b_pairs = sums("viterbi_pairs")
        combined.append({
            "model": model, "trips": len(sample), "aligned": aligned,
            "observations": sums("observations"), "candidate_misses": sums("candidate_misses"),
            "top1_correct": sums("top1_correct"), "viterbi_correct": sums("viterbi_correct"),
            "top1_accuracy": top, "viterbi_accuracy": vit,
            "accuracy_gain_pp": 100 * (vit - top),
            "top1_continuous": sums("top1_continuous"), "top1_pairs": a_pairs,
            "top1_continuity": sums("top1_continuous") / a_pairs if a_pairs else None,
            "viterbi_continuous": sums("viterbi_continuous"), "viterbi_pairs": b_pairs,
            "viterbi_continuity": sums("viterbi_continuous") / b_pairs if b_pairs else None,
            "route_segments": sums("route_segments"),
            "route_breaks": sums("route_breaks"),
            "gps_gaps": sums("gps_gaps"),
            "unmatched_prediction_observations": sums("unmatched_prediction_observations"),
            "routing_seconds": sums("routing_seconds"),
            "dijkstra_seconds": sums("dijkstra_seconds"),
            "shortest_searches": sums("shortest_searches"),
            "cache_hits": sums("cache_hits"),
            "score_seconds": sums("score_seconds"), "decode_seconds": sums("decode_seconds"),
            "end_to_end_seconds": sums("end_to_end_seconds"),
            "trips_improved": sum(r["viterbi_correct"] > r["top1_correct"] for r in sample),
            "trips_worsened": sum(r["viterbi_correct"] < r["top1_correct"] for r in sample),
            "trips_tied": sum(r["viterbi_correct"] == r["top1_correct"] for r in sample),
        })
    return combined


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report_markdown(summary, per_trip):
    lines = [
        "# Trained-classifier + Viterbi comparison (PathSnap)",
        "",
        "Actual development results for all ten student-test trips, **not evaluator-testing data**.",
        "The previously reported 54.45% → 78.05% result is for the **distance baseline only**.",
        "",
        "## Aggregate (micro-averaged)",
        "",
        "| Model | Aligned obs. | Top-1 | Viterbi | Change (pp) | Top-1 continuity | Viterbi continuity | Scoring (s) | Decoding (s) | End-to-end (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary:
        lines.append(
            f"| {DISPLAY_NAMES[r['model']]} | {r['aligned']:,} | {r['top1_accuracy']:.2%} "
            f"| {r['viterbi_accuracy']:.2%} | {r['accuracy_gain_pp']:+.2f} "
            f"| {r['top1_continuity']:.2%} | {r['viterbi_continuity']:.2%} "
            f"| {r['score_seconds']:.2f} | {r['decode_seconds']:.2f} | {r['end_to_end_seconds']:.2f} |"
        )
    lines += [
        "",
        "## Per-trip results",
        "",
        "| Model | Trip | Aligned | Top-1 | Viterbi | Change (pp) | Top-1 continuity | Viterbi continuity | Decode (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in per_trip:
        lines.append(
            f"| {r['model']} | `{r['trip_id']}` | {r['aligned']:,} | {r['top1_accuracy']:.2%} "
            f"| {r['viterbi_accuracy']:.2%} | {r['accuracy_gain_pp']:+.2f} "
            f"| {r['top1_continuity']:.2%} | {r['viterbi_continuity']:.2%} | {r['decode_seconds']:.2f} |"
        )
    lines += [
        "",
        "## Method",
        "",
        "- Scores: Person 2's committed Logistic Regression and KNN student-test probability exports; no retraining or distance-score substitution.",
        "- Keys: full `(split, trip_id, obs_id, arc_id)` validation/join, one probability per candidate. Candidate ranking breaks ties by smaller candidate distance rank, then smaller arc ID.",
        "- Accuracy: independently ranked Top-1 versus Viterbi road-arc assignment on aligned observations, with absent reference arcs counted wrong. Unmatched and ambiguous labels excluded and counted.",
        "- Continuity: same **strict directly adjacent road-arc metric** as baseline training script: among consecutive observations without GPS gaps/row breaks and with valid predictions, the two arcs must be identical or their directed endpoints directly connect. This can penalize real multi-arc paths; it is not a complete graph-path continuity measure.",
        "- Runtime: per-model score export load and key join (scoring), Viterbi decode only (decoding), and full trip pipeline including I/O, graph assembly and evaluation (end-to-end). Times are cumulative and machine dependent.",
        "- Only student-test data is used for this experiment. Reference labels are never passed to the scorer or decoder.",
        "",
        "Reproduce with `python scripts/benchmark_trained_decoding.py` from the repository root.",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params-file", type=Path, default=None, help="Training-selected Viterbi parameters JSON")
    parser.add_argument("--cache-size", type=int, default=0, help="Per-trip bounded shortest-path cache entries")
    parser.add_argument("--processed", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--archive", type=Path, default=Path(os.environ.get(
        "PATHSNAP_DATA_ZIP", ROOT / "data/raw/Student folder.zip",
    )))
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/route_comparison")
    args = parser.parse_args()
    summary_file = args.processed / "trip_summary.csv"
    summary = pd.read_csv(summary_file, dtype={"trip_id": str})
    trips = sorted(summary.loc[summary["split"].eq(SPLIT), "trip_id"].unique().tolist())
    if len(trips) != 10:
        raise ValueError(f"Expected all ten student-test trips, received {len(trips)}: {trips}")
    if not args.archive.is_file():
        raise FileNotFoundError(f"Original road-network ZIP missing: {args.archive}")
    candidates_file = args.processed / "candidates_student_test.parquet"
    labels_file = args.processed / "reference_labels_student_test.parquet"
    for path in [candidates_file, labels_file, *(SCORE_FILES[m] for m in MODELS)]:
        if not path.is_file():
            raise FileNotFoundError(f"Required project export unavailable: {path}")

    args.output.mkdir(parents=True, exist_ok=True)
    if args.cache_size < 0:
        raise ValueError("Cache size must be >= 0")
    params = json.loads(args.params_file.read_text()) if args.params_file else {}
    allowed = {"gap_seconds", "beta_m", "max_speed_mps", "routing_slack_m"}
    if set(params) - allowed:
        raise ValueError("Unexpected decoder parameters")
    rows = []
    for model in MODELS:
        print(f"\nScorer: {model}", flush=True)
        for trip_id in trips:
            row = one_trip(model, trip_id, args.processed, args.archive,
                           candidates_file, labels_file, SCORE_FILES[model], decoder_kwargs=params, cache_size=args.cache_size)
            rows.append(row)
            write_csv(args.output / "per_trip.csv", rows)  # save partial measurements
            print(
                f"{model} {trip_id} | Top-1 {row['top1_accuracy']:.2%}, "
                f"Viterbi {row['viterbi_accuracy']:.2%}, "
                f"direct continuity {row['top1_continuity']:.2%} -> {row['viterbi_continuity']:.2%}, "
                f"decode {row['decode_seconds']:.2f}s",
                flush=True,
            )
    aggregate = summarize(rows)
    write_csv(args.output / "aggregate.csv", aggregate)
    (args.output / "README.md").write_text(report_markdown(aggregate, rows), encoding="utf-8")
    (args.output / "environment.json").write_text(
        json.dumps({
            "python": __import__("sys").version,
            "numpy": np.__version__, "pandas": pd.__version__,
            "pyarrow": __import__("pyarrow").__version__,
            "split": SPLIT, "models": MODELS, "cache_size": args.cache_size,
            "chosen_decoder_overrides": params,
            "route_breaks_definition": "max(0, number_of_decoded_segments-1) per trip; includes starts after GPS gaps, missing candidates and unreachable directed transitions",
            "decoder_parameters": {
                "gap_seconds": 5.0, "beta_m": 30.0,
                "max_speed_mps": 65.0, "routing_slack_m": 100.0,
            },
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    print("\n" + report_markdown(aggregate, rows), flush=True)


if __name__ == "__main__":
    main()
