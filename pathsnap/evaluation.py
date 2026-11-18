"""Aggregate point accuracy against server-side aligned references."""

import re
from collections import Counter
from zipfile import ZipFile

import numpy as np


def evaluate(labels, candidates, decoded):
    counts = Counter(labels["alignment_status"])
    aligned = labels.loc[labels.alignment_status.eq("aligned")]
    top1_correct = viterbi_correct = misses = 0
    for row in aligned.itertuples(index=False):
        options = candidates.get(row.obs_id, ())
        if not any(option.position.arc_id == row.reference_arc_id for option in options):
            misses += 1
            continue
        best = max(options, key=lambda c: (c.probability, -c.candidate_rank, -c.position.arc_id))
        top1_correct += int(best.position.arc_id == row.reference_arc_id)
        viterbi_correct += int(decoded.get(row.obs_id) == row.reference_arc_id)
    total = len(aligned)
    return {
        "aligned": total,
        "ambiguous_excluded": counts["ambiguous"],
        "unmatched_excluded": sum(count for status, count in counts.items() if status.startswith("unmatched_")),
        "candidate_misses": misses,
        "top1_correct": top1_correct,
        "viterbi_correct": viterbi_correct,
        "top1_accuracy": top1_correct / total if total else None,
        "viterbi_accuracy": viterbi_correct / total if total else None,
    }


def load_reference_route(archive_path, trip_id, arc_count, *, split=None):
    """Read the server-side ordered route for one trip after prediction."""
    trip_prefix = f"{split}/" if split else ""
    suffix = f"{trip_prefix}{trip_id}/{trip_id}.route"
    pattern = re.compile(r"(?:^|/)" + re.escape(suffix) + r"$")
    with ZipFile(archive_path) as archive:
        matches = [item for item in archive.infolist() if pattern.search(item.filename)]
        if not matches:
            raise KeyError(f"No reference route for {trip_id}")
        if len(matches) != 1 or matches[0].file_size > 75_000_000:
            raise ValueError(f"Invalid reference entry for {trip_id}")
        with archive.open(matches[0]) as source:
            route = np.loadtxt(source, dtype=np.int64, ndmin=2)
    if route.shape[1] != 1 or not len(route) or (route < 0).any() or (route >= arc_count).any():
        raise ValueError(f"Invalid reference arc IDs for {trip_id}")
    return route[:, 0]


def route_pair_counts(segments):
    """Count ordered, distinct-arc transitions within each route segment."""
    return Counter((a, b) for segment in segments
                   for a, b in zip(segment, segment[1:]) if a != b)
