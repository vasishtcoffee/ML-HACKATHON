"""Aggregate point accuracy against server-side aligned references."""

from collections import Counter


def evaluate(labels, candidates, decoded):
    counts = Counter(labels["alignment_status"])
    aligned = labels.loc[labels.alignment_status.eq("aligned")]
    top1_correct = viterbi_correct = misses = 0
    for row in aligned.itertuples(index=False):
        options = candidates.get(row.obs_id, ())
        if not any(option.position.arc_id == row.reference_arc_id for option in options):
            misses += 1
            continue
        best = max(options, key=lambda c: (c.probability, -c.position.arc_id))
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
