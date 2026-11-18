# Graph-aware route decoder

The decoder combines classifier candidate probabilities with directed local road
transitions. Candidate probabilities are normalized within each observation and
converted to log emissions. For an adjacent observation pair, a transition is
allowed only when both candidates lie on the same directed arc in forward order,
or when the first arc ends at the second arc's start node. The transition score is
`-|GPS displacement - road distance| / 30 m`. The scale is fixed at 30 m for this
run; it was not selected using student-test results.

GPS gaps, original row breaks and observations without candidates split the
sequence. If the current candidates have no reachable predecessor, Viterbi
restarts there and records the chosen local route. The decoder does not invent
missing arcs. This local transition rule does not search paths with intermediate
unobserved arcs, so such cases trigger a restart.

## Student-test comparison

The fixed decoder was evaluated on all ten student-test trips using the saved
Random Forest and XGBoost candidate scores. Top-1 is weighted across aligned
observations and candidate misses are wrong. Continuity counts predicted adjacent
or same arcs only across non-gap observation pairs. Reference labels are produced
by geometric alignment to the supplied route; ambiguous and unmatched labels are
excluded from top-1. The observed improvement should be treated cautiously because
reference alignment is not independent observation-level ground truth.

| Model | Classifier top-1 | Decoded top-1 | Classifier continuity | Decoded continuity |
|---|---:|---:|---:|---:|
| Random Forest | 55.02% | 81.57% | 80.34% | 99.96% |
| XGBoost | 54.78% | 81.47% | 80.26% | 99.96% |

This is a final held-out student-test measurement, not a decoder tuning result.
Do not use these values to choose transition scale or model parameters. The near
perfect continuity follows partly from allowing only local connected transitions
and restarting at impossible steps; report restart frequency alongside continuity
in follow-up evaluation. Decoder-only metrics do not establish a final model win.

## Reproduce

```bash
python scripts/evaluate_decoder.py --models random_forest xgboost --phase student_test
```

The script writes per-trip and pooled metrics under `experiments/decoder/`.
For model selection or transition-scale experiments, implement cross-validation
with trip groups using only the 80 training trips. The current script's `--phase
cv` path refits models and can take several minutes per fold; the recorded
comparison above uses the already saved model score exports.
