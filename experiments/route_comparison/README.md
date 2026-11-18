# PathSnap — trained scorer route reconstruction benchmark

**Execution:** GitHub Actions run [#44248262552](https://github.com/vasishtcoffee/ML-HACKATHON/actions/runs/44248262552) (successful, 18 November 2026).
**Full-precision artifacts:** [download CSVs, execution log and environment](https://github.com/vasishtcoffee/ML-HACKATHON/actions/runs/44248262552/artifacts/14402128537).

These are **measured results** on all ten **student-test** trips using Person 2's committed
trained Logistic Regression and KNN probability exports, not distance-based
scores. The earlier **54.45% → 78.05%** result is a separate **distance-baseline**
experiment and must not be described as a trained-model result.

## Aggregate results (micro-averaged by aligned observation)

| Model | Aligned observations | Top-1 accuracy | Viterbi accuracy | Accuracy gain (pp) | Top-1 continuity | Viterbi continuity | Score-file load + key join (s) | Viterbi decode (s) | Full pipeline (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression (trained) | 16,317 | 54.58% | **80.22%** | +25.64 | 82.11% | 95.46% | 0.38 | 20.62 | 25.68 |
| KNN (trained) | 16,317 | 54.70% | **81.01%** | +26.32 | 82.04% | 95.66% | 0.37 | 20.74 | 25.93 |

All reported aggregate accuracies use the same **16,317 aligned observations**
per model. The two models' separate timings sum across all ten trips. The
measured end-to-end time includes reading files, building graph/candidates,
scoring-file join, decoding, and evaluation. **It does not include running the
original trained classifier's `predict_proba` inference**, since precomputed
student-test scores were used.

## Per-trip measured results

Values below are rounded to two decimals. For full-precision counts and timing,
download the workflow artifact above.

| Model | Trip ID | Top-1 accuracy | Viterbi accuracy | Top-1 continuity | Viterbi continuity | Viterbi decode (seconds) |
|---|---|---:|---:|---:|---:|---:|
| logistic_regression | `00000000` | 53.51% | 95.10% | 90.33% | 98.44% | 0.39 |
| logistic_regression | `00000007` | 45.44% | 86.59% | 92.09% | 99.56% | 0.07 |
| logistic_regression | `00000009` | 55.67% | 82.72% | 66.60% | 89.41% | 5.08 |
| logistic_regression | `00000040` | 53.78% | 86.82% | 89.58% | 98.13% | 0.37 |
| logistic_regression | `00000052` | 80.79% | 99.51% | 77.01% | 95.79% | 0.05 |
| logistic_regression | `00000056` | 70.36% | 88.32% | 76.27% | 95.53% | 3.43 |
| logistic_regression | `00000065` | 74.96% | 75.89% | 79.40% | 95.54% | 0.61 |
| logistic_regression | `00000076` | 46.04% | 80.89% | 71.37% | 89.42% | 4.81 |
| logistic_regression | `00000077` | 42.80% | 55.39% | 84.12% | 94.30% | 2.51 |
| logistic_regression | `00000091` | 57.29% | 80.09% | 75.26% | 93.47% | 3.30 |
| knn | `00000000` | 52.70% | 94.51% | 90.17% | 98.80% | 0.40 |
| knn | `00000007` | 48.79% | 87.22% | 93.13% | 99.78% | 0.07 |
| knn | `00000009` | 47.45% | 83.29% | 66.03% | 89.12% | 5.07 |
| knn | `00000040` | 54.47% | 88.18% | 89.35% | 98.50% | 0.34 |
| knn | `00000052` | 82.76% | 100.00% | 75.86% | 96.17% | 0.05 |
| knn | `00000056` | 69.35% | 89.03% | 74.96% | 95.40% | 3.42 |
| knn | `00000065` | 71.56% | 76.35% | 79.53% | 96.46% | 0.65 |
| knn | `00000076` | 48.56% | 82.29% | 72.89% | 90.80% | 4.98 |
| knn | `00000077` | 44.08% | 56.65% | 85.06% | 94.12% | 2.51 |
| knn | `00000091` | 58.59% | 81.24% | 73.54% | 93.70% | 3.25 |

## Method and interpretation

- Inputs: `data/processed/observations.csv.gz`,
  `candidates_student_test.parquet`, `reference_labels_student_test.parquet`,
  the original road graph archive, and the **two committed trained scorer
  exports** `experiments/baselines/*_student_scores.parquet`.
- Scores are validated and joined by complete
  `(split, trip_id, obs_id, arc_id)` keys, never by row order.
- Top-1 is the highest-score candidate; ties use lower candidate distance
  rank and then lower arc ID. The same probabilities feed Viterbi.
- Per-observation accuracy uses the aligned reference arc (estimated from
  the supplied reference route); candidate misses count against both methods.
  Ambiguous/unmatched references are excluded and logged.
- Continuity is the **strict direct adjacency** metric from
  `scripts/train_baselines.py`: between adjacent valid predictions not
  crossing GPS gaps/row breaks, arcs are identical or the first arc's head
  directly equals the next arc's tail. This may count valid multi-arc paths
  as discontinuous. It is not a full route-reachability score.
- The decoder uses gap threshold 5 seconds, transition penalty scale 30 m,
  max speed 65 m/s, and routing slack 100 m. These are development settings;
  no claim that they are optimally tuned.
- Both models improved on **all ten** student-test trips by this pointwise
  accuracy measure. KNN's overall decoded accuracy exceeded Logistic
  Regression's by 0.79 percentage points.
- No evaluator-testing data was used. This experiment supports the **Person 3
  development report**, not final evaluator-trip performance.

## Reproduce

Run from the repository root with Python and the required data dependencies:

```bash
python scripts/benchmark_trained_decoding.py
```

Or inspect/run the GitHub Actions workflow
`.github/workflows/compare-trained-routes.yml`.
