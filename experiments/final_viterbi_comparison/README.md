# Five saved-model comparisons after Viterbi

**Measured run:** [successful GitHub Actions run #44256618239](https://github.com/vasishtcoffee/ML-HACKATHON/actions/runs/44256618239) (18 November 2026) · [download exact per-trip and aggregate CSVs and run log](https://github.com/vasishtcoffee/ML-HACKATHON/actions/runs/44256618239/artifacts/14405744386).

All five models were trained previously on the **80 training trips**. Their saved score exports were evaluated using the **same 10 student-test trips**, reference-alignment denominator (16,317), candidate keys, decoder parameters (`beta_m=30`, `gap_seconds=5`), road-graph structure and 20,000-entry per-trip routing cache. Model optimisation was Person 2's responsibility; **Random Forest is the saved untuned baseline**, while the others below are saved tuned models.

## Accuracy and continuity

| Model | Classifier-only top-1 | Viterbi | Gain (pp) | Top-1 direct continuity | Viterbi direct continuity | Total route breaks |
|---|---:|---:|---:|---:|---:|---:|
| Tuned Logistic Regression | 54.58% | 80.24% | +25.65 | 82.11% | 95.46% | 221 |
| **Tuned KNN** | 54.72% | **81.15%** | +26.43 | 82.07% | 95.58% | 221 |
| Tuned MLP | 54.96% | 80.77% | +25.81 | 82.34% | **95.89%** | 221 |
| Random Forest (untuned) | **55.02%** | 80.95% | +25.93 | **82.38%** | 95.72% | 221 |
| Tuned XGBoost | 54.83% | 80.73% | +25.91 | 82.20% | 95.77% | 221 |

### Interpretation and model recommendation

- **Tuned KNN has the highest measured Viterbi observation-level accuracy: 81.15%**, despite Random Forest having higher classifier-only accuracy (55.02%). Choosing the winning classifier by top-1 alone would therefore be misleading for route reconstruction.
- Tuned MLP has the highest measured strict direct-arc continuity, 95.89%. This measure only checks directly adjacent road arcs, not an arbitrarily long connecting road path.
- The 221 route breaks are the **same across the five models**, likely because the graph, candidate coverage, and 5-second gap handling determine which steps can connect. This is not evidence that all model predictions are the same.
- For a demo prioritising **route matching accuracy**, tuned KNN is the leading option *on this student-test dataset*. However, KNN's full fresh inference may be slower and more memory-intensive than MLP or Logistic Regression; decide on the final model with the team using actual server-side inference timings and the hackathon judging criteria.
- **No final model is hard-coded** in the backend. Set `PATHSNAP_FINAL_MODEL=live_knn` on Laptop 1 *if the team chooses tuned KNN*.

## Timing and limitations

| Model | 10-trip Viterbi decode time with caching | 10-trip total benchmark time* |
|---|---:|---:|
| Tuned Logistic Regression | 8.55 s | 13.52 s |
| Tuned KNN | 8.57 s | 13.48 s |
| Tuned MLP | 8.71 s | 13.83 s |
| Random Forest | 8.64 s | 13.71 s |
| Tuned XGBoost | 8.77 s | 13.88 s |

*These timings cover precomputed score-file loading, graph creation, Viterbi and reference evaluation. They **exclude original live fitted-model prediction/inference** (`predict_proba`) and must not be interpreted as total real demo latency. Actual live inference can be measured using the new FastAPI `timings.score_load_seconds` field with `live_*` models.

Trip-level results, candidate misses, break counts, exact timings and aggregate counts are in the downloadable GitHub Actions artifact above. No evaluator labels were read or used. The ten student-test trips are development evidence and do not establish performance on the ten reserved evaluator trips.
