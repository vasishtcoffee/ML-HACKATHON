# Person 3 — Viterbi tuning, route breaks and demo speed

**Reproducible experiments:** [successful training-tuning workflow](https://github.com/vasishtcoffee/ML-HACKATHON/actions/runs/44252207304) · [complete CSVs, configurations, profiling JSON and logs](https://github.com/vasishtcoffee/ML-HACKATHON/actions/runs/44252207304/artifacts/14404517835).

The 20 trained-score comparisons use all ten **student-test** trips. Classifier probabilities are from Person 2's committed Logistic Regression and KNN score exports, joined by the exact observation/candidate key. **No trained classifier was tuned here.** Parameters were selected only using *training* trips and **distance-based** emission scores. The reserved evaluator split was not inspected.

## 1. Untuned 20-comparison baseline (beta=30 m, gap=5 s)

| Metric (all 10 student-test trips) | Logistic Regression | KNN |
|---|---:|---:|
| Aligned reference observations | 16,317 | 16,317 |
| Classifier-only correct | 8,906 | 8,925 |
| Classifier-only top-1 accuracy | 54.58% | 54.70% |
| Viterbi correct | 13,090 | 13,219 |
| Viterbi accuracy | **80.22%** | **81.01%** |
| Classifier-only direct-arc continuity | 82.11% | 82.04% |
| Viterbi direct-arc continuity | 95.46% | 95.66% |
| Viterbi route breaks (total across trips) | 221 | 221 |
| GPS time gaps in observations | 44 | 44 |
| Unmatched decoded observations | 0 | 0 |
| Viterbi decode runtime (instrumented Ubuntu runner) | 29.82 s | 29.96 s |
| Whole benchmark pipeline runtime | 35.97 s | 35.99 s |

**Route-break definition:** `max(0, number_of_decoded_route_segments - 1)` per trip; includes expected restarts after GPS gaps and restarts after disconnected candidate transitions. These counts are **not** a direct count of invalid roads. A strict continuity pair is counted when adjacent predicted arcs are identical or share directly connected directed endpoints, excluding GPS gaps/row breaks. This stricter measure can penalize otherwise valid multi-arc travel.

### Breakdown by student-test trip (same breaks for both baseline-trained models)

| Trip | Route breaks | GPS gaps | Logistic Viterbi accuracy | KNN Viterbi accuracy |
|---|---:|---:|---:|---:|
| `00000000` | 3 | 0 | 95.10% | 94.51% |
| `00000007` | 67 | 0 | 86.59% | 87.22% |
| `00000009` | 1 | 0 | 82.72% | 83.29% |
| `00000040` | 48 | 0 | 86.82% | 88.18% |
| `00000052` | 0 | 0 | 99.51% | 100.00% |
| `00000056` | 7 | 0 | 88.32% | 89.03% |
| `00000065` | 9 | 5 | 75.89% | 76.35% |
| `00000076` | 39 | 39 | 80.89% | 82.29% |
| `00000077` | 46 | 0 | 55.39% | 56.65% |
| `00000091` | 1 | 0 | 80.09% | 81.24% |

**Interpretation:** Viterbi helps on every student-test trip but remains weak on `00000077`. `00000007` (67 breaks with no GPS time gaps), `00000040` (48 breaks), and `00000077` (46 breaks) reveal road-transition fragmentation. `00000076` has 39 breaks and 39 time gaps; this is compatible with intended segmentation but should be evaluated separately from disconnected-road restarts. Inspect actual route maps before assigning a specific cause to any one break.

## 2. TRAIN-only decoder parameter search

Tried 9 combinations: distance-penalty scale `beta_m` of 15, 30, or 60 metres and splitting thresholds `gap_seconds` of 3, 5, or 10 seconds. Emission probabilities were generated from distance to arc (no trained classifier's training scores were available). Six training trips were used for calibration; four **different training trips** were used for validation.

Calibration preferred `beta_m=60`, `gap_seconds=10` (72.48% calibrated aligned-point accuracy, 48 breaks across the six calibration trips). On the four training holdout trips:

| Metric | Defaults 30/5 | Training-selected 60/10 |
|---|---:|---:|
| Aligned observations | 7,320 | 7,320 |
| Correct | 4,808 | 4,864 |
| Accuracy | 65.68% | **66.45%** |
| Direct-arc continuity | 98.25% | **98.33%** |
| Route breaks | 49 | 45 |

Then, **after locking in the training-selected setting**, checked it on the ten student-test trips:

| Metric | Logistic: default | Logistic: 60/10 | KNN: default | KNN: 60/10 |
|---|---:|---:|---:|---:|
| Viterbi accuracy | **80.22%** | 80.08% | **81.01%** | 80.65% |
| Viterbi continuity | **95.46%** | 93.87% | **95.66%** | 94.42% |
| Total route breaks | 221 | 204 | 221 | 204 |

**Decision:** Retain the original `30 m / 5 s` setting for the current trained-score development demo: it had higher student-test accuracy and continuity. The training-distance improvement did not transfer to trained classifier emissions. No repeated student-test hyperparameter search was performed; these were one-off checks of the train-selected setting.

## 3. Slow-trip profiling and optional cache

With KNN on student-test trip `00000009`, recorded separate routing, decoding and Folium-render times.

| Measurement | No cache | 20,000-entry per-trip cache |
|---|---:|---:|
| GPS observations / candidate rows | 1,049 / 20,779 | 1,049 / 20,779 |
| Decoder runtime | 7.264 s | **2.283 s** |
| Road-routing runtime (within decode) | 6.752 s | 1.785 s |
| Dijkstra runtime (within routing) | 5.843 s | 0.682 s |
| Actual Dijkstra search executions | 584,505 | 52,976 |
| Cache hits | 0 | 531,529 |
| Folium map generation + HTML rendering | 0.908 s | 0.939 s |
| Route segments / route breaks | 2 / 1 | **2 / 1** |

Cached decoding was **68.58% faster** in this one-pair Linux runner comparison. **The complete predicted arc assignment and all reconstructed route segments matched exactly**, asserted programmatically; the change only avoids repeat directed shortest-path searches. The cache is bounded, per-trip, and distinguishes a shortest-path success from a *bounded* failure. A failure under a small length limit cannot incorrectly rule out a valid route under a larger limit.

### Demo integration

`pathsnap.api` now enables a 20,000-entry per-trip cache by default and returns separate total-server, scoring, decoding, road-routing and map-render timings. Set `PATHSNAP_SHORTEST_CACHE_SIZE=0` before starting Uvicorn to reproduce uncached timings. The UI displays segment breaks, road-routing time and map-render time.

**Limits:** Timing is from a GitHub Actions Ubuntu runner and includes instrumentation overhead; compare machines only with a shared environment. The 20-comparison baseline uses precomputed model score exports and excludes original `predict_proba` classifier inference time. These are development-trip experiments, not measured final reserved evaluator performance.

## Reproduction

```bash
python -m unittest discover -s tests -v
python scripts/benchmark_trained_decoding.py --output experiments/viterbi_tuning/student_baseline
python scripts/tune_viterbi.py
python scripts/benchmark_trained_decoding.py --params-file experiments/viterbi_tuning/chosen.json --output experiments/viterbi_tuning/student_tuned
python scripts/profile_trip.py
```

Reproducible workflows: `.github/workflows/viterbi-tuning-speed.yml`, `.github/workflows/demo-speed-smoke.yml`.
