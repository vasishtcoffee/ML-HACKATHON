# Earlier development Viterbi benchmark

This benchmark documents the earlier KNN selection. The
[submission comparison](../../README.md#model-comparison) uses a newer training
pipeline and records the selection of tuned XGBoost.

The [earlier development comparison](https://github.com/vasishtcoffee/ML-HACKATHON/blob/e5121e7fb8ad4c163bf7354ab0ccddcbeabc91f8/README.md#model-comparison)
ranks the highest-scoring saved variant from each of the five classifier
families. [Aggregate results](aggregate.csv) and [per-trip results](per_trip.csv)
include all 13 tested variants.

## Protocol

- All models fit only on the 80 training trips. The benchmark uses the same
  ten student-test trips, 19,160 GPS observations, 209,582 candidates, and
  directed road graphs for every model.
- Accuracy uses the 16,317 unambiguously aligned GPS observations. The six
  aligned observations whose reference arc was outside the candidate set
  count as wrong. Alignment excluded 2,822 ambiguous and 21 unmatched
  observations. Accuracy pools observation counts across trips.
- Every run calls the saved estimator on the trip's candidates and checks
  the scores against its committed export by the full candidate key. The
  geometry and reference alignment are shared. The enhanced variants add
  GPS heading or motion to the original four geometry features.
- Every variant uses `beta_m=60`, `gap_seconds=10`, and a 20,000-entry
  per-trip routing cache. Those decoder parameters came from the
  [training-only calibration](../../docs/person3_viterbi_tuning_speed.md).
  The `--decoder original` option runs the earlier 30 m / 5 s setting.
- Direct-arc continuity checks same-arc or directly connected predictions
  on adjacent observations outside GPS gaps over five seconds and row
  breaks. A valid route with intermediate arcs can fail this stricter test.
  The route-pair F1 compares adjacent **ordered** arc pairs with the reference
  sequence, counting repeated transitions up to their observed frequency.
  Independent top-1 uses consecutive valid observations. Viterbi also
  includes the intermediate arcs it reconstructs. Neither measure crosses
  a predicted segment boundary.
- Live scorer time sums model scoring, feature construction for enhanced
  variants and first model load across the ten trips. Decode time and total
  benchmark time are recorded separately. Timing was measured on the same
  Windows laptop with Python 3.12 and depends on system load. Folium
  rendering and HTTP transport are outside the benchmark.

## Reproduce

```sh
python -m pip install -r requirements/requirements-live-demo.txt
python scripts/benchmark_final_models.py
```

To reproduce the training-fold candidate metrics for the selected heading
and motion models, install `requirements/requirements-classifier-tuning.txt`
and run `python scripts/train_motion_models.py --selected-cv-only`.
The fold means are post-selection diagnostics on the original three search
folds. The student-test ranking supplies the held-out comparison.

## Evidence and earlier runs

- [Motion configuration](../motion_features/selection.json),
  [training-fold metrics](../motion_features/selected_cv_metrics.csv) and
  [student-test candidate metrics](../motion_features/metrics.csv) record
  the feature-enhanced variants, including precision, recall, log loss and
  confusion counts.
- The earlier [KNN Bayesian search](../classifier_tuning/knn/search_trials.csv)
  used Optuna TPE with group-aware training folds for the geometry-only KNN.
  The enhanced KNN used the subsequent
  [motion-feature search](../motion_features/validation_trials.csv) on the
  same trip folds. Both are preserved in the benchmark.
- The tuned Random Forest uses trial 4 from cell 22 of the
  [full-team Kaggle notebook](../../notebooks/pathsnap_full_team_run.ipynb). Its
  [trial log](../random_forest_tuned/notebook_trials.json) and
  [CV output](../random_forest_tuned/notebook_cv_macro_f1.csv) are from that run.
  The [local refit metrics](../random_forest_tuned/refit_metrics.json)
  use the committed processed data. Install `requirements/requirements-tuning.txt`
  and run `python scripts/tune_tree_models.py --model random_forest --refit-notebook`
  to reproduce the refit. See the [original XGBoost metrics](../xgboost/metrics.csv),
  [tuned XGBoost metrics](../xgboost_tuned/metrics.csv) and
  [search trials](../xgboost_tuned/search_trials.csv) for the boosting run.
- The earlier [five-model run](https://github.com/vasishtcoffee/ML-HACKATHON/actions/runs/44256618239)
  used geometry-only scores and 30 m / 5 s Viterbi. The [Kaggle notebook's
  base revision](https://github.com/vasishtcoffee/ML-HACKATHON/tree/9fc9298037430e3a11f95ae24b7639306a42a74f)
  predates the motion-feature branch. Their numbers describe those earlier runs.

Codex assisted with integration, experiment reproduction and verification.
No reserved evaluator data was supplied or used for model selection.
