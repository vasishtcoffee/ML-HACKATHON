# XGBoost candidate classifier

This extends the Random Forest training workflow with histogram gradient-boosted
trees. It uses the same four geometric candidate features, reference-derived
labels, five trip-grouped folds and ten student-test trips. Training excludes
ambiguous/unmatched alignments and observations whose reference candidate is
missing. Top-1 evaluation includes candidate misses on aligned observations.
See `random_forest.md` for label limitations and binary metric denominators.

## Reproduce

```bash
# macOS requires an OpenMP runtime:
brew install libomp
python -m pip install -r requirements/requirements-xgboost.txt
python scripts/train_baselines.py --models xgboost --output experiments/xgboost
```

The fixed configuration uses 200 trees, depth 6, learning rate 0.05,
minimum child weight 10, row/column sampling 0.8 and seed 42. The binary logistic
objective produces positive-class candidate probabilities, without normalizing
across candidates at each observation. No scaling is applied. This is an untuned
comparison; student-test results do not determine parameters or tree count.
XGBoost relates to the syllabus gradient boosting topic but is not named in the
syllabus screenshot.

Artifacts include configuration and library versions, trip fold assignments,
metrics, a fitted joblib model and keyed student-test candidate scores. Route
continuity is unavailable without the raw graph archive. These results precede
Viterbi decoding. Use the same package versions when loading the joblib model.

## Recorded results

Five-fold mean top-1: 0.564532; candidate macro F1:
0.754545. Student-test top-1: 0.547834; candidate
macro F1: 0.749448; candidate log loss: 0.160584.
The forest comparison has student-test top-1 0.550224 and macro F1 0.745881.
XGBoost improves binary macro F1 slightly but has lower top-1 in this run;
these results do not establish it as the best route model.
