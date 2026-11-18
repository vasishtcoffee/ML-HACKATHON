# Six-trial XGBoost search

Parameters were selected by mean top-1 accuracy across three training-trip folds.
The selected configuration was then evaluated on the standard five folds and ten
student-test trips. This is not nested cross-validation: the five-fold results
share training data with the search and are not an unbiased estimate of tuning.

Student-test top-1 increased from 0.547834 to 0.548263 (0.043 percentage points).
Candidate macro F1 decreased from 0.749448 to 0.708116. This is a small ranking
change with a substantial threshold-metric tradeoff, not a general improvement.
No maximum accuracy claim is supported by six trials.

The raw archive became available during this run, so continuity was computed
for the tuned model. Untuned continuity is missing; these must not be compared.
The model, scores, trial history, parameters and metrics are preserved here.
