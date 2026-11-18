# Random Forest baseline

This experiment adds the syllabus Random Forest model to the existing candidate
classification workflow. Each candidate is a binary example: does its arc ID
match the reference-aligned arc for this observation?

## Reproduce

```bash
python -m pip install -r requirements-baselines.txt
python scripts/train_baselines.py --models random_forest --output experiments/random_forest
```

Inputs are the existing processed observations, candidate tables and reference
alignment labels. Only aligned training observations with a reference arc in the
candidate set are used for fitting. Ambiguous and unmatched alignments are
excluded. These labels are derived by reference-route alignment; they are not
independently annotated observation-level ground truth.

Five GroupKFold splits hold out entire trips from the 80 training trips. A final
model fits all eligible training examples and is evaluated on the ten student-test
trips. The student-test partition is never used to fit the forest.

The fixed baseline uses 100 trees, maximum depth 16, minimum leaf size 10,
square-root feature selection and seed 42. Depth and leaf limits control model
size and overfitting. No hyperparameter search has been performed. Inputs are
candidate distance, distance rank, fraction along the arc and offset in metres;
trip/arc IDs, reference alignment diagnostics and labels are excluded from features.

The saved scores are binary positive-class probabilities for each candidate,
not a probability distribution normalized across an observation's candidates.
Top-1 selects the highest score, breaking ties by distance rank then arc ID.
Candidate misses count as incorrect in top-1 evaluation on aligned observations.
Binary metrics use aligned observations whose reference occurs in the candidate
set; inspect both metrics together to avoid hiding candidate-generation failures.

Results, fold assignments, package versions, the fitted model and keyed student
scores are saved under `experiments/random_forest/`. A missing raw archive makes
route continuity unavailable (blank in the metrics CSV); supply `--archive` to
include that diagnostic. The classifier has not yet been integrated with Viterbi.

## Recorded run

Five-fold mean candidate macro F1: 0.749668; mean top-1
accuracy: 0.565433. Student-test candidate macro F1:
0.745881; top-1 accuracy: 0.550224; candidate log
loss: 0.161924. See metrics.csv for denominators, exclusions,
confusion counts and timings. These are classifier-only results against the
reference-aligned labels. Continuity was not evaluated because the raw archive
was unavailable. The model was fitted on 1,469,576 candidates from all 80 training trips.
