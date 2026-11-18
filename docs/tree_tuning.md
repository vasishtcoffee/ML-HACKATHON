# Tree-model tuning workflow

Install `requirements/requirements-tuning.txt` in the project environment.
On macOS, XGBoost also needs `brew install libomp`.

```bash
python scripts/tune_tree_models.py --model random_forest --trials 6
python scripts/tune_tree_models.py --model xgboost --trials 6
python scripts/compare_tree_models.py
```

Each search uses three folds grouped by training trip and selects mean top-1
accuracy on aligned observations. Candidate misses count as incorrect. Student
trips are evaluated after selection and must not guide repeated search choices.
Six trials are a small exploratory budget, not evidence of maximum accuracy.

The next Random Forest search preserves the baseline's 100 trees. Earlier code
forced 60 trees; no completed forest search is being reported in these commits.
Searches now save their fold mapping and protocol before training, and write
trial history after every completed trial. Output directories are overwritten by
a new run, so preserve previous experiments before repeating a search.

The selected parameters are evaluated with five folds and refitted on all
eligible training observations. These five-fold metrics are not nested CV:
parameter selection has already used those trips. A stronger estimate would
require nested trip folds or an untouched validation partition.

The completed six-trial XGBoost search produced only a 0.043 percentage-point
student top-1 change and reduced binary macro F1. The current data do not justify
replacing Random Forest for top-1 performance. Keep all configurations and
results, including unsuccessful experiments. Candidate scores remain binary
probabilities; decoder integration must specify how they become emissions.

Further gains should investigate motion and heading agreement features and
directed-network transition scores, then compare them using training-trip folds.
These require geometry and consistent handling of GPS gaps. Do not introduce
reference alignment diagnostics or reference arc IDs as predictor features.
