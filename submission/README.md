# PathSnap: complete local delivery

This project reconstructs road routes from raw GPS observations with a trained
candidate classifier and directed HMM/Viterbi decoding. All files remain local
apart from the explicitly requested **private Kaggle GPU run**. No GitHub
operations are part of this workflow.

## Install and launch

Use Python 3.13 in a virtual environment:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-inference.txt
python -m uvicorn pathsnap.api:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`, click Connect, choose a trip, then Reconstruct route.
The API creates candidates and features from the raw archive for every request;
it does not load saved candidate scores. Only the model artifact is cached.
Reference data is read separately on the server after predictions are fixed.
Maps returned by the API contain GPS, graph, and predictions, never references.

## Two laptops

Laptop 1: run Uvicorn as above. Set `PATHSNAP_UI_ORIGIN` to Laptop 2's exact UI
origin, such as `http://192.168.1.20:8001`, before starting the API.

Laptop 2: copy `ui/index.html`, serve its folder with
`python -m http.server 8001 --bind 0.0.0.0`, open its page, and enter Laptop 1's
LAN URL, such as `http://192.168.1.10:8000`. Click Connect, then select a trip.
Allow TCP 8000 on Laptop 1. The server console logs each request and result.
Alternatively open Laptop 1's served UI directly from Laptop 2.

## Reserved evaluator deployment

Reserved evaluator trips are not included in the supplied student archive.
They are not opened, tuned on, or represented by invented results. At evaluation
time set the archive and actual partition directory name, for example:

```sh
export PATHSNAP_ARCHIVE='/absolute/path/Evaluator folder.zip'
export PATHSNAP_SPLIT='evaluator_test'
export PATHSNAP_REFERENCE_ARCHIVE='/absolute/path/Evaluator references.zip'
export PATHSNAP_UI_ORIGIN='http://192.168.1.20:8001'
python -m uvicorn pathsnap.api:app --host 0.0.0.0 --port 8000
```

Required layout within the archive is `[optional prefixes]/<split>/<trip_id>/`
with `<trip_id>.track`, `.nodes`, and `.arcs`. The server enumerates actual trip
IDs. A separate reference archive can contain matching `.route` files in the
same layout. If reference files are absent, predictions work and accuracy is
reported as unavailable. The reference archive stays on Laptop 1.
The final model and decoder settings must be frozen before reserved inputs are
provided. Do not use evaluator outputs for further model selection.

## Reproduce experiments

The private executed Kaggle notebook is in `deployment/kaggle/pathsnap_gpu.ipynb`.
It uses the source bundle and raw student archive uploaded privately, reconstructs
ZIPs expanded by Kaggle ingestion, rebuilds preprocessing/candidates/labels,
trains five model families with three trip-grouped CV folds, and runs twelve
sequential Optuna TPE trials. CUDA is required for that notebook. XGBoost and the
PyTorch MLP actually train on GPU; linear, neighbour and forest models use CPU.
No repository clone or remote source fetch is used.

Extract a fresh copy before reproducing so the delivered experiment evidence remains available.
For model retraining, additionally install `requirements.txt` (includes PyTorch and notebook execution).
For a local reproduction, first execute `notebooks/gps_preprocessing.ipynb` from
the project root, then:

```sh
python scripts/generate_candidates.py
python scripts/generate_reference_labels.py
python scripts/train_complete.py
python scripts/evaluate_complete.py
python scripts/capture_evidence.py
# Capture the actual UI, map and terminal as described in report/README.md.
python scripts/build_report.py
python -m unittest discover -s tests -v
```

Training uses the 80 training trips. The ten student-test trips are used for
reported development comparisons, never hyperparameter search. The direction ablation holds XGBoost parameters and CPU backend fixed, comparing
the three heading-related features against the same folds with those inputs zeroed.
A fresh
calibration classifier fitted to two training folds supplies scores for HMM
parameter selection on four trips from the excluded fold.

## Standalone inference without reference access

```sh
python scripts/predict_trip.py --list-trips
python scripts/predict_trip.py --split student_test --trip-id 00000052
```

Use `--archive` and `--split` for new raw inputs. This command writes per-observation
CSV predictions, reconstructed arc sequences and a Folium map without reading
any reference route. Its configuration comes from the frozen final decoder.

## How to read the results

`results/model_metrics.csv` records per-fold CV, full-training diagnostics and
student-test metrics. Observation top-1 includes candidate misses as errors.
Candidate macro F1 is a binary match/non-match metric at threshold 0.5 on
unambiguous aligned observations with a positive candidate available. It is
**not** F1 over globally meaningful arc IDs (arc IDs are local to each graph).
Ambiguous and unmatched observations are excluded and counted separately.

`results/search_trials.csv` and `search.sqlite3` preserve every Bayesian trial.
`selection.json` records the training-CV selection decision. Tuned CV estimates
are selection scores, not unbiased nested-CV estimates. `route_per_trip.csv`
records every model on every student trip; `route_comparison.csv` aggregates
point accuracy, transition continuity, route-arc F1, ordered edit distance,
segments, and runtime. Route-arc F1 uses unique directed arc sets; ordered edit
distance uses consecutive-duplicate-collapsed arc sequences. Gap boundaries can
create legitimate segments, so segment count alone is not an accuracy metric.
Timing comparisons with a shared graph cache are warm-cache development timings.

`results/maps/` contains development-only Folium comparisons with a reference
layer. The live UI never receives these reference-bearing maps.

## Provenance and originality

This is AI-assisted implementation and analysis. It adapts the user's existing
local PathSnap preprocessing, geometric candidate generation, reference alignment,
directed routing, Viterbi and Folium viewer. It adds fresh direction-aware
features, a fresh five-family training pipeline, actual GPU training, persisted
Bayesian search, model selection, live raw-trip inference, a new UI, route
experiments and a consolidated evidence-based report. No other team's code,
weights or results were imported. Existing experiment results are not presented
as newly run work. Third-party libraries implement standard algorithms and are
listed in the dependency manifests. AI assistance must be disclosed; team
members must review and be able to defend the work. A plagiarism-free guarantee
or a claim about cross-team similarity cannot be made without organisers' checks.

## Boundaries of verification

All student-test trips can be exercised locally through the actual API.
Reserved evaluation accuracy, physical two-laptop LAN connectivity, event-time
viva, and publication/submission are separate checks requiring their inputs or
human participation. The documents request public GitHub publication, but the
user's explicit instruction defers all GitHub activity. This package is prepared
locally for later review and publication only when requested.

The deployment artifact is `results/final_model.ubj`, exported using the original
XGBoost 3.4.1 runtime, with its feature metadata in `final_model.json`. Training
joblib artifacts are retained as experiment evidence. The verified inference
environment is Python 3.13 with XGBoost 3.4.1 and scikit-learn 1.7.2.

Report evidence provenance is recorded in `results/screenshots/EVIDENCE_PROVENANCE.json`.
The UI image is the actual integration-model browser screenshot. The geographical
figure is an export of the actual generated Folium layers. The server-log figure
is rendered from real recorded output, not claimed as a native CMD screenshot.
Browser control became unavailable and native Terminal access was rejected by
the computer-use safety tool; capture the required receiving-terminal screenshot
on the evaluation laptop before submitting.
