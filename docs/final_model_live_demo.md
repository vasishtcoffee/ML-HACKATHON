# PathSnap final-model live inference — Person 3 handoff

## What is implemented

Laptop 1 can load one of the **existing trained and trusted** model artifacts
and score road-arc candidates for a new trip directly with the fitted model.
Unlike the old distance-baseline and student-test score-file choices, the
`live_*` models call the model's real saved inference routine for each new
candidate. Their probabilities go into the unchanged Viterbi decoder, with
`beta_m=30`, `gap_seconds=5`, and optional per-trip routing cache.

Supported saved artifacts:

| UI model ID | Trusted repository model | Comment |
|---|---|---|
| `live_logistic_regression` | `experiments/classifier_tuning/logistic_regression/logistic_regression.joblib` | Bayesian-tuned |
| `live_knn` | `experiments/classifier_tuning/knn/knn.joblib` | Bayesian-tuned, uses its saved KDTree and smoothed neighbor vote |
| `live_mlp` | `experiments/classifier_tuning/mlp/mlp.joblib` | Bayesian-tuned |
| `live_random_forest` | `experiments/random_forest/random_forest.joblib` | Saved baseline; do not call Bayesian-tuned |
| `live_xgboost` | `experiments/xgboost_tuned/xgboost.joblib` | Tuned |
| `final` | The model selected in `PATHSNAP_FINAL_MODEL` | Only appears when team explicitly sets it |

The final model is **NOT implicitly chosen** by Person 3. The team must make
the final choice based on its model-comparison protocol. Once selected, set
`PATHSNAP_FINAL_MODEL` to one of the `live_*` IDs and restart Laptop 1.

## Install and run

Use a trusted local clone of the complete repository with its committed
model files. Model artifacts are Python joblib pickles: do not accept or load
untrusted joblib uploads.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/requirements-live-demo.txt
# Example ONLY: substitute the team's actually selected model
export PATHSNAP_FINAL_MODEL=live_mlp
# Once the evaluator ZIP is provided, configure its trusted local path:
export PATHSNAP_EVALUATOR_ZIP="/absolute/path/to/evaluator-trip-data.zip"
# Allow the UI's actual origin; see CORS note below
export PATHSNAP_UI_ORIGIN="http://localhost:8001"
python -m uvicorn pathsnap.api:app --host 0.0.0.0 --port 8000
```

The evaluator ZIP is **not committed to GitHub**. It stays on Laptop 1.
The configured archive must contain the same plain-text geometry conventions
as student data, with zero-based node and directed-arc IDs, including
`<trip_id>/<trip_id>.track`, `.nodes` and `.arcs` (an arbitrary outer
directory is allowed). Track rows are
`longitude latitude timestamp_seconds`; nodes are `longitude latitude`;
arcs are `from_node_id to_node_id`. The loader deliberately does not read
`.route` or any reference/ground-truth files, even if they exist in that ZIP.
It preserves original observation IDs, removes invalid coordinates and repeated
timestamps using the same cleaning rules as the training notebook, and
generates 50m road-arc candidate geometry with the **original candidate
feature generator**.

**CORS note:** When Laptop 2 serves `ui/index.html` from a local
`http://localhost:8001` webserver, its browser origin is normally
`http://localhost:8001` even when the backend is on another IP. If Laptop 2
uses a different UI origin, set `PATHSNAP_UI_ORIGIN` accordingly before
starting the backend.

On Laptop 2, from a repository checkout:

```bash
python -m http.server 8001 --bind 127.0.0.1 --directory ui
```

Browse to `http://localhost:8001`, enter **Laptop 1's actual LAN
address** (for example `http://192.168.1.10:8000`) and choose **Connect**.
Select `Evaluator (live inference)`, a discovered evaluator trip, then
`TEAM FINAL` (or a `live_*` model) and **Decode route**.

## Development and evaluator API

`GET /trips` returns:
- `trip_ids`: the ten *student-test* trip IDs, for development;
- `evaluator_trip_ids`: valid trip IDs found in the trusted evaluator ZIP,
  or empty if no evaluator archive is configured;
- `models`: all approved scorer choices; `final` is present only when
  `PATHSNAP_FINAL_MODEL` is configured.

`POST /predict` payload:

```json
{"split": "evaluator", "trip_id": "YOUR_EVALUATOR_TRIP_ID", "model": "final"}
```

For student-test development requests, omit `split` (defaults to
`student_test`) and select any scorer. To confirm that live inference
reproduces known results, select `live_mlp` on trip `00000052`.
Student-test requests return reference-based metrics **server-side only**.
Evaluator responses return a decoded route, Folium map and timing information
but **null `evaluation` and `accuracy`**, because evaluator labels are
not available to this program. The browser does not upload a model or choose
a filesystem path.

The server logs the model, trip, split, decode time and routing/cache/map
timings to the Uvicorn terminal. The UI shows the matched count, number of
route segments/breaks and routing/rendering times.

## Verification

```bash
python -m unittest discover -s tests -v
```

Integration tests independently load the saved tuned classifier artifacts,
score the known trip's candidates, and check they reproduce the committed
student-test score files. A separate synthetic evaluator ZIP test verifies
that the backend:
- discovers the new trip via GET /trips;
- refuses use of precomputed student-test scores in evaluator mode;
- generates candidates, performs real MLP inference and Viterbi decoding;
- returns a Folium route but **no reference metrics**;
- does not open the `.route` member, even when present;
- rejects evaluator requests when no evaluator archive is configured.

See the [GitHub Actions workflow](../.github/workflows/live-inference-ci.yml)
and its recorded test run.

## Remaining final verification

This integration is a **tested software implementation**, not a claim that
the actual evaluator ZIP has been delivered or that a real two-laptop
network demonstration was completed. When the evaluator format is supplied,
verify its exact field layout and member structure before the assessment.
The chosen final classifier must also be tested on actual evaluator trips,
without looking at hidden ground truth or hardcoding expected answers.

If the evaluator provides a different format, adapt the input adapter
(`pathsnap/evaluator.py`) while preserving the trusted model, decoder and
reference-data isolation. Do **not** select the best model by evaluator labels.
