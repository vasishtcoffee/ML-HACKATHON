# PathSnap

GPS track exploration and preprocessing for the ML hackathon.

## Setup

The additional pinned environment is verified on macOS with Python 3.11. From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name pathsnap --display-name "Python (PathSnap)"
python -m notebook
```

Open
`notebooks/gps_preprocessing.ipynb`, select **Python (PathSnap)** and run the
cells in order. Map tiles require internet access.

`requirements.in` includes the existing `requirements-preprocessing.txt`
versions and adds the Jupyter interface. `requirements.txt` pins the resolved dependencies for the Python
3.11 macOS environment. On other platforms, resolve `requirements.in` in a
fresh environment to account for platform-specific Jupyter dependencies. Model-training and API packages will be added when those
components exist.


## Reproduce the preprocessing

1. Install the dependencies with `python -m pip install -r requirements-preprocessing.txt`. Use a Python 3.13 notebook kernel for the tested environment.
2. Copy the supplied archive to `data/raw/Student folder.zip` without extracting it. This is the same relative path on every teammate's checkout. For another location, set `PATHSNAP_DATA_ZIP` to the full path.
3. Open `notebooks/gps_preprocessing.ipynb` from the project root or the `notebooks/` directory and run all cells. The notebook regenerates the files in `data/processed/`, including a local Folium HTML map. Map tiles need internet access.

## Processed data

| File | Grain and contents |
|---|---|
| `observations.csv.gz` | One row per retained GPS observation, keyed by `split`, `trip_id`, and `obs_id`. Includes longitude, latitude, timestamp in seconds, interval and gap flags, distance in metres, speed in metres per second, and initial heading in degrees. |
| `trip_summary.csv` | One row per trip with GPS quality counts, bounds, gap statistics, road network size and connectivity, and reference discontinuity counts. |
| `split_summary.csv` | Counts and sampling statistics for the training and student-test splits. |
| `archive_inventory.csv` | File availability and uncompressed sizes by split, trip and file type. |
| `sampling.png` | GPS observations per trip and sampling-interval distributions by split. |
| `sample_trip_map.html` (generated locally) | Interactive GPS, nearby road-network and reference-route layers for a representative training trip. |

```python
import pandas as pd

observations = pd.read_csv(
    "data/processed/observations.csv.gz",
    dtype={"trip_id": str},
)
```

Keep `trip_id` as a string to preserve leading zeros. Empty motion values mean the feature is undefined. The notebook's `load_graph` function regenerates graph features for a trip from its original node and arc rows. The observations export carries no reference arc IDs or observation-level labels.

## Preprocessing decisions

- Validate coordinate bounds, timestamps, graph endpoints and reference arc IDs. Keep original zero-based observation, node and arc IDs. Preserve repeated directed arc endpoints and reference traversal order.
- Sort observations by timestamp, keep the first valid row at repeated timestamps, and flag intervals longer than five seconds. Leave speed and heading empty across gaps or breaks in original observation order. Keep GPS observations at their recorded positions.
- Calculate distances in metres with the haversine formula. Arc lengths approximate the distance between road-node endpoints. Audit weak connectivity and record breaks in directed reference adjacency.
- Processed all 80 training trips and 10 student-test trips. The run kept 208,417 training and 19,160 student-test observations, and found 1,216 training gaps and 44 student-test gaps longer than five seconds. The longest gap was 153,064 seconds. There were 13 directed reference breaks in training and one in student testing.
- Preserve these partitions for the next phase. Cross-validation and learned preprocessing should use trip-grouped training folds. Evaluator trips are reserved for final evaluation.

Implementation and verification were assisted by Codex.

## Updating dependencies

With uv installed, resolve and install changes in a Python 3.11 environment:

```bash
uv pip compile requirements.in --python .venv/bin/python --output-file requirements.txt
uv pip sync requirements.txt --python .venv/bin/python
uv pip check --python .venv/bin/python
```

Review the dependency diff and rerun the notebook before committing an update.

## Road-arc candidates

`data/processed/candidates_train.parquet` and `data/processed/candidates_student_test.parquet` contain one row per GPS observation and nearby directed arc. The default search includes every arc whose straight segment is within 50 metres of the GPS point in a trip-local metric projection. The radius is an initial development setting and can be changed with `--radius-m`. The generator uses only the GPS coordinates and graph.

| Columns | Meaning |
|---|---|
| `split`, `trip_id`, `obs_id`, `arc_id` | Join key for classifier scores. Arc IDs are zero-based rows within each trip's original arc file. |
| `candidate_rank`, `distance_to_arc_m` | One-based rank within the observation and perpendicular or endpoint-clamped distance to the arc. Ties use arc ID. |
| `fraction_along_arc`, `offset_from_arc_start_m` | Position along the directed arc from its source node (0) to its destination node (1). Offset is the fraction times the arc's haversine endpoint length. |
| `projected_longitude`, `projected_latitude` | Closest point on the straight arc segment, in degrees. |

Use the processed observations table for timestamps, order, and GPS gap flags. An observation with no arc within the radius has no candidate rows. Preserve it with a left join on `split`, `trip_id`, and `obs_id`. The per-trip counts are in `data/processed/candidate_coverage.csv`. Coverage measures whether any candidate exists, and reference-arc recall requires separate label alignment.

To reproduce, run the preprocessing notebook, then install the candidate export dependency in the same environment and run:

```sh
python -m pip install -r requirements-candidates.txt
python scripts/generate_candidates.py
```

Load either table with `pandas.read_parquet()`. Parquet preserves the leading zeros in `trip_id`.
