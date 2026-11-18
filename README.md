# PathSnap

GPS track exploration and preprocessing for the ML hackathon.

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
