# Run the PathSnap demo

The API runs the selected KNN with GPS motion features, then uses HMM/Viterbi
to reconstruct a directed route. The checkout contains the trained model and
student-test files. The [model comparison](../README.md#model-comparison)
records the results used to select this model.

## Laptop 1 · FastAPI

Run from the repository root. Python 3.11 or 3.12 is suitable for the live demo.

**Windows PowerShell**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements/requirements-live-demo.txt
.\.venv\Scripts\python.exe -m uvicorn pathsnap.api:app --host 0.0.0.0 --port 8000
```

**macOS or Linux**

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements/requirements-live-demo.txt
.venv/bin/python -m uvicorn pathsnap.api:app --host 0.0.0.0 --port 8000
```

Keep Uvicorn visible to show requests and inference timings during the demo.
The optional `PATHSNAP_DATA_ZIP` variable overrides the student archive path.
The default UI origins are `http://localhost:8001` and
`http://127.0.0.1:8001`. Set `PATHSNAP_UI_ORIGIN` on Laptop 1 to the actual
browser origin if using another port or hostname. Multiple origins can be
comma separated.

## Laptop 2 · Browser

From the repository root, run:

```sh
python -m http.server 8001 --bind 127.0.0.1 --directory ui
```

1. Open `http://localhost:8001`.
2. Enter Laptop 1's LAN address with port 8000, for example
   `http://192.168.1.10:8000`, and click **Connect**. For a one-laptop demo,
   use the prefilled localhost address. Both laptops need network access to
   Laptop 1 on port 8000.
3. For a practice run, pick **Student-test (development)**, trip
   **00000052**, and **Final model**. Click **Decode route**.
4. Show the measured GPS points in blue, the decoded route in pink, and the
   two accuracy values above the map. The map loads Leaflet and map tiles over
   the internet.

## Reserved evaluator trips

Set these environment variables on Laptop 1 **before starting Uvicorn**, once
the organisers supply the data:

| Variable | Server-side input |
| --- | --- |
| `PATHSNAP_EVALUATOR_ZIP` | Trusted ZIP with one `<trip_id>/<trip_id>.track`, `.nodes`, and `.arcs` group per trip. An outer folder is allowed. |
| `PATHSNAP_EVALUATOR_REFERENCE_ZIP` | Trusted private ZIP with one `<trip_id>/<trip_id>.route` per trip, if the organisers supply references for server-side scoring. It can point to the geometry archive if that archive also contains the references. |

The expected geometry format matches the student data. Track columns are
longitude, latitude and timestamp in seconds. Node columns are longitude and
latitude. Arc columns are zero-based source and destination node IDs. Each
reference file is an ordered list of zero-based arc IDs. Keep both
evaluator inputs on Laptop 1. They are configured by the server operator
outside the repository.

The UI selects **Evaluator (live inference)** when it finds evaluator trips
and locks the scorer to **Final model**. The backend generates candidates and
motion features from the new trip, runs the saved classifier and Viterbi,
then aligns the private reference when configured. It returns aggregate
metrics and the predicted route. Without a reference archive, it displays
the route with accuracy marked unavailable.

The actual reserved files and a second laptop were unavailable during
development. The evaluator archive flow was verified with all ten student-test
trips repackaged into separate geometry and reference ZIPs. Confirm the
organisers' file format and test the LAN connection on the evaluation laptops.

## API and verification

`GET /trips` lists the student-test IDs, configured evaluator IDs and model
choices. For live student-test inference:

```json
{"trip_id":"00000052","split":"student_test","model":"final"}
```

Send the JSON to `POST /predict`. The response includes timings, the score
summary and Folium HTML. The evaluator request uses `"split":"evaluator"`
and one of the server's listed evaluator IDs.

```sh
python -m unittest discover -s tests -v
```
