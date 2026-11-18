# PathSnap

PathSnap reconstructs directed road routes from GPS observations. The
[submission package](submission/README.md) contains the delivered source,
fitted models, notebooks, run logs, results and
[technical report](submission/report/PathSnap_Technical_Report.pdf).

## Model comparison

**Tuned XGBoost is the deployed model.** It had the highest mean top-1
accuracy in three-fold, trip-grouped training cross-validation. The model
was then evaluated on the same ten student-test trips as the other models.

| Model | Training CV top-1 | Student-test top-1 | Student-test accuracy after HMM/Viterbi | Decoded point continuity |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 86.09% | 86.53% | 89.71% | 96.84% |
| Distance-weighted KNN | 85.62% | 87.14% | 89.81% | 97.39% |
| Random Forest | 86.33% | 87.28% | 89.40% | 97.37% |
| PyTorch MLP | 86.35% | 86.78% | 89.29% | 97.26% |
| XGBoost baseline | 86.46% | 87.00% | 89.70% | 97.37% |
| **XGBoost after TPE (selected)** | 86.50% | 86.82% | 89.74% | 97.33% |

Tuned XGBoost matched **14,643 of 16,317** aligned student-test observations
after Viterbi. KNN had the highest decoded student-test accuracy in the
comparison. Training CV determined the deployable model before the
student-test comparison. The tuned CV score also guided the TPE search.

See the [selection record](submission/results/selection.json),
[full route comparison](submission/results/route_comparison.csv),
[per-trip results](submission/results/route_per_trip.csv) and
[classifier metrics](submission/results/model_metrics.csv).

## Run the submitted demo

Run from the repository root with Python 3.13 available.

**Windows PowerShell**

```powershell
cd submission
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-inference.txt
.\.venv\Scripts\python.exe -m uvicorn pathsnap.api:app --host 0.0.0.0 --port 8000
```

**macOS or Linux**

```sh
cd submission
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements-inference.txt
.venv/bin/python -m uvicorn pathsnap.api:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`, click **Connect**, select a student-test trip,
and click **Reconstruct route**. The API loads the fitted
[UBJSON model](submission/results/final_model.ubj), generates features from raw
GPS and the road graph, and decodes the route live. The page displays
per-trip results. The saved figures above pool all ten student-test trips.

See the [delivery instructions](submission/README.md#two-laptops) for LAN
setup and reserved evaluator inputs. The training evidence includes the
[GPU notebook source](submission/notebooks/gpu_training_source.ipynb),
[recorded GPU run](submission/results/gpu_execution.log), and
[preprocessing notebook with outputs](submission/notebooks/gps_preprocessing_executed.ipynb).
The delivered files retain their recorded
[checksums](submission/SHA256SUMS.txt). The publication notes inside the
package describe its original local handoff.

The table reports the delivered benchmark. A Windows live rerun matched nine
trips exactly and decoded **14,642 of 16,317** observations correctly (89.73%).
On trip `00000077`, near-equal distances changed candidate ranks when the API
rebuilt candidates. Scoring the delivered candidate geometry reproduced the
recorded counts for that trip.

## Earlier development

The previous notebooks, code, and experiments remain in the repository.
The [earlier README](https://github.com/vasishtcoffee/ML-HACKATHON/blob/e5121e7fb8ad4c163bf7354ab0ccddcbeabc91f8/README.md)
documents that workflow and its original evaluation protocol.
