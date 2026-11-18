# Delivered and externally pending

## Completed and evidenced

- Raw data exploration, validation, preprocessing and geographical visualisation
  for all 80 training and ten student-test trips.
- Fresh geometry/motion feature engineering, with a controlled training-only
  ablation showing the value of direction.
- Five trained algorithm families with configurations, seeds, three trip-grouped
  CV folds, training diagnostics, ten-trip development metrics and saved models.
- Twelve recorded sequential Optuna TPE trials, persisted search database,
  training-CV selection and full-data final refit on a real Tesla T4 Kaggle run.
- Training-only decoder calibration; directed HMM/Viterbi, shortest paths,
  missing-candidate handling, gaps and same-arc GPS jitter handling.
- Classifier-only versus HMM comparisons for every model on all ten student
  trips, point/sequence metrics, continuity and uncertainty analysis.
- Stable XGBoost UBJSON deployment artifact, with exact top-1 portability checks.
- Live raw-trip FastAPI inference, server-only evaluation, a working UI, a
  two-laptop launcher/client check and standalone prediction export.
- All ten final-model HTTP predictions checked against experiment counts.
- Twenty-eight tests passing, real execution logs and an actual integration UI
  browser screenshot.
- One consolidated 21-page PDF report with actual charts, map-layer export,
  per-model analysis, Bayesian tuning, decoder analysis, deployment and limits.
- Local clean ZIP and SHA-256 manifest for the delivered source and evidence.

## Requires the evaluation setting or human review

- Reserved evaluator trips were not supplied. Their final performance must be
  measured only at evaluation, using the frozen model/configuration.
- Physical two-laptop LAN connectivity must be demonstrated on the actual
  laptops. Instructions and a client verification utility are included.
- A native receiving CMD/terminal screenshot remains to be captured there.
  Browser control became unavailable in this chat and native Terminal access
  was rejected by the computer-use safety tool. The report instead clearly
  labels its actual integration UI screenshot, exported Folium layers and
  rendered real receiving-log evidence; none is misrepresented.
- Team members must review and understand the AI-assisted work and complete
  the viva; cross-team originality checks are controlled by the organisers.
- Public GitHub publication is explicitly deferred by the user's instruction.
  No GitHub modifications, pushes, commits or publication were performed.

The package is a completed computational implementation and measured development
study, not a claim that these event-only requirements have already happened.
