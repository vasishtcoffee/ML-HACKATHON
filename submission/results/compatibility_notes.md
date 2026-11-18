# Execution and compatibility notes

- The first cloud attempt expected Kaggle to preserve the uploaded ZIP. Kaggle
  recursively expanded it, including the nested raw data ZIP. The successful
  notebook reconstructs the raw archive and supports either mounted layout.
- Two local CPU ablation attempts terminated in the native XGBoost metadata
  library with Torch loaded in the same macOS process. They produced no usable
  metrics. Importing Torch only for the neural fit path allowed the controlled
  XGBoost CPU ablation to finish. No model parameters were changed for this fix.
- Full fresh model training and Bayesian optimisation run on Kaggle Linux with
  CUDA. Use that notebook for the verified training environment. CPU evaluation,
  raw inference, route decoding and report generation are checked locally.
- A screenshot-browser download exceeded available local disk space. Existing
  Google Chrome was used for actual UI and map inspection instead; no existing
  user browser caches or unrelated files were removed.
- The local integration Logistic Regression is a setup diagnostic, fitted only
  on training trips. It is separate from the final artifact and selection rule.

- The Kaggle source notebook download does not include cell outputs. The actual
  remote execution log is preserved separately as `gpu_execution.log`, and the
  executed raw preprocessing notebook includes outputs. The GPU run is linked
  in `kaggle_run.json`; the source-only notebook is named accordingly.

- Final decoder calibration was repeated in Python 3.13 / XGBoost 3.4.1.
  It selects beta=60, gap=5 s and jitter=15 m. The final student HMM score is
  89.7408%; the earlier XGBoost 3.2 calibration selected beta=15 and its
  preliminary 90.2678% result is superseded. No student-test result chooses
  between these settings; the matching-version training calibration decides.
- Stable UBJSON was exported using the original XGBoost 3.4.1 version. It is
  the default deployment format. The full ten-trip model portability check
  reports maximum probability drift 6.56e-7 and zero changed top-1 decisions.
