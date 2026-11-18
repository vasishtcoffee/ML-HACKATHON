"""Use XGBoost's stable model format for deployment; retain training pickles as evidence."""
import json
from pathlib import Path
import joblib
from .pipeline import FEATURES

def load_artifact(path):
 path=Path(path)
 if path.suffix=='.ubj':
  from xgboost import XGBClassifier
  metadata=json.loads(path.with_suffix('.json').read_text())
  if metadata['features']!=FEATURES:raise ValueError('Feature schema mismatch')
  estimator=XGBClassifier();estimator.load_model(path);estimator.set_params(device='cpu',n_jobs=4)
  return {**metadata,'kind':'sklearn','estimator':estimator}
 return joblib.load(path)
