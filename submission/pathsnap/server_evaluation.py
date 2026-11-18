"""Server-only reference alignment. Call only after all predictions are fixed."""
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from scripts.generate_reference_labels import align_trip
from .evaluation import evaluate

def reference_labels(archive,split,trip_id,track,nodes,arcs):
 with ZipFile(archive) as z:
  suffix=f'{split}/{trip_id}/{trip_id}.route'
  names=[n for n in z.namelist() if n==suffix or n.endswith('/'+suffix)]
  if not names: return None
  if len(names)!=1: raise ValueError('Duplicate reference route')
  route=np.loadtxt(z.open(names[0]),dtype=np.int64,ndmin=1)
 if route.ndim!=1 or not len(route) or (route<0).any() or (route>=len(arcs)).any(): raise ValueError('Invalid reference route')
 t=track.assign(split=split,trip_id=trip_id)
 return align_trip(t,nodes,arcs,route)

def evaluate_fixed(archive,split,trip_id,track,nodes,arcs,options,predictions):
 labels=reference_labels(archive,split,trip_id,track,nodes,arcs)
 return evaluate(labels,options,predictions) if labels is not None else None
