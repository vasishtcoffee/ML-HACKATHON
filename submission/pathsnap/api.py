"""Live raw-trip inference on Laptop 1; serves the UI for local review as well."""
import os,time,logging,json
from pathlib import Path
from functools import lru_cache
from zipfile import ZipFile
import joblib
from fastapi import FastAPI,HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel,Field
from threadpoolctl import threadpool_limits
from .model_io import load_artifact
from .pipeline import trip_members,read_trip,live_candidates,score
from .matching import reconstruct
from .server_evaluation import evaluate_fixed
from scripts.view_trip import render_trip
ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=Path(os.environ.get('PATHSNAP_ARCHIVE',ROOT/'data/raw/Student folder.zip'))
SPLIT=os.environ.get('PATHSNAP_SPLIT','student_test')
# A separate server-side reference archive can be supplied for final evaluation.
REFERENCE=Path(os.environ.get('PATHSNAP_REFERENCE_ARCHIVE',ARCHIVE))
MODEL=Path(os.environ.get('PATHSNAP_MODEL',ROOT/'results/final_model.ubj'))
logger=logging.getLogger('uvicorn.error')
app=FastAPI(title='PathSnap Live Inference',version='2.0')
app.add_middleware(CORSMiddleware,allow_origins=[os.environ.get('PATHSNAP_UI_ORIGIN','http://localhost:8001')],allow_methods=['GET','POST'],allow_headers=['Content-Type'])
class Request(BaseModel): trip_id:str=Field(pattern=r'^\d{1,12}$')
@lru_cache(maxsize=1)
def model(): return load_artifact(MODEL)
def parameters():
 p=ROOT/'results/decoder_config.json'
 return json.loads(p.read_text())['parameters'] if p.exists() else {'beta_m':30.0,'gap_seconds':5.0}
def ids():
 with ZipFile(ARCHIVE) as z: return sorted(trip_members(z,SPLIT))
@app.get('/')
def ui(): return FileResponse(ROOT/'ui/index.html')
@app.get('/health')
def health(): return {'status':'ok' if MODEL.is_file() and ARCHIVE.is_file() else 'setup_required','model_available':MODEL.is_file(),'split':SPLIT}
@app.get('/trips')
def trips():
 try: return {'trip_ids':ids(),'split':SPLIT,'model':model()['name'],'inference':'live raw track and graph'}
 except (OSError,ValueError) as exc: raise HTTPException(503,str(exc))
@app.post('/predict')
def predict(request:Request):
 if request.trip_id not in ids(): raise HTTPException(404,'Unknown trip in configured partition')
 started=time.perf_counter(); logger.info('REQUEST trip=%s split=%s live_model=%s',request.trip_id,SPLIT,MODEL.name)
 try:
  with threadpool_limits(limits=4):
   t,n,a=read_trip(ARCHIVE,SPLIT,request.trip_id)
   c,x=live_candidates(t,n,a,SPLIT,request.trip_id)
   p=score(model(),x); inference=time.perf_counter()-started
   dec_start=time.perf_counter(); result,options,g=reconstruct(t,c,p,n,a,parameters()); decode_s=time.perf_counter()-dec_start
   # References are first read AFTER live predictions are completely fixed.
   evaluation=evaluate_fixed(REFERENCE,SPLIT,request.trip_id,t,n,a,options,result.arcs_by_observation)
   m,_=render_trip(t,n,a,SPLIT,request.trip_id,route_segments=result.segments)
   html=m.get_root().render()
 except (OSError,ValueError,KeyError) as exc:
  logger.exception('Prediction failed'); raise HTTPException(503,'Pipeline error: '+str(exc))
 total=time.perf_counter()-started
 logger.info('SUCCESS trip=%s observations=%d matched=%d segments=%d inference=%.3fs decode=%.3fs total=%.3fs accuracy=%s',request.trip_id,len(t),sum(v is not None for v in result.arcs_by_observation.values()),len(result.segments),inference,decode_s,total,evaluation['viterbi_accuracy'] if evaluation else 'unavailable')
 return {'trip_id':request.trip_id,'split':SPLIT,'model':model()['name'],'observations':len(t),'matched_observations':sum(v is not None for v in result.arcs_by_observation.values()),'route_segments':result.segments,'arcs_by_observation':result.arcs_by_observation,'evaluation':evaluation,'runtime_seconds':round(total,3),'timings':{'features_and_inference_seconds':round(inference,4),'decode_seconds':round(decode_s,4)},'map_html':html}
