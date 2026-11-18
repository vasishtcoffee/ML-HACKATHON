"""Portable label-free inference on a raw trip, including previously unseen trips."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from zipfile import ZipFile
import joblib,pandas as pd
from threadpoolctl import threadpool_limits
from pathsnap.model_io import load_artifact
from pathsnap.pipeline import read_trip,trip_members,live_candidates,score
from pathsnap.matching import reconstruct
from scripts.view_trip import render_trip

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--archive',type=Path,default=ROOT/'data/raw/Student folder.zip')
 p.add_argument('--split',default='student_test');p.add_argument('--trip-id')
 p.add_argument('--model',type=Path,default=ROOT/'results/final_model.ubj')
 p.add_argument('--output',type=Path,default=ROOT/'results/predictions');p.add_argument('--list-trips',action='store_true')
 args=p.parse_args()
 if args.list_trips:
  with ZipFile(args.archive) as z:print('\n'.join(sorted(trip_members(z,args.split))))
  return
 if not args.trip_id:p.error('--trip-id is required unless --list-trips is set')
 start=time.perf_counter();t,n,a=read_trip(args.archive,args.split,args.trip_id)
 c,x=live_candidates(t,n,a,args.split,args.trip_id);model=load_artifact(args.model)
 params=json.loads((ROOT/'results/decoder_config.json').read_text())['parameters']
 with threadpool_limits(limits=4):
  probs=score(model,x);r,_,_=reconstruct(t,c,probs,n,a,params)
 output=args.output/args.split/args.trip_id;output.mkdir(parents=True,exist_ok=True)
 t[['obs_id','longitude','latitude','timestamp_s']].assign(predicted_arc_id=t.obs_id.map(r.arcs_by_observation)).to_csv(output/'predictions.csv',index=False)
 summary={'trip_id':args.trip_id,'split':args.split,'model':model['name'],'observations':len(t),'matched':sum(v is not None for v in r.arcs_by_observation.values()),'route_segments':r.segments,'parameters':params,'inference_seconds':time.perf_counter()-start,'references_read':False}
 (output/'route.json').write_text(json.dumps(summary,indent=2))
 m,_=render_trip(t,n,a,args.split,args.trip_id,route_segments=r.segments);m.save(str(output/'prediction.html'))
 print(json.dumps(summary,indent=2));print('Outputs:',output)
if __name__=='__main__':main()
