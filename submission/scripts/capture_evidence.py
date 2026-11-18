"""Exercise the actual receiving HTTP service on every development trip."""
import sys,time,json,subprocess,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
import httpx,pandas as pd
OUT=ROOT/'results';(OUT/'screenshots').mkdir(exist_ok=True)

def main():
 port=os.environ.get('PATHSNAP_PORT','8765');url='http://127.0.0.1:'+port
 env=os.environ.copy();env['PATHSNAP_UI_ORIGIN']=url
 (OUT/'runtime').mkdir(exist_ok=True)
 live_log=OUT/'runtime/api_server.log'
 log=live_log.open('w')
 proc=subprocess.Popen([sys.executable,'-m','uvicorn','pathsnap.api:app','--host','127.0.0.1','--port',port],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
 (OUT/'api_server_pid.txt').write_text(str(proc.pid))
 client=httpx.Client(timeout=600)
 for attempt in range(60):
  if proc.poll() is not None: raise RuntimeError(live_log.read_text())
  try:
   r=client.get(url+'/trips');r.raise_for_status();available=r.json();break
  except httpx.HTTPError:time.sleep(1)
 else:raise TimeoutError('API startup timeout')
 (OUT/'api_trips.json').write_text(json.dumps(available,indent=2));rows=[]
 (OUT/'api_predictions').mkdir(exist_ok=True)
 selected=json.loads((OUT/'selection.json').read_text())['selected_model']
 expected=pd.read_csv(OUT/'route_per_trip.csv',dtype={'trip_id':str})
 expected=expected.loc[expected.model.eq(selected)].set_index('trip_id')
 for trip in available['trip_ids']:
  started=time.perf_counter();r=client.post(url+'/predict',json={'trip_id':trip});elapsed=time.perf_counter()-started
  if r.status_code!=200:raise RuntimeError(r.text)
  d=r.json();evaluation=d['evaluation'] or {}
  # No reference labels/arcs may be present in the public prediction response.
  assert 'reference_arc_id' not in r.text and 'reference_route_step' not in r.text
  rows.append({'trip_id':trip,'status':r.status_code,'client_seconds':elapsed,'observations':d['observations'],**evaluation})
  if evaluation:
   for field in ['aligned','candidate_misses','top1_correct','viterbi_correct']:
    assert int(evaluation[field])==int(expected.loc[trip,field]),f'Live/recorded mismatch: {trip} {field}'
  pd.DataFrame({'obs_id':[int(k) for k in d['arcs_by_observation']],'predicted_arc_id':list(d['arcs_by_observation'].values())}).to_csv(OUT/'api_predictions'/f'{trip}.csv',index=False)
  (OUT/'api_predictions'/f'{trip}_route.json').write_text(json.dumps(d['route_segments']))
  safe={k:v for k,v in d.items() if k not in ('map_html','arcs_by_observation','route_segments')}
  (OUT/f'api_response_{trip}.json').write_text(json.dumps(safe,indent=2))
  if trip=='00000052':(OUT/'screenshots/live_prediction_map.html').write_text(d['map_html'])
  pd.DataFrame(rows).to_csv(OUT/'api_all_trips.csv',index=False)
  print('LIVE API',trip,elapsed,evaluation.get('viterbi_accuracy'),flush=True)
 r=client.post(url+'/predict',json={'trip_id':'99999999'});assert r.status_code==404
 r=client.post(url+'/predict',json={'trip_id':'../../secret'});assert r.status_code==422
 (OUT/'api_validation.json').write_text(json.dumps({'unknown_trip_status':404,'invalid_id_status':422,'reference_fields_exposed':False,'successful_trips':len(rows),'live_metrics_match_recorded_route_experiments':True},indent=2))
 (OUT/'api_server.log').write_text(live_log.read_text())
 print('API stays running for visual review at',url,flush=True)
if __name__=='__main__':main()
