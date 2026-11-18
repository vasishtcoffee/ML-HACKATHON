"""Print live HTTP evidence and receiving-server logs for a native terminal capture."""
import json,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
req=urllib.request.Request('http://127.0.0.1:8765/predict',data=b'{"trip_id":"00000052"}',headers={'Content-Type':'application/json'})
print('PATHSNAP | LIVE RECEIVING-SERVER VERIFICATION\n')
with urllib.request.urlopen(req,timeout=600) as r:
 d=json.load(r);e=d['evaluation'];print('POST /predict -> HTTP',r.status)
 print(f"Trip {d['trip_id']} | model {d['model']} | {d['observations']} GPS points")
 print(f"Classifier top-1: {e['top1_accuracy']:.2%} | HMM: {e['viterbi_accuracy']:.2%}")
 print(f"Aligned: {e['aligned']} | candidate misses: {e['candidate_misses']} | runtime: {d['runtime_seconds']:.3f} s")
 print('Reference-label fields in response:', 'reference_arc_id' in json.dumps(d))
print('\nACTUAL RECEIVING UVICORN LOG\n')
log=(ROOT/'results/runtime/api_server.log').read_text()
(ROOT/'results/api_server.log').write_text(log)
print('\n'.join(log.splitlines()[-4:]))
