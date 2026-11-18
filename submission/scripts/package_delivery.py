"""Validate evidence, create a checksum manifest and a clean local delivery ZIP."""
from pathlib import Path
import hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[1]
def main():
 required=['results/final_model.ubj','results/final_model.json','results/final_model.joblib','results/model_metrics.csv','results/route_per_trip.csv','results/route_comparison.csv','results/search_trials.csv','results/search.sqlite3','results/decoder_config.json','results/nvidia-smi.txt','results/environment.json','results/api_all_trips.csv','report/PathSnap_Technical_Report.pdf','results/screenshots/working_ui.jpg','results/screenshots/development_map.jpg','results/screenshots/server_evidence.jpg','results/tests.txt','README.md']
 missing=[p for p in required if not (ROOT/p).is_file()]
 if missing:raise RuntimeError('Incomplete delivery: '+', '.join(missing))
 import csv
 rows=list(csv.DictReader((ROOT/'results/api_all_trips.csv').open()))
 assert len(rows)==10 and all(r['status']=='200' for r in rows)
 env=json.loads((ROOT/'results/environment.json').read_text());assert env['gpu']
 excluded={'__pycache__','.venv','kaggle_data','monitor','monitor_v3','runtime','download','qa'}
 files=[p for p in ROOT.rglob('*') if p.is_file() and not any(d in excluded for d in p.relative_to(ROOT).parts) and p.name not in {'feature_cache.joblib','integration_logistic.joblib','integration_ui.jpg','integration_api_smoke.csv','api_server_pid.txt','SHA256SUMS.txt','PathSnap_Submission.zip'}]
 manifest='\n'.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(ROOT)) for p in sorted(files))+'\n'
 (ROOT/'SHA256SUMS.txt').write_text(manifest)
 output=ROOT.parent/'PathSnap_Submission.zip'
 with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in sorted([*files,ROOT/'SHA256SUMS.txt']):z.write(p,str(Path('PathSnap_complete')/p.relative_to(ROOT)))
 with zipfile.ZipFile(output) as z:
  assert z.testzip() is None
  for line in manifest.splitlines():
   digest,name=line.split('  ',1)
   assert hashlib.sha256(z.read('PathSnap_complete/'+name)).hexdigest()==digest,name
 print(output,output.stat().st_size)
if __name__=='__main__':main()
