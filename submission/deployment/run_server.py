"""Laptop 1 launcher. All model and reference access stays in this process."""
import argparse,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--archive',type=Path,default=ROOT/'data/raw/Student folder.zip')
p.add_argument('--split',default='student_test');p.add_argument('--reference-archive',type=Path)
p.add_argument('--model',type=Path,default=ROOT/'results/final_model.ubj')
p.add_argument('--ui-origin',default='http://localhost:8001');p.add_argument('--port',type=int,default=8000)
p.add_argument('--host',default='0.0.0.0')
a=p.parse_args();env=os.environ.copy()
for key,value in {'PATHSNAP_ARCHIVE':a.archive.resolve(),'PATHSNAP_SPLIT':a.split,'PATHSNAP_MODEL':a.model.resolve(),'PATHSNAP_REFERENCE_ARCHIVE':(a.reference_archive or a.archive).resolve(),'PATHSNAP_UI_ORIGIN':a.ui_origin}.items():env[key]=str(value)
raise SystemExit(subprocess.call([sys.executable,'-m','uvicorn','pathsnap.api:app','--host',a.host,'--port',str(a.port)],cwd=ROOT,env=env))
