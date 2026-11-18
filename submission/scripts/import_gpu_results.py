"""Import the completed private Kaggle run without mutating shared input files."""
import ast,hashlib,json,os,shutil,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'deployment/download';archive=D/'pathsnap_gpu_results.zip'
assert archive.exists(),archive
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None
 for item in z.infolist():
  target=(ROOT/item.filename).resolve()
  if not target.is_relative_to(ROOT):raise ValueError('Unsafe archive path')
  if target.is_file() and target.stat().st_nlink>1:target.unlink()
  z.extract(item,ROOT)
# Break deduplicated raw input links as well, so the delivery folder is independent.
for p in (ROOT/'data').rglob('*'):
 if p.is_file() and p.stat().st_nlink>1:
  temp=p.with_name(p.name+'.independent');shutil.copyfile(p,temp);os.replace(temp,p)
executed=D/'__notebook__.ipynb'
if executed.exists():shutil.copy2(executed,ROOT/'notebooks/gpu_training_executed.ipynb')
notebook=json.loads((executed if executed.exists() else ROOT/'deployment/kaggle/pathsnap_gpu.ipynb').read_text())
for cell in notebook['cells']:
 if cell['cell_type']!='code':continue
 tree=ast.parse(''.join(cell['source']))
 for n in ast.walk(tree):
  if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='write_text' and n.args and isinstance(n.args[0],ast.Constant) and isinstance(n.args[0].value,str) and 'def load_data(split)' in n.args[0].value:
   (ROOT/'results/gpu_training_source.py').write_text(n.args[0].value)
shutil.copy2(ROOT/'deployment/gpu_live.log',ROOT/'results/gpu_execution.log')
(ROOT/'results/kaggle_run.json').write_text(json.dumps({'url':'https://www.kaggle.com/code/muchat007/pathsnap-original-gpu-complete-20261118','version':3,'visibility':'private','status':'completed','gpu_archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest()},indent=2))
print((ROOT/'results/selection.json').read_text())
print((ROOT/'results/environment.json').read_text())
