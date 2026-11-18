"""Training-trip-only directional feature ablation; no student-test tuning."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import pandas as pd
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
from scripts.train_complete import load_data,take,fit,evaluate,BASE,OUT

def main():
 d=load_data('train'); rows=[]
 output=OUT/'direction_ablation.csv'
 if output.exists():rows=pd.read_csv(output).to_dict('records')
 for variant in ['no_direction','full_direction']:
  xx=d['x'].copy()
  if variant=='no_direction':d['x'][:,[4,5,6]]=0
  for k,(i,j) in enumerate(GroupKFold(3).split(d['labels'],groups=d['labels'].trip_id),1):
   if any(r['model']=='xgboost_'+variant and r['fold']==k for r in rows):continue
   fi=d['labels'].iloc[i].trip_id.unique();vi=d['labels'].iloc[j].trip_id.unique()
   a,_=fit('xgboost',BASE['xgboost'],take(d,fi),'direction_ablation')
   row,_=evaluate(a,take(d,vi),'xgboost_'+variant,'cv',k);rows.append(row)
   pd.DataFrame(rows).to_csv(output,index=False)
   print(row,flush=True)
  d['x']=xx
if __name__=='__main__':
 with threadpool_limits(limits=4):main()
