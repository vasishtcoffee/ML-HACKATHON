"""Training-only HMM calibration, all-model ten-trip evaluation, real Folium maps."""
import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import pandas as pd
import joblib,folium
from threadpoolctl import threadpool_limits
from pathsnap.pipeline import read_trip,live_candidates,score
from pathsnap.matching import reconstruct,continuity,make_options
from pathsnap.routing import RoadGraph
from pathsnap.evaluation import evaluate
from scripts.train_complete import load_data,take,fit,BASE,OUT
from scripts.view_trip import render_trip
from zipfile import ZipFile

def ref_route(split,trip):
 with ZipFile(ROOT/'data/raw/Student folder.zip') as z:
  name=next(n for n in z.namelist() if n.endswith(f'{split}/{trip}/{trip}.route'))
  return np.loadtxt(z.open(name),dtype=int,ndmin=1).tolist()

def collapse(seq): return [v for i,v in enumerate(seq) if i==0 or v!=seq[i-1]]
def sequence_metrics(seq,reference):
 from rapidfuzz.distance import Levenshtein
 seq=collapse(seq); reference=collapse(reference)
 s,r=set(seq),set(reference); p=len(s&r)/len(s) if s else 0; rec=len(s&r)/len(r) if r else 0
 return dict(route_arc_precision=p,route_arc_recall=rec,route_arc_f1=2*p*rec/(p+rec) if p+rec else 0,
             normalized_route_edit_distance=Levenshtein.distance(seq,reference)/max(len(seq),len(reference),1))

def save_map(t,n,a,result,split,trip,name):
 m,_=render_trip(t,n,a,split,trip,route_segments=result.segments)
 for key,child in list(m._children.items()):
  if isinstance(child,folium.map.LayerControl):del m._children[key]
 # This is an offline development analysis map, separate from the evaluator UI.
 ref=ref_route(split,trip)
 group=folium.FeatureGroup(name='Reference route (development only)',show=True)
 lines=[n[a[i]][:,::-1].tolist() for i in ref]
 folium.PolyLine(lines,color='#22c55e',weight=6,opacity=.55,tooltip='Supplied reference').add_to(group)
 group.add_to(m); folium.LayerControl(collapsed=False).add_to(m)
 m.get_root().html.add_child(folium.Element('<div style="position:fixed;top:15px;left:15px;background:white;padding:12px;z-index:9999;font:14px system-ui;border-radius:8px">GPS: blue · roads: grey · reference: green · prediction: magenta<br>Development analysis; reference layer is excluded from the live API.</div>'))
 (OUT/'maps').mkdir(exist_ok=True); m.save(str(OUT/'maps'/name))

def main():
 selected=json.loads((OUT/'selection.json').read_text())['selected_model']
 final=joblib.load(OUT/'final_model.joblib')
 # A fresh calibration classifier excludes the complete training holdout trips.
 train=load_data('train'); fold=pd.read_csv(OUT/'cv_folds.csv',dtype={'trip_id':str})
 held=fold.loc[fold.fold.eq(1),'trip_id'].tolist(); fitids=fold.loc[fold.fold.ne(1),'trip_id'].tolist()
 sizes=train['labels'].loc[train['labels'].trip_id.isin(held)].groupby('trip_id').size().sort_values()
 calibration=sizes.index[np.linspace(0,len(sizes)-1,4,dtype=int)].tolist()
 # Avoid the longest outlier so calibration has bounded latency; deterministic sizes only.
 calibration=sizes.index[[len(sizes)//5,2*len(sizes)//5,3*len(sizes)//5,4*len(sizes)//5]].tolist()
 name='xgboost' if selected=='xgboost_tuned' else selected
 params=final['params']; a,_=fit(name,params,take(train,fitids),'decoder_calibration')
 del train
 labels=pd.read_parquet(ROOT/'data/processed/reference_labels_train.parquet')
 prepared=[]
 for trip in calibration:
  t,n,arcs=read_trip(ROOT/'data/raw/Student folder.zip','train',trip);c,x=live_candidates(t,n,arcs,'train',trip);p=score(a,x)
  prepared.append((trip,t,n,arcs,c,p,labels.loc[labels.trip_id.eq(trip)],RoadGraph(n,arcs,shortest_cache_size=50000)))
 grid=[]
 for beta in [15.,60.,240.]:
  for jitter in [0.,15.]:
   gap=5.
   correct=total=segments=0; runtime=0
   for trip,t,n,arcs,c,p,l,g in prepared:
    start=time.perf_counter();r,o,_=reconstruct(t,c,p,n,arcs,{'beta_m':beta,'gap_seconds':gap,'jitter_m':jitter},g);runtime+=time.perf_counter()-start
    e=evaluate(l,o,r.arcs_by_observation);correct+=e['viterbi_correct'];total+=e['aligned'];segments+=len(r.segments)
   grid.append(dict(beta_m=beta,gap_seconds=gap,jitter_m=jitter,accuracy=correct/total,aligned=total,correct=correct,segments=segments,seconds=runtime))
   print('CALIBRATION',grid[-1],flush=True)
 pd.DataFrame(grid).to_csv(OUT/'decoder_search.csv',index=False)
 best=max(grid,key=lambda r:(r['accuracy'],-r['segments']));params={'beta_m':best['beta_m'],'gap_seconds':best['gap_seconds'],'jitter_m':best['jitter_m']}
 (OUT/'decoder_config.json').write_text(json.dumps({'parameters':params,'selected_on':'Four disjoint held-out training trips, with calibration model fitted to the other two CV folds','calibration_trips':calibration,'excluded_holdout_trips':held,'fit_trips':fitids,'holdout_accuracy':best['accuracy'],'tie_break':'fewer route segments','limitation':'Small training-only calibration sample; score is used for tuning, not unbiased final evaluation.'},indent=2))
 del prepared,a
 labels=pd.read_parquet(ROOT/'data/processed/reference_labels_student_test.parquet')
 scores={name:pd.read_parquet(OUT/f'{name}_scores.parquet') for name in ['logistic','knn','forest','xgboost','mlp','xgboost_tuned']}
 rows=[]
 for trip in sorted(labels.trip_id.unique()):
  t,n,arcs=read_trip(ROOT/'data/raw/Student folder.zip','student_test',trip)
  c,x=live_candidates(t,n,arcs,'student_test',trip);g=RoadGraph(n,arcs,shortest_cache_size=50000)
  l=labels.loc[labels.trip_id.eq(trip)]; ref=ref_route('student_test',trip)
  for name,s in scores.items():
   s=s.loc[s.trip_id.eq(trip)];joined=c.merge(s,on=['split','trip_id','obs_id','arc_id'],how='left',validate='one_to_one')
   if joined.score.isna().any() or len(joined)!=len(c): raise ValueError('Incomplete keyed scores')
   p=joined.score.to_numpy();start=time.perf_counter();r,o,_=reconstruct(t,c,p,n,arcs,params,g);seconds=time.perf_counter()-start
   e=evaluate(l,o,r.arcs_by_observation)
   top={i:max(v,key=lambda v:(v.probability,-v.candidate_rank,-v.position.arc_id)).position.arc_id for i,v in o.items()}
   cc,ct=continuity(t,arcs,top);vc,vt=continuity(t,arcs,r.arcs_by_observation)
   seq=[arc for segment in r.segments for arc in segment]
   row=dict(model=name,trip_id=trip,observations=len(t),**e,top1_continuous=cc,top1_transitions=ct,viterbi_continuous=vc,viterbi_transitions=vt,segments=len(r.segments),breaks=max(0,len(r.segments)-1),decode_seconds=seconds,**sequence_metrics(seq,ref),**{'top1_'+k:v for k,v in sequence_metrics([top[i] for i in t.obs_id if i in top],ref).items()})
   rows.append(row);pd.DataFrame(rows).to_csv(OUT/'route_per_trip.csv',index=False)
   if name==selected and trip in ['00000009','00000052']:
    save_map(t,n,arcs,r,'student_test',trip,f'development_{trip}.html')
   print('ROUTE',name,trip,e['top1_accuracy'],e['viterbi_accuracy'],seconds,flush=True)
 df=pd.DataFrame(rows);agg=[]
 for name,d in df.groupby('model'):
  agg.append(dict(model=name,trips=len(d),aligned=int(d.aligned.sum()),top1_accuracy=d.top1_correct.sum()/d.aligned.sum(),viterbi_accuracy=d.viterbi_correct.sum()/d.aligned.sum(),trip_macro_viterbi=d.viterbi_accuracy.mean(),top1_continuity=d.top1_continuous.sum()/d.top1_transitions.sum(),viterbi_point_continuity=d.viterbi_continuous.sum()/d.viterbi_transitions.sum(),route_arc_f1=d.route_arc_f1.mean(),normalized_route_edit_distance=d.normalized_route_edit_distance.mean(),top1_route_arc_f1=d.top1_route_arc_f1.mean(),top1_normalized_route_edit_distance=d.top1_normalized_route_edit_distance.mean(),segments=int(d.segments.sum()),breaks=int(d.breaks.sum()),perfect_trips=int(d.viterbi_accuracy.eq(1).sum()),decode_seconds=d.decode_seconds.sum()))
 pd.DataFrame(agg).to_csv(OUT/'route_comparison.csv',index=False)
 selected_rows=df.loc[df.model.eq(selected)].sort_values('trip_id')
 rng=np.random.default_rng(261118);indices=rng.integers(0,len(selected_rows),size=(5000,len(selected_rows)))
 denominator=selected_rows.aligned.to_numpy()[indices].sum(axis=1)
 top=selected_rows.top1_correct.to_numpy()[indices].sum(axis=1)/denominator
 hmm=selected_rows.viterbi_correct.to_numpy()[indices].sum(axis=1)/denominator
 intervals={name:np.quantile(values,[.025,.975]).tolist() for name,values in [('top1_accuracy',top),('hmm_accuracy',hmm),('hmm_minus_top1',hmm-top)]}
 (OUT/'trip_bootstrap.json').write_text(json.dumps({'model':selected,'replicates':5000,'sampling_unit':'complete student-test trip, paired resampling','percentile_95_intervals':intervals,'limitation':'Exploratory uncertainty with only ten development trips; not reserved-evaluation confidence bounds.'},indent=2))
 print(pd.DataFrame(agg).to_string(index=False),flush=True)
if __name__=='__main__':
 with threadpool_limits(limits=4): main()
