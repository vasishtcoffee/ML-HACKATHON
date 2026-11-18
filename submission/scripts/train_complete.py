"""Fresh five-family experiments, grouped CV and genuine sequential TPE search."""
import os,sys,json,time,platform,subprocess,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import joblib,optuna
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,log_loss,confusion_matrix
from threadpoolctl import threadpool_limits
from pathsnap.pipeline import FEATURES,KEY,read_trip,features,score
SEED=261118
OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)
BASE={
 'logistic':{'C':1.0},
 'knn':{'n_neighbors':25,'max_fit_rows':160000},
 'forest':{'n_estimators':80,'max_depth':16,'min_samples_leaf':10},
 'xgboost':{'n_estimators':240,'max_depth':5,'learning_rate':0.07,'min_child_weight':12,'subsample':0.85,'colsample_bytree':0.9,'reg_lambda':5.0},
 'mlp':{'epochs':12,'learning_rate':0.001,'weight_decay':0.0001,'batch_size':16384}}

def write(name,obj): (OUT/name).write_text(json.dumps(obj,indent=2,default=str)+'\n')

def load_data(split):
 c=pd.read_parquet(ROOT/f'data/processed/candidates_{split}.parquet')
 labels=pd.read_parquet(ROOT/f'data/processed/reference_labels_{split}.parquet')
 chunks=[]; xx=[]
 for trip,rows in c.groupby('trip_id',sort=True):
  t,n,a=read_trip(ROOT/'data/raw/Student folder.zip',split,trip)
  chunks.append(rows); xx.append(features(t,rows,n,a))
 c=pd.concat(chunks,ignore_index=True)
 x=np.concatenate(xx)
 l=labels.merge(c[KEY[:-1]].drop_duplicates().assign(any_candidate=True),on=KEY[:-1],how='left',validate='one_to_one')
 c=c.merge(labels[KEY[:-1]+['reference_arc_id','alignment_status','reference_in_candidates','usable_for_training']],on=KEY[:-1],validate='many_to_one')
 y=c.arc_id.eq(c.reference_arc_id).fillna(False).to_numpy(dtype='uint8')
 usable=(c.alignment_status.eq('aligned')&c.reference_in_candidates).to_numpy()
 return dict(c=c,x=x,y=y,usable=usable,labels=l)

def take(data,trips):
 m=data['c'].trip_id.isin(trips).to_numpy()
 return dict(c=data['c'].loc[m].reset_index(drop=True),x=data['x'][m],y=data['y'][m],usable=data['usable'][m],labels=data['labels'].loc[data['labels'].trip_id.isin(trips)])

def metrics(data,p):
 c=data['c']; m=data['usable']; y=data['y'][m]; pp=p[m]; pred=pp>=0.5
 w=c[KEY+['candidate_rank']].assign(score=p).sort_values(['score','candidate_rank','arc_id'],ascending=[False,True,True]).drop_duplicates(KEY[:-1])
 aligned=data['labels'].loc[data['labels'].alignment_status.eq('aligned')]
 z=aligned.merge(w[KEY],on=KEY[:-1],how='left',validate='one_to_one')
 correct=z.arc_id.eq(z.reference_arc_id).fillna(False)
 trip_acc=correct.groupby(z.trip_id).mean()
 cm=confusion_matrix(y,pred,labels=[0,1]).ravel()
 return dict(top1_accuracy=float(correct.mean()),trip_macro_top1=float(trip_acc.mean()),macro_f1=float(f1_score(y,pred,average='macro',zero_division=0)),
  precision=float(precision_score(y,pred,zero_division=0)),recall=float(recall_score(y,pred,zero_division=0)),positive_f1=float(f1_score(y,pred,zero_division=0)),
  candidate_accuracy=float(accuracy_score(y,pred)),log_loss=float(log_loss(y,pp,labels=[0,1])),aligned=len(z),
  candidate_misses=int((~z.reference_in_candidates).sum()),candidate_miss_rate=float((~z.reference_in_candidates).mean()),
  ambiguous_excluded=int(data['labels'].alignment_status.eq('ambiguous').sum()),unmatched_excluded=int(data['labels'].reference_arc_id.isna().sum()),
  tn=int(cm[0]),fp=int(cm[1]),fn=int(cm[2]),tp=int(cm[3]))

def fit(name,params,data,tag,validation=None):
 np.random.seed(SEED)
 if name=='mlp':
  import torch
  torch.manual_seed(SEED)
 mask=data['usable']; x=data['x'][mask]; y=data['y'][mask]
 start=time.perf_counter(); history=[]
 artifact=dict(name=name,params=params,features=FEATURES,kind='sklearn',seed=SEED)
 if name=='logistic': estimator=make_pipeline(StandardScaler(),LogisticRegression(C=params['C'],max_iter=300,random_state=SEED))
 elif name=='knn':
  if len(x)>params['max_fit_rows']:
   indices=np.random.default_rng(SEED).choice(len(x),params['max_fit_rows'],replace=False); x=x[indices]; y=y[indices]
  estimator=make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=params['n_neighbors'],weights='distance',algorithm='kd_tree',n_jobs=4))
 elif name=='forest': estimator=RandomForestClassifier(**params,n_jobs=4,random_state=SEED,max_features='sqrt')
 elif name=='xgboost':
  from xgboost import XGBClassifier
  estimator=XGBClassifier(**params,device='cuda' if os.environ.get('PATHSNAP_REQUIRE_GPU')=='1' else 'cpu',tree_method='hist',objective='binary:logistic',eval_metric='logloss',n_jobs=4,random_state=SEED)
 elif name=='mlp':
  scaler=StandardScaler().fit(x); x=scaler.transform(x).astype('float32')
  device='cuda' if torch.cuda.is_available() else 'cpu'
  torch.set_num_threads(4)
  net=torch.nn.Sequential(torch.nn.Linear(len(FEATURES),64),torch.nn.ReLU(),torch.nn.Linear(64,32),torch.nn.ReLU(),torch.nn.Linear(32,1)).to(device)
  opt=torch.optim.AdamW(net.parameters(),lr=params['learning_rate'],weight_decay=params['weight_decay'])
  xx=torch.from_numpy(x).to(device); yy=torch.from_numpy(y.astype('float32')).to(device)
  if validation is not None:
   vm=validation['usable']; vx=torch.from_numpy(scaler.transform(validation['x'][vm]).astype('float32')).to(device)
   vy=torch.from_numpy(validation['y'][vm].astype('float32')).to(device)
  for epoch in range(params['epochs']):
   net.train(); order=torch.randperm(len(x),device=device); total=0
   for idx in order.split(params['batch_size']):
    opt.zero_grad(set_to_none=True); loss=torch.nn.functional.binary_cross_entropy_with_logits(net(xx[idx]).ravel(),yy[idx]); loss.backward(); opt.step(); total+=loss.item()*len(idx)
   row=dict(run=tag,epoch=epoch+1,train_bce=total/len(x))
   if validation is not None:
    net.eval()
    with torch.no_grad(): row['validation_bce']=float(torch.nn.functional.binary_cross_entropy_with_logits(net(vx).ravel(),vy).item())
   history.append(row)
  artifact.update(kind='torch',state={k:v.detach().cpu() for k,v in net.state_dict().items()},scaler=scaler,device_trained=device)
 else: raise ValueError(name)
 if name!='mlp': estimator.fit(x,y); artifact['estimator']=estimator
 artifact['fit_rows']=len(x); artifact['fit_seconds']=time.perf_counter()-start
 return artifact,history

def evaluate(model,data,name,phase,fold=None):
 start=time.perf_counter(); p=score(model,data['x']); elapsed=time.perf_counter()-start
 return dict(model=name,phase=phase,fold=fold,inference_seconds=elapsed,fit_seconds=model['fit_seconds'],fit_rows=model['fit_rows'],**metrics(data,p)),p

def main():
 import torch
 if os.environ.get('PATHSNAP_REQUIRE_GPU')=='1' and not torch.cuda.is_available(): raise RuntimeError('GPU run requested but CUDA is unavailable')
 gpu=subprocess.run(['nvidia-smi'],capture_output=True,text=True) if torch.cuda.is_available() else None
 (OUT/'nvidia-smi.txt').write_text(gpu.stdout if gpu else 'No CUDA GPU')
 write('environment.json',dict(python=sys.version,platform=platform.platform(),torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,seed=SEED,started_utc=pd.Timestamp.now(tz='UTC').isoformat()))
 subprocess.run([sys.executable,'-m','pip','freeze'],stdout=(OUT/'environment_freeze.txt').open('w'),check=True)
 if (OUT/'search.sqlite3').exists():
  raise RuntimeError('Existing experiment evidence found. Reproduce in a fresh extracted copy with an empty results directory; preserve the delivered results.')
 train=load_data('train'); test=load_data('student_test')
 trainids=sorted(train['c'].trip_id.unique()); testids=sorted(test['c'].trip_id.unique())
 assert len(trainids)==80 and len(testids)==10 and not set(trainids)&set(testids)
 joblib.dump({'train':train,'student_test':test},OUT/'feature_cache.joblib',compress=0)
 # Split complete trips, balancing observation counts through the label timeline.
 folds=[]; assignments=[]
 for k,(i,j) in enumerate(GroupKFold(3).split(train['labels'],groups=train['labels'].trip_id),1):
  fitids=train['labels'].iloc[i].trip_id.unique(); valid=train['labels'].iloc[j].trip_id.unique()
  assert not set(fitids)&set(valid); folds.append((fitids,valid))
  assignments.extend(dict(trip_id=t,fold=k) for t in valid)
 pd.DataFrame(assignments).to_csv(OUT/'cv_folds.csv',index=False)
 write('protocol.json',dict(seed=SEED,features=FEATURES,folds=3,selection_objective='mean grouped-CV observation top-1 accuracy; candidate misses count as errors',binary_metrics='Aligned observations with reference inside candidates, threshold 0.5; binary candidate macro F1, not arc-class F1',knn_sample='Uniform without replacement, at most 160000 candidate rows from each fit partition',evaluator_data_read=False,configs=BASE))
 rows=[]; history=[]; model_cv={}
 for name,params in BASE.items():
  cv=[]
  for k,(fitids,valid) in enumerate(folds,1):
   print(f'START {name} fold {k}',flush=True)
   a,h=fit(name,params,take(train,fitids),f'{name}_fold{k}',take(train,valid) if name=='mlp' else None); history+=h
   row,p=evaluate(a,take(train,valid),name,'cv',k); rows.append(row); cv.append(row['top1_accuracy'])
   print(json.dumps(row),flush=True); pd.DataFrame(rows).to_csv(OUT/'model_metrics.csv',index=False)
  model_cv[name]=float(np.mean(cv))
  a,h=fit(name,params,train,f'{name}_full'); history+=h; joblib.dump(a,OUT/f'{name}.joblib',compress=3)
  for phase,data in [('train_fit',train),('student_test',test)]:
   row,p=evaluate(a,data,name,phase); rows.append(row)
   if phase=='student_test': data['c'][KEY].assign(score=p).to_parquet(OUT/f'{name}_scores.parquet',index=False)
   print(json.dumps(row),flush=True)
  pd.DataFrame(rows).to_csv(OUT/'model_metrics.csv',index=False); pd.DataFrame(history).to_csv(OUT/'learning_curves.csv',index=False)
 # Boosting is a promising probability model and efficiently supports GPU TPE.
 # Final selection compares its tuned CV objective with every baseline.
 study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=SEED,n_startup_trials=4),storage=f'sqlite:///{OUT}/search.sqlite3',study_name='pathsnap_xgboost',load_if_exists=True)
 study.enqueue_trial(BASE['xgboost'])
 def objective(trial):
  params=dict(n_estimators=trial.suggest_int('n_estimators',120,480,step=60),max_depth=trial.suggest_int('max_depth',3,8),learning_rate=trial.suggest_float('learning_rate',0.02,0.2,log=True),min_child_weight=trial.suggest_int('min_child_weight',4,64),subsample=trial.suggest_float('subsample',0.65,1.0),colsample_bytree=trial.suggest_float('colsample_bytree',0.65,1.0),reg_lambda=trial.suggest_float('reg_lambda',0.5,30,log=True))
  scores=[]; f1=[]
  for k,(fi,vi) in enumerate(folds,1):
   a,_=fit('xgboost',params,take(train,fi),f'trial{trial.number}_{k}'); row,_=evaluate(a,take(train,vi),'xgboost_tuned','search',k); scores.append(row['top1_accuracy']); f1.append(row['macro_f1'])
  trial.set_user_attr('fold_top1',scores); trial.set_user_attr('fold_macro_f1',f1)
  return float(np.mean(scores))
 study.optimize(objective,n_trials=12,callbacks=[lambda s,t:s.trials_dataframe().to_csv(OUT/'search_trials.csv',index=False)])
 best=study.best_trial; model_cv['xgboost_tuned']=best.value
 write('tuning.json',dict(best_trial=best.number,best_params=best.params,cv_top1=best.value,trials=len(study.trials),sampler='TPE',startup_trials=4,search_space={'n_estimators':'120..480 step 60','max_depth':'3..8','learning_rate':'log 0.02..0.2','min_child_weight':'4..64','subsample':'0.65..1','colsample_bytree':'0.65..1','reg_lambda':'log 0.5..30'},selected_on='training trips only'))
 a,h=fit('xgboost',best.params,train,'xgboost_tuned_full'); joblib.dump(a,OUT/'xgboost_tuned.joblib',compress=3)
 for k in range(3):
  rows.append(dict(model='xgboost_tuned',phase='cv',fold=k+1,top1_accuracy=best.user_attrs['fold_top1'][k],macro_f1=best.user_attrs['fold_macro_f1'][k]))
 for phase,data in [('train_fit',train),('student_test',test)]:
  row,p=evaluate(a,data,'xgboost_tuned',phase); rows.append(row)
  if phase=='student_test': data['c'][KEY].assign(score=p).to_parquet(OUT/'xgboost_tuned_scores.parquet',index=False)
 selected=max(model_cv,key=model_cv.get)
 joblib.dump(joblib.load(OUT/f'{selected}.joblib'),OUT/'final_model.joblib',compress=3)
 final_artifact=joblib.load(OUT/'final_model.joblib')
 if selected in ('xgboost','xgboost_tuned'):
  final_artifact['estimator'].save_model(OUT/'final_model.ubj')
  write('final_model.json',{k:final_artifact[k] for k in ['name','params','features','kind','seed']})
 write('selection.json',dict(selected_model=selected,cv_top1=model_cv,reason='Highest mean three-fold held-out-trip top-1; student-test scores not used for selection',optimism_note='Tuned CV is used both for search and selection, not an unbiased nested-CV estimate. Student-test remains the development comparison.'))
 pd.DataFrame(rows).to_csv(OUT/'model_metrics.csv',index=False)
 print('SELECTED',selected,model_cv,flush=True)

if __name__=='__main__':
 with threadpool_limits(limits=4): main()
