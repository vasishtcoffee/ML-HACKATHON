"""Compare independent candidate ranking with graph-aware Viterbi decoding."""
import argparse
from pathlib import Path
from zipfile import ZipFile
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from train_baselines import ROOT, OBS_KEY, load_split, fit_model, top_candidates
from score_baseline import KEY, FEATURES, score_candidates
from generate_candidates import read_graph
from route_reconstruction.graph import arc_lengths,transition_matrix
from route_reconstruction.viterbi import decode


def decode_trip(timeline,candidates,scores,graph,beta_m=30.0):
    nodes,arcs=graph; lengths=arc_lengths(nodes,arcs)
    candidates=candidates.assign(score=scores)
    by_obs={int(k):g.sort_values(['candidate_rank','arc_id']).to_dict('records')
            for k,g in candidates.groupby('obs_id',sort=False)}
    results=[]; independent=[]
    rows=timeline.sort_values(['timestamp_s','obs_id']).to_dict('records')
    segments=[]; current=[]
    for row in rows:
        obs=int(row['obs_id']); options=by_obs.get(obs,[])
        if row['gap_before'] or row['row_break_before'] or not options:
            if current: segments.append(current); current=[]
            if options: current=[(row,options)]
            else: results.append((obs,None)); independent.append((obs,None))
        else: current.append((row,options))
    if current: segments.append(current)
    for segment in segments:
        emissions=[]
        for _,opts in segment:
            probs=np.clip(np.array([x['score'] for x in opts],float),1e-6,1-1e-6)
            probs=probs/probs.sum()
            emissions.append(np.log(probs))
        transitions=[]
        for (prevrow,prev),(row,opts) in zip(segment,segment[1:]):
            from math import hypot
            # Haversine displacement between observed GPS points.
            lat1,lon1=np.deg2rad([prevrow['latitude'],prevrow['longitude']]);lat2,lon2=np.deg2rad([row['latitude'],row['longitude']])
            dlat=lat2-lat1;dlon=lon2-lon1
            h=np.sin(dlat/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
            displacement=2*6371008.8*np.arcsin(np.sqrt(np.clip(h,0,1)))
            prevstates=[dict(arc_id=x['arc_id'],offset_from_arc_start_m=x['offset_from_arc_start_m'],to_node_id=arcs[int(x['arc_id']),1]) for x in prev]
            nextstates=[dict(arc_id=x['arc_id'],offset_from_arc_start_m=x['offset_from_arc_start_m'],from_node_id=arcs[int(x['arc_id']),0]) for x in opts]
            transitions.append(transition_matrix(prevstates,nextstates,displacement,arcs,lengths,beta_m))
        # Split at impossible network steps. A decoder restart preserves the best
        # score within each connected stretch and never fabricates a graph edge.
        bounds=[0]; reachable=np.ones(len(emissions[0]),dtype=bool)
        for i,matrix in enumerate(transitions):
            next_reachable=np.isfinite(matrix[reachable]).any(axis=0)
            if not next_reachable.any():
                bounds.append(i+1)
                reachable=np.ones(len(emissions[i+1]),dtype=bool)
            else:
                reachable=next_reachable
        bounds.append(len(segment))
        for left,right in zip(bounds,bounds[1:]):
            part=decode(emissions[left:right],transitions[left:right-1])
            for (row,opts),idx in zip(segment[left:right],part):
                results.append((int(row['obs_id']),int(opts[idx]['arc_id'])))
        for row,opts in segment:
            independent.append((int(row['obs_id']),int(max(opts,key=lambda x:(x['score'],-x['candidate_rank'],-x['arc_id']))['arc_id'])))
    eligible=timeline.loc[timeline['alignment_status'].eq('aligned')].set_index('obs_id')
    truth=eligible['reference_arc_id']
    aligned=truth.dropna().index
    dec=pd.Series(dict(results)); ind=pd.Series(dict(independent))
    return {
        'aligned_observations':len(aligned),
        'classifier_top1':float((ind.reindex(aligned).to_numpy()==truth.loc[aligned].astype(int).to_numpy()).mean()),
        'decoded_top1':float((dec.reindex(aligned).to_numpy()==truth.loc[aligned].astype(int).to_numpy()).mean()),
        'candidate_misses':int((~eligible.loc[aligned,'reference_in_candidates']).sum()),
        'decoded_continuity':continuity(dec,timeline,arcs),
        'classifier_continuity':continuity(ind,timeline,arcs),
    }

def continuity(pred,timeline,arcs):
    t=timeline.sort_values(['timestamp_s','obs_id']).set_index('obs_id')
    ids=t.index.to_numpy();values=pred.reindex(ids).to_numpy();ok=(values[:-1]>=0)&(values[1:]>=0)&~t.gap_before.to_numpy()[1:]&~t.row_break_before.to_numpy()[1:]
    a,b=values[:-1][ok].astype(int),values[1:][ok].astype(int)
    return float(np.mean((a==b)|(arcs[a,1]==arcs[b,0]))) if len(a) else float('nan')

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--models',nargs='+',default=['random_forest','xgboost']);ap.add_argument('--beta-m',type=float,default=30.);ap.add_argument('--output',type=Path,default=ROOT/'experiments/decoder');ap.add_argument('--phase',choices=['student_test','cv'],default='student_test');args=ap.parse_args()
    processed=ROOT/'data/processed'; archive_path=ROOT/'data/raw/Student folder.zip'; output=args.output;output.mkdir(parents=True,exist_ok=True)
    train_timeline,train=load_split(processed,'train'); test_timeline,test=load_split(processed,'student_test')
    coords=pd.read_csv(processed/'observations.csv.gz',dtype={'trip_id':str},usecols=OBS_KEY+['longitude','latitude'])
    train_timeline=train_timeline.merge(coords,on=OBS_KEY,validate='one_to_one')
    test_timeline=test_timeline.merge(coords,on=OBS_KEY,validate='one_to_one')
    inventory=pd.read_csv(processed/'archive_inventory.csv',dtype={'trip_id':str}); graph_members=inventory.loc[inventory.kind.isin(['nodes','arcs'])].pivot(index=['split','trip_id'],columns='kind',values='member')
    rows=[]
    if args.phase == 'student_test':
      for name in args.models:
        score_file=ROOT/'experiments'/name/f'{name}_student_scores.parquet'
        if not score_file.is_file(): raise FileNotFoundError(score_file)
        score_table=pd.read_parquet(score_file)
        if 'score' not in score_table: raise ValueError(f"No candidate scores in {score_file}")
        keyed=test[KEY].merge(score_table[KEY+['score']],on=KEY,validate='one_to_one')
        score=keyed.score.to_numpy()
        with ZipFile(archive_path) as z:
          for trip_id,t in test_timeline.groupby('trip_id',sort=True):
            mask=test.trip_id.eq(trip_id).to_numpy();c=test.loc[mask];s=score[mask]
            graph=read_graph(z,graph_members.loc[('student_test',trip_id)])
            rows.append({'model':name,'phase':'student_test','fold':None,'trip_id':trip_id,**decode_trip(t,c,s,graph,args.beta_m)})
            print(name,'trip',trip_id,'decoded',flush=True)
    pd.DataFrame(rows).to_csv(output/'trip_metrics.csv',index=False)
    pd.DataFrame(rows).to_csv(output/'trip_metrics.csv',index=False)
    detail=pd.DataFrame(rows)
    grouped=[]
    for keys,frame in detail.groupby(['model','phase','fold'],dropna=False):
        weight=frame['aligned_observations'].to_numpy(float)
        item=dict(zip(['model','phase','fold'],keys))
        for metric in ['classifier_top1','decoded_top1']:
            item[metric]=float(np.average(frame[metric],weights=weight))
        item['aligned_observations']=int(weight.sum())
        item['candidate_misses']=int(frame.candidate_misses.sum())
        item['classifier_continuity']=float(frame.classifier_continuity.mean())
        item['decoded_continuity']=float(frame.decoded_continuity.mean())
        grouped.append(item)
    result=pd.DataFrame(grouped)
    result.to_csv(output/'summary.csv',index=False);print(result.to_string(index=False))
if __name__=='__main__':main()
