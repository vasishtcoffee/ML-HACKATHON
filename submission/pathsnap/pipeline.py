"""Geometry-only live inference. Reference route files are never read here."""
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pandas as pd
from scripts.generate_candidates import project_candidates

FEATURES = ['distance_m','log_rank','fraction','log_length','heading_cos','heading_abs_sin',
            'heading_known','log_speed','log_candidates','distance_minus_best','relative_distance',
            'near_start','near_end']
KEY = ['split','trip_id','obs_id','arc_id']


def bearing(a,b):
    a,b=np.deg2rad(a),np.deg2rad(b)
    dl=b[:,0]-a[:,0]
    return np.arctan2(np.sin(dl)*np.cos(b[:,1]), np.cos(a[:,1])*np.sin(b[:,1])-np.sin(a[:,1])*np.cos(b[:,1])*np.cos(dl))


def clean_track(raw):
    raw=np.asarray(raw,dtype=float)
    if raw.ndim!=2 or raw.shape[1]!=3: raise ValueError('Track requires longitude, latitude, timestamp')
    t=pd.DataFrame(raw,columns=['longitude','latitude','timestamp_s'])
    t['obs_id']=np.arange(len(t))
    valid=np.isfinite(raw).all(axis=1)&t.longitude.between(-180,180)&t.latitude.between(-90,90)
    t=t.loc[valid].sort_values('timestamp_s',kind='stable').drop_duplicates('timestamp_s').copy()
    if t.empty: raise ValueError('No valid observations')
    t['gap_before']=t.timestamp_s.diff().gt(5)
    t['row_break_before']=t.obs_id.diff().ne(1)&t.timestamp_s.diff().notna()
    return t.reset_index(drop=True)


def trip_members(archive,split):
    """Only tracks/nodes/arcs are visible to prediction; no .route lookup."""
    result={}
    for name in archive.namelist():
        parts=Path(name).parts
        if len(parts)<3 or parts[-3]!=split: continue
        p=Path(parts[-1])
        if p.suffix not in ('.track','.nodes','.arcs') or p.stem!=parts[-2]: continue
        if p.suffix[1:] in result.setdefault(p.stem,{}): raise ValueError('Duplicate input file')
        result[p.stem][p.suffix[1:]]=name
    return {k:v for k,v in result.items() if set(v)=={'track','nodes','arcs'}}


def read_trip(path,split,trip_id):
    with ZipFile(path) as z:
        m=trip_members(z,split).get(trip_id)
        if m is None: raise ValueError('Unknown trip')
        values={k:np.loadtxt(z.open(v),ndmin=2,dtype=np.int64 if k=='arcs' else float) for k,v in m.items()}
    nodes,arcs=values['nodes'],values['arcs']
    if nodes.shape[1]!=2 or not np.isfinite(nodes).all() or (np.abs(nodes)>[180,90]).any(): raise ValueError('Invalid nodes')
    if arcs.shape[1]!=2 or (arcs<0).any() or (arcs>=len(nodes)).any(): raise ValueError('Invalid arcs')
    return clean_track(values['track']),nodes,arcs


def features(track,candidates,nodes,arcs):
    """Offline trip direction over a centred 6-sample window, bounded by gaps."""
    t=track.sort_values(['timestamp_s','obs_id']).copy()
    segments=(t.timestamp_s.diff().gt(5)|t.get('row_break_before',pd.Series(False,index=t.index))).cumsum()
    start=t.groupby(segments)[['longitude','latitude','timestamp_s']].shift(3)
    end=t.groupby(segments)[['longitude','latitude','timestamp_s']].shift(-3)
    # Near segment boundaries use one-sided multi-sample estimates.
    start=start.fillna(t[['longitude','latitude','timestamp_s']])
    end=end.fillna(t[['longitude','latitude','timestamp_s']])
    a=start[['longitude','latitude']].to_numpy(); b=end[['longitude','latitude']].to_numpy()
    xy=np.deg2rad(b-a)*6371008.8; xy[:,0]*=np.cos(np.deg2rad((a[:,1]+b[:,1])/2))
    displacement=np.linalg.norm(xy,axis=1)
    dt=(end.timestamp_s-start.timestamp_s).to_numpy()
    known=(displacement>=3)&(dt>0)&(dt<=10)
    direction=bearing(a,b)
    motion=pd.DataFrame({'obs_id':t.obs_id,'heading':direction,'known':known,
                         'speed':np.divide(displacement,dt,out=np.zeros_like(dt),where=dt>0)})
    c=candidates.merge(motion,on='obs_id',validate='many_to_one')
    ids=c.arc_id.to_numpy(dtype=int); p=nodes[arcs[ids,0]]; q=nodes[arcs[ids,1]]
    arc_direction=bearing(p,q)
    dx=np.deg2rad(q-p)*6371008.8; dx[:,0]*=np.cos(np.deg2rad((p[:,1]+q[:,1])/2))
    length=np.linalg.norm(dx,axis=1)
    delta=arc_direction-c.heading.to_numpy()
    grouped=c.groupby('obs_id').distance_to_arc_m
    best=grouped.transform('min').to_numpy(); dist=c.distance_to_arc_m.to_numpy()
    x=np.column_stack([dist,np.log1p(c.candidate_rank),c.fraction_along_arc,np.log1p(length),
        np.where(c.known,np.cos(delta),0),np.where(c.known,np.abs(np.sin(delta)),0),c.known,
        np.log1p(np.clip(c.speed,0,100)),np.log1p(grouped.transform('size')),
        dist-best,dist/(best+3),c.fraction_along_arc<0.05,c.fraction_along_arc>0.95]).astype('float32')
    if not np.isfinite(x).all(): raise ValueError('Non-finite features')
    return x


def live_candidates(track,nodes,arcs,split,trip_id,radius=50):
    c=project_candidates(track,nodes,arcs,radius)
    c['split']=split; c['trip_id']=trip_id
    return c,features(track,c,nodes,arcs)


def score(artifact,x):
    if artifact.get('features') != FEATURES:
        raise ValueError('Model feature schema does not match live inference')
    if x.ndim != 2 or x.shape[1] != len(FEATURES) or not np.isfinite(x).all():
        raise ValueError('Invalid inference feature matrix')
    if len(x)==0: return np.empty(0)
    if artifact['kind']=='torch':
        import torch
        net=torch.nn.Sequential(torch.nn.Linear(len(FEATURES),64),torch.nn.ReLU(),
            torch.nn.Linear(64,32),torch.nn.ReLU(),torch.nn.Linear(32,1))
        net.load_state_dict(artifact['state']); net.eval()
        xx=artifact['scaler'].transform(x).astype('float32')
        with torch.no_grad(): p=torch.sigmoid(net(torch.from_numpy(xx))).numpy().ravel()
    else:
        p=artifact['estimator'].predict_proba(x)[:,1]
        if artifact['name']=='knn': p=(artifact['params']['n_neighbors']*p+1)/(artifact['params']['n_neighbors']+2)
    if p.shape!=(len(x),) or not np.isfinite(p).all() or ((p<0)|(p>1)).any(): raise ValueError('Invalid model scores')
    return p
