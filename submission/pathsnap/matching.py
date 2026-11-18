"""Classifier scores and directed HMM decoding share the exact candidate keys."""
import numpy as np
from .decoder import Candidate,Observation,decode
from .routing import Position,RoadGraph

def make_options(c,p):
 options={}
 for row,probability in zip(c.itertuples(index=False),p,strict=True):
  options.setdefault(row.obs_id,[]).append(Candidate(Position(int(row.arc_id),float(row.fraction_along_arc)),float(probability),int(row.candidate_rank)))
 return options

def reconstruct(track,c,p,nodes,arcs,params=None,graph=None):
 options=make_options(c,p)
 obs=[Observation(int(t.obs_id),float(t.timestamp_s),(float(t.longitude),float(t.latitude)),tuple(options.get(t.obs_id,())),bool(getattr(t,'row_break_before',False))) for t in track.itertuples(index=False)]
 graph=graph or RoadGraph(nodes,arcs,shortest_cache_size=50000)
 result=decode(graph,obs,**(params or {}))
 return result,options,graph

def continuity(track,arcs,pred):
 ids=np.array([pred.get(int(i),-1) if pred.get(int(i)) is not None else -1 for i in track.obs_id],dtype=int)
 eligible=(ids[:-1]>=0)&(ids[1:]>=0)&(np.diff(track.timestamp_s)<=5)
 if 'row_break_before' in track: eligible &= ~track.row_break_before.to_numpy(dtype=bool)[1:]
 prev,curr=ids[:-1][eligible],ids[1:][eligible]
 count=int(((prev==curr)|(arcs[prev,1]==arcs[curr,0])).sum())
 return count,len(prev)
