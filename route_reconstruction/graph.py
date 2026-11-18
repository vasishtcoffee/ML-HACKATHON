"""Local directed transitions between candidate road arcs."""
import numpy as np
EARTH_RADIUS_M=6_371_008.8

def arc_lengths(nodes,arcs):
    coords=np.deg2rad(np.asarray(nodes,float));edges=np.asarray(arcs,int)
    lat1,lon1=coords[edges[:,0],1],coords[edges[:,0],0]
    lat2,lon2=coords[edges[:,1],1],coords[edges[:,1],0]
    hav=np.sin((lat2-lat1)/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
    return 2*EARTH_RADIUS_M*np.arcsin(np.sqrt(np.clip(hav,0,1)))

def transition_matrix(previous,current,displacement_m,arcs,lengths,beta_m=30.0):
    """Score same-arc or one-directed-edge transitions by distance agreement.

    The decoder deliberately uses local transitions between same or adjacent
    arcs. Longer network paths are split and restarted; no unobserved route is
    invented across a disconnected step.
    """
    a=np.asarray([x['arc_id'] for x in previous],int);b=np.asarray([x['arc_id'] for x in current],int)
    ao=np.asarray([x['offset_from_arc_start_m'] for x in previous],float)
    bo=np.asarray([x['offset_from_arc_start_m'] for x in current],float)
    end=np.asarray([x['to_node_id'] for x in previous],int)
    start=np.asarray([x['from_node_id'] for x in current],int)
    same=(a[:,None]==b[None,:])&(bo[None,:]>=ao[:,None])
    adjacent=end[:,None]==start[None,:]
    distance=np.where(same,bo[None,:]-ao[:,None],
        np.where(adjacent,lengths[a,None]-ao[:,None]+bo[None,:],np.inf))
    return np.where(np.isfinite(distance),-np.abs(float(displacement_m)-distance)/beta_m,-np.inf)
