"""Laptop 2 LAN check against the actual Laptop 1 API."""
import argparse,json,urllib.request
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--api-url',required=True);p.add_argument('--trip-id');a=p.parse_args()
base=a.api_url.rstrip('/')
with urllib.request.urlopen(base+'/trips',timeout=30) as r:print('Available trips:',r.read().decode())
if a.trip_id:
 req=urllib.request.Request(base+'/predict',data=json.dumps({'trip_id':a.trip_id}).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=600) as r:
  d=json.load(r);print('HTTP',r.status);print(json.dumps({k:v for k,v in d.items() if k not in ('map_html','route_segments','arcs_by_observation')},indent=2))
