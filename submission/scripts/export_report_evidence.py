"""Static exports from actual Folium-generated layers and recorded execution logs.

This does not fabricate browser or native-terminal screenshots. The existing
browser screenshot is retained and explicitly labelled as an integration test.
"""
import json,re,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results';D=OUT/'screenshots';D.mkdir(exist_ok=True)

def extract_calls(html,method):
 decoder=json.JSONDecoder()
 for match in re.finditer(r'\bL\.'+method+r'\s*\(\s*',html):
  text=html[match.end():];value,end=decoder.raw_decode(text)
  tail=text[end:].lstrip()
  options={}
  if tail.startswith(','):
   try:options,_=decoder.raw_decode(tail[1:].lstrip())
   except json.JSONDecodeError:pass
  yield value,options

def main():
 source=OUT/'maps/development_00000009.html';html=source.read_text()
 fig,ax=plt.subplots(figsize=(11,7),layout='constrained');ax.set_facecolor('#eff5f4')
 polylines=list(extract_calls(html,'polyline'));points=np.array([coords for coords,_ in extract_calls(html,'circleMarker')])
 colors={'#64748b':('Road network',.7,.30),'#be185d':('HMM prediction',2.0,.95),'#22c55e':('Supplied reference',4.5,.55)}
 # Keep reference under the prediction; both are the actual Folium coordinates.
 polylines.sort(key=lambda item: {'#64748b':0,'#22c55e':1,'#be185d':2}.get(item[1].get('color'),0))
 legend=[]
 for coords,opts in polylines:
  color=opts.get('color')
  if color not in colors:continue
  label,width,alpha=colors[color];arr=np.asarray(coords)
  if arr.ndim==2:arr=arr[None,:,:]
  ax.add_collection(LineCollection(arr[:,:,::-1],colors=color,linewidths=width,alpha=alpha))
  legend.append(Line2D([0],[0],color=color,lw=max(width,2),label=label))
 if len(points):
  ax.scatter(points[:,1],points[:,0],s=5,color='#2563eb',alpha=.7,zorder=5,label='Original GPS')
  lo,hi=points.min(axis=0),points.max(axis=0);margin=np.maximum((hi-lo)*.08,.001)
  ax.set_xlim(lo[1]-margin[1],hi[1]+margin[1]);ax.set_ylim(lo[0]-margin[0],hi[0]+margin[0]);ax.set_aspect(1/np.cos(np.deg2rad(points[:,0].mean())))
 legend.append(Line2D([0],[0],marker='o',color='none',markerfacecolor='#2563eb',markersize=5,label='Original GPS'))
 ax.legend(handles=legend,loc='upper left',framealpha=.95)
 ax.set(xlabel='Longitude (degrees)',ylabel='Latitude (degrees)',title='Trip 00000009 | actual Folium layer export')
 ax.grid(alpha=.2);fig.suptitle('Student-test development map: GPS, roads, reference and HMM prediction',fontsize=11)
 fig.savefig(D/'development_map.jpg',dpi=170);plt.close(fig)
 shutil.copy2(D/'integration_ui.jpg',D/'working_ui.jpg')
 logs=(OUT/'api_server.log').read_text().splitlines()
 success=[line for line in logs if 'SUCCESS trip=' in line]
 response=json.loads((OUT/'api_response_00000052.json').read_text())
 text='ACTUAL HTTP RESPONSE AND RECEIVING-SERVER LOG\nRendered evidence view; not a native terminal screenshot.\n\n'
 text+='POST /predict -> HTTP 200\n'+json.dumps({k:response[k] for k in ['trip_id','model','observations','matched_observations','runtime_seconds','evaluation']},indent=2)
 text+='\n\nRECORDED UVICORN OUTPUT\n'+'\n'.join(success[-4:])
 import textwrap
 lines=[]
 for line in text.splitlines():lines.extend(textwrap.wrap(line,width=118,replace_whitespace=False,drop_whitespace=False) or [''])
 fig=plt.figure(figsize=(12,8));fig.patch.set_facecolor('#f2f7f8');fig.text(.035,.97,'\n'.join(lines),va='top',fontfamily='monospace',fontsize=9,color='#123743',linespacing=1.45)
 fig.savefig(D/'server_evidence.jpg',dpi=160,bbox_inches='tight');plt.close(fig)
 (D/'EVIDENCE_PROVENANCE.json').write_text(json.dumps({'working_ui.jpg':{'type':'Actual previously captured browser screenshot','model':'Training-only integration Logistic Regression, not final XGBoost','trip':'00000052'},'development_map.jpg':{'type':'Static geographical export of the actual Folium PolyLine and CircleMarker coordinates','source':str(source.relative_to(ROOT)),'model':'Frozen final XGBoost + HMM','not_a_browser_screenshot':True},'server_evidence.jpg':{'type':'Rendered actual HTTP response and Uvicorn log text','source':['results/api_response_00000052.json','results/api_server.log'],'not_a_native_terminal_screenshot':True},'capture_limit':'Browser-control access became unavailable; native Terminal application access was rejected by the computer-use safety tool.'},indent=2))
 print('Exported genuine Folium layers and log evidence; retained actual integration UI screenshot.')
if __name__=='__main__':main()
