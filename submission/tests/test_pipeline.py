import unittest
from unittest.mock import patch
from io import BytesIO
from zipfile import ZipFile
import numpy as np
from pathsnap.pipeline import clean_track,live_candidates,features,trip_members
from pathsnap.decoder import Candidate,Observation,decode
from pathsnap.routing import Position,RoadGraph

class LivePipelineTests(unittest.TestCase):
 def setUp(self):
  self.nodes=np.array([[0.,0.],[.001,0.]])
  self.arcs=np.array([[0,1],[1,0]])
  self.track=clean_track(np.array([[.0001+i*.0001,0.,i] for i in range(7)]))
 def test_direction_distinguishes_reverse_arcs(self):
  c,x=live_candidates(self.track,self.nodes,self.arcs,'student_test','00000001')
  self.assertTrue((x[c.arc_id.eq(0),4]>.99).all())
  self.assertTrue((x[c.arc_id.eq(1),4]<-.99).all())
 def test_cleaning_preserves_original_ids_and_marks_breaks(self):
  t=clean_track([[0,0,1],[float('nan'),0,2],[.001,0,3],[.002,0,3],[.003,0,10]])
  self.assertEqual(t.obs_id.tolist(),[0,2,4]);self.assertTrue(t.row_break_before.iloc[1]);self.assertTrue(t.gap_before.iloc[-1])
 def test_missing_candidates_return_valid_empty_features(self):
  t=clean_track([[1.,1.,0],[1.001,1.,1]])
  c,x=live_candidates(t,self.nodes,self.arcs,'student_test','00000001')
  self.assertEqual(x.shape,(0,13))
 def test_reference_files_not_visible_to_prediction_inventory(self):
  bio=BytesIO()
  with ZipFile(bio,'w') as z:
   for ext in ['track','nodes','arcs','route']:z.writestr('student_test/001/001.'+ext,'0 0')
  with ZipFile(bio) as z:
   m=trip_members(z,'student_test')
  self.assertEqual(set(m['001']),{'track','nodes','arcs'})
 def test_explicit_original_row_break_splits_hmm(self):
  g=RoadGraph(self.nodes,self.arcs)
  c=Candidate(Position(0,.2),.9,1)
  r=decode(g,[Observation(0,0,(0.,0.),(c,)),Observation(2,1,(0.,0.),(c,),True)])
  self.assertEqual(len(r.segments),2)
 def test_small_offset_retreat_is_noise_not_reverse_road(self):
  g=RoadGraph(self.nodes,self.arcs[:1])
  a=Candidate(Position(0,.50),.9,1);b=Candidate(Position(0,.49),.9,1)
  obs=[Observation(0,0,(.0005,0.),(a,)),Observation(1,1,(.00049,0.),(b,))]
  self.assertEqual(len(decode(g,obs,jitter_m=0).segments),2)
  self.assertEqual(len(decode(g,obs,jitter_m=15).segments),1)
  self.assertIsNone(g.route(a.position,b.position))
 def test_feature_values_do_not_depend_on_reference_columns(self):
  c,x=live_candidates(self.track,self.nodes,self.arcs,'student_test','00000001')
  c['reference_arc_id']=999
  np.testing.assert_array_equal(x,features(self.track,c,self.nodes,self.arcs))
if __name__=='__main__':unittest.main()
