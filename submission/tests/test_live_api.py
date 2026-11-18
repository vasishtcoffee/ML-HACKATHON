"""Integration checks use a real fitted artifact and raw inputs, never score replay."""
import unittest,tempfile
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from unittest.mock import patch
from fastapi.testclient import TestClient
from pathsnap import api

class LiveAPIIsolationTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.artifact=api.ROOT/'results/final_model.ubj'
  if not cls.artifact.exists():raise unittest.SkipTest('Final fitted artifact not available yet')
 def test_missing_references_still_predicts_raw_inputs(self):
  with tempfile.TemporaryDirectory() as d:
   archive=Path(d)/'raw_inputs_only.zip'
   with ZipFile(api.ROOT/'data/raw/Student folder.zip') as src,ZipFile(archive,'w',ZIP_DEFLATED) as dst:
    for name in src.namelist():
     if '/student_test/00000052/' in name and Path(name).suffix in ('.track','.nodes','.arcs'):dst.writestr(name,src.read(name))
   with patch.object(api,'ARCHIVE',archive),patch.object(api,'REFERENCE',archive),patch.object(api,'MODEL',self.artifact):
    api.model.cache_clear();client=TestClient(api.app)
    r=client.post('/predict',json={'trip_id':'00000052'})
    self.assertEqual(r.status_code,200,r.text[:300]);data=r.json()
    self.assertIsNone(data['evaluation']);self.assertEqual(data['observations'],262)
    self.assertGreater(data['matched_observations'],0)
    self.assertNotIn('reference_arc_id',r.text)
    self.assertNotIn('Reference route',data['map_html'])
   api.model.cache_clear()
 def test_malformed_and_unknown_ids_rejected(self):
  client=TestClient(api.app)
  self.assertEqual(client.post('/predict',json={'trip_id':'../escape'}).status_code,422)
  self.assertEqual(client.post('/predict',json={'trip_id':'99999999'}).status_code,404)
if __name__=='__main__':unittest.main()
