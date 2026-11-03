"""Checks for ownership ambiguity, working likelihood behaviour and conservative calibration."""
import sys
from pathlib import Path
import unittest
import tempfile
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'analysis'))
from ownership_audit import owner,polygon_is_simple
from filament_likelihood import FilamentLikelihood,patch_at,threshold_at_fpr


class ScientificAuditTests(unittest.TestCase):
    def test_ownership_does_not_force_overlapping_or_external_boxes(self):
        a=np.zeros((100,100),np.uint8);a[10:80,10:80]=1;b=np.zeros_like(a);b[40:90,40:90]=1
        cells=[{'cell_id':'a','mask':a,'geometry_valid':True},{'cell_id':'b','mask':b,'geometry_valid':True}]
        self.assertEqual(owner([15,15,25,25],cells)['cell_id'],'a')
        self.assertEqual(owner([45,45,55,55],cells)['owner_status'],'ambiguous')
        self.assertEqual(owner([0,0,5,5],cells)['owner_status'],'unowned')
        cells[0]['geometry_valid']=False
        self.assertEqual(owner([15,15,25,25],cells)['owner_status'],'ambiguous')

    def test_self_crossing_cell_contours_are_flagged(self):
        self.assertTrue(polygon_is_simple([[0,0],[10,0],[10,10],[0,10]]))
        self.assertFalse(polygon_is_simple([[0,0],[10,10],[0,10],[10,0]]))

    def test_patch_extraction_uses_native_pixels_including_image_edges(self):
        image=np.arange(40*40,dtype=float).reshape(40,40)
        self.assertEqual(patch_at(image,20,20).reshape(31,31)[15,15],image[20,20])
        self.assertEqual(patch_at(image,0,0).shape,(961,))

    def test_mle_working_score_distinguishes_synthetic_filaments_and_spots(self):
        rng=np.random.default_rng(1);y,x=np.mgrid[-15:16,-15:16]
        line=30+20*np.exp(-.5*((x/10)**2+(y/2)**2));spot=30+20*np.exp(-.5*(x*x+y*y)/9)
        pos=np.stack([(line+rng.normal(0,1,line.shape)).ravel() for _ in range(20)])
        neg=np.stack([(spot+rng.normal(0,1,spot.shape)).ravel() for _ in range(20)])
        model=FilamentLikelihood();fit=model.fit(pos,neg)
        self.assertEqual(fit['line_width_pixels'],2);self.assertEqual(fit['line_length_scale_pixels'],10)
        self.assertGreater(float(model.score(pos).mean()),float(model.score(neg).mean()))
        self.assertTrue(np.isfinite(model.score(np.zeros((1,961)))).all())

    def test_empirical_fpr_threshold_handles_ties_conservatively(self):
        scores=np.array([0,0,1,1,2,3,4,5,6,7])
        threshold=threshold_at_fpr(scores,.1)
        self.assertLessEqual(np.mean(scores>=threshold),.1)
        self.assertEqual(np.mean(np.zeros(20)>=threshold_at_fpr(np.zeros(20),.05)),0)

    def test_no_verified_missing_labels_leaves_the_dataset_untouched(self):
        import pandas as pd
        from prepare_label_correction import prepare
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'base').mkdir();reviews=root/'reviews.csv';manifest=root/'manifest.csv'
            pd.DataFrame([{'decision':'ambiguous','source_id':'a'}]).to_csv(reviews,index=False)
            pd.DataFrame([{'source_id':'a','subset':'fit'}]).to_csv(manifest,index=False)
            result=prepare(root/'base',reviews,manifest,root/'corrected',root/'gate.json')
            self.assertEqual(result['status'],'not_activated');self.assertFalse((root/'corrected').exists())

    def test_native_training_accepts_source_image_lists_without_resizing(self):
        import yaml
        from PIL import Image
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
        from training.train import native_size
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);Image.new('RGB',(2048,2048)).save(root/'a.png')
            (root/'sources.txt').write_text(str(root/'a.png')+'\n')
            (root/'data.yaml').write_text(yaml.safe_dump({'path':str(root),'train':'sources.txt','val':'sources.txt'}))
            self.assertEqual(native_size(root/'data.yaml'),2048)

    def test_verified_correction_appends_without_removing_original_positives(self):
        import pandas as pd
        from prepare_label_correction import prepare
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);base=root/'base';(base/'images/train').mkdir(parents=True);(base/'labels/train').mkdir(parents=True)
            original='0 0.10 0.10 0.02 0.02\n1 0.8 0.8 0.1 0.1\n'
            (base/'labels/train/a.txt').write_text(original);(base/'labels/train/b.txt').write_text(original)
            reviews=root/'reviews.csv';manifest=root/'manifest.csv'
            pd.DataFrame([{'decision':'verified_cytosolic','source_id':'a','reviewer':'Confirming reviewer','reason':'Distinct cytosolic object confirmed','x1':900,'y1':900,'x2':950,'y2':950}]).to_csv(reviews,index=False)
            pd.DataFrame([{'source_id':'a','subset':'fit'},{'source_id':'b','subset':'calibration'}]).to_csv(manifest,index=False)
            result=prepare(base,reviews,manifest,root/'corrected',root/'gate.json')
            self.assertEqual(result['verified_missing_positives'],1)
            self.assertTrue((root/'corrected/labels/train/a.txt').read_text().startswith(original))
            self.assertEqual((base/'labels/train/a.txt').read_text(),original)
            self.assertEqual((root/'corrected/labels/train/b.txt').read_text(),original)

    def test_correction_cannot_modify_a_calibration_source(self):
        import pandas as pd
        from prepare_label_correction import prepare
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);reviews=root/'reviews.csv';manifest=root/'manifest.csv'
            pd.DataFrame([{'decision':'verified_cytosolic','source_id':'b'}]).to_csv(reviews,index=False)
            pd.DataFrame([{'source_id':'b','subset':'calibration'}]).to_csv(manifest,index=False)
            with self.assertRaisesRegex(ValueError,'never calibration'):
                prepare(root/'base',reviews,manifest,root/'corrected',root/'gate.json')
            self.assertFalse((root/'corrected').exists())


if __name__=='__main__':unittest.main()
