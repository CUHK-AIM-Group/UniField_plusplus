import json, os, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import nibabel as nib
import torch
from unifield.registration import REGIONS, SCHEMA, cache_folder
from unifield.dataset import RegisteredMRIDataset

class GeometryTests(unittest.TestCase):
    def test_shared_flip_sampling_and_padding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'low.nii.gz'; high=root/'high.nii.gz'
            volume=np.zeros((16,20,8),dtype=np.float32); volume[3:10,4:13,2:6]=1
            affine=np.diag([-2.,1.5,3.,1.]); affine[:3,3]=[30,-12,6]
            for p in (source,high): nib.save(nib.Nifti1Image(volume,affine),p)
            folder=cache_folder(source,root/'cache'); folder.mkdir(parents=True)
            for region in REGIONS: nib.save(nib.Nifti1Image(volume,affine),folder/f'mask_{region}.nii.gz')
            (folder/'registration.json').write_text(json.dumps({'schema':SCHEMA,'regions':REGIONS}))
            pairs=root/'pairs.csv'; pairs.write_text(f'{source},{high}\n')
            with patch.dict(os.environ,UNIFIELD_ATLAS_CACHE=str(root/'cache')):
                ds=RegisteredMRIDataset(pairs,spatial_size=(16,20,8),num_frames=12,full_volume=True)
                with patch('numpy.random.choice',return_value=0): baseline=ds[0]
                for flip in range(4):
                    with patch('numpy.random.choice',return_value=flip): item=ds[0]
                    self.assertTrue(torch.equal(item['lq'],item['hq']))
                    expected=baseline['region_masks']
                    if flip: expected=torch.flip(expected,[{1:2,2:3,3:1}[flip]])
                    self.assertTrue(torch.equal(item['region_masks'],expected))
                    mask=item['region_masks'][0]>.5
                    image=item['lq'][0]>0
                    delta=torch.argwhere(mask).float().mean(0)-torch.argwhere(image).float().mean(0)
                    self.assertLess(float(delta.abs().max()),1.0)
                ds.full_volume=False
                item=ds[0]
                self.assertTrue(torch.equal(item['region_masks'][:,7],item['region_masks'][:,-1]))
                self.assertEqual(item['lq'].shape,(3,12,16,20))
                nib.save(nib.Nifti1Image(volume,np.eye(4)),folder/'mask_frontal_lobe.nii.gz')
                with self.assertRaisesRegex(ValueError,'geometry'): ds[0]

    def test_unbound_atlas_fails(self):
        from diffsynth.agents.brain_atlas import BrainAtlas
        with self.assertRaisesRegex(RuntimeError,'Registered'): BrainAtlas().generate_region_mask('frontal_lobe',(8,16,20))

if __name__=='__main__': unittest.main()
