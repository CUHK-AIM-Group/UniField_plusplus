"""Paired MRI loading with spatially synchronized registered regional masks."""
import csv
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import monai.transforms as T
from .registration import mask_paths, REGIONS

class RegisteredMRIDataset(Dataset):
    def __init__(self, data_list_file, spatial_size=(256,256,160), num_frames=40,
                 augment=True, use_monai=True, full_volume=False):
        if not use_monai: raise ValueError('Registered MRI loading requires MONAI')
        self.num_frames, self.augment, self.full_volume = num_frames, augment, full_volume
        self.cache = os.environ.get('UNIFIELD_ATLAS_CACHE','cache/atlas')
        with open(data_list_file,newline='') as f:
            self.data_pairs = [tuple(x.strip() for x in row[:2]) for row in csv.reader(f)
                               if len(row)>=2 and not row[0].lstrip().startswith('#')]
        if not self.data_pairs: raise ValueError(f'No MRI pairs in {data_list_file}')
        keys = ['lq','hq',*REGIONS]
        modes = ['bilinear','bilinear']+['nearest']*len(REGIONS)
        self.preprocess = T.Compose([
            T.LoadImaged(keys,image_only=True), T.EnsureChannelFirstd(keys),
            T.Spacingd(keys,pixdim=(1,1,1),mode=modes),
            T.ScaleIntensityRangePercentilesd(['lq','hq'],lower=.5,upper=99.5,b_min=-1,b_max=1,clip=True),
            T.Resized(keys,spatial_size=spatial_size,mode=modes),
            T.EnsureTyped(keys,dtype=torch.float32,track_meta=False),
        ])
    def __len__(self): return len(self.data_pairs)
    def __getitem__(self,index):
        low,high = self.data_pairs[index]
        data = self.preprocess({'lq':low,'hq':high,**mask_paths(low,self.cache)})
        if self.augment:
            choice = np.random.choice(4,p=[.5,.17,.17,.16])
            if choice:
                for key in data: data[key] = torch.flip(data[key],[int(choice)])
        depth = data['lq'].shape[-1]
        if self.full_volume:
            indices = torch.arange(depth)
        elif depth >= self.num_frames:
            start = int(np.random.randint(0,depth-self.num_frames+1))
            indices = torch.arange(start,start+self.num_frames)
        else:
            indices = torch.arange(self.num_frames).clamp_max(depth-1)
        def video(x): return x.index_select(-1,indices).permute(0,3,1,2).contiguous()
        return {'lq':video(data['lq']).repeat(3,1,1,1),'hq':video(data['hq']).repeat(3,1,1,1),
                'lq_path':low,'hq_path':high,'flip_axes':[],
                'region_masks':torch.cat([video(data[r]) for r in REGIONS],dim=0)}
