"""Harvard-Oxford atlas registration to the native low-field MRI grid."""
from pathlib import Path
import hashlib
import json
import os
import tempfile
import numpy as np
import nibabel as nib
from filelock import FileLock

REGIONS = {
    'frontal_lobe': ('cort', [1,3,4,5,6,7,25,26,27,28,29,33,41]),
    'parietal_lobe': ('cort', [17,18,19,20,21,30,31,43]),
    'temporal_lobe': ('cort', [8,9,10,11,12,13,14,15,16,34,35,37,38,39,44,45,46]),
    'occipital_lobe': ('cort', [22,23,24,32,36,40,47,48]),
    'insular_lobe': ('cort', [2]),
    'ventricles': ('sub', [3,14]),
    'white_matter': ('sub', [1,12]),
}
RESOURCE_NAMES = [
    'tpl-MNI152NLin6Asym_res-02_T1w.nii.gz',
    'tpl-MNI152NLin6Asym_res-02_desc-brain_mask.nii.gz',
    'HarvardOxford-cort-maxprob-thr25-1mm.nii.gz',
    'HarvardOxford-sub-maxprob-thr25-1mm.nii.gz',
]
SCHEMA = 1

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def cache_folder(source, cache):
    return Path(cache) / sha256(source)

def mask_paths(source, cache):
    folder = cache_folder(source, cache)
    metadata = folder / 'registration.json'
    if not metadata.is_file():
        raise FileNotFoundError(f'No registered atlas for {source}. Run scripts/prepare_atlas.py first.')
    meta = json.loads(metadata.read_text())
    if meta['schema'] != SCHEMA or meta['regions'] != {k:[v[0],v[1]] for k,v in REGIONS.items()}:
        raise ValueError('Incompatible registration cache')
    paths = {region: str(folder/f'mask_{region}.nii.gz') for region in REGIONS}
    reference = nib.load(source)
    for path in paths.values():
        mask = nib.load(path)
        if mask.shape != reference.shape or not np.allclose(mask.affine, reference.affine, atol=1e-4):
            raise ValueError(f'Registered mask geometry differs from source: {path}')
    return paths

def register(source, resources, cache, threads=4):
    os.environ.setdefault('ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS', str(threads))
    import ants
    from scipy.ndimage import binary_dilation
    source, resources, cache = Path(source), Path(resources), Path(cache)
    for name in RESOURCE_NAMES:
        if not (resources/name).is_file(): raise FileNotFoundError(resources/name)
    cache.mkdir(parents=True, exist_ok=True)
    folder = cache_folder(source, cache)
    fingerprints = {name: sha256(resources/name) for name in RESOURCE_NAMES}
    with FileLock(str(folder)+'.lock'):
        if (folder/'registration.json').is_file():
            meta = json.loads((folder/'registration.json').read_text())
            if meta['resources'] != fingerprints: raise ValueError('Atlas resources changed; use a new cache directory')
            return mask_paths(source, cache)
        folder.mkdir(exist_ok=True)
        def normalize(im):
            a = im.numpy(); values = a[np.isfinite(a) & (a>0)]
            if not len(values): raise ValueError('Registration requires a nonempty low-field volume')
            lo,hi = np.percentile(values,[0.5,99.5])
            return im.new_image_like(np.clip((np.nan_to_num(a)-lo)/max(hi-lo,1e-8),0,1).astype('float32'))
        ants.config.set_ants_deterministic(on=False, seed_value=20261102)
        native = ants.image_read(str(source))
        fixed = normalize(ants.resample_image(native,(2,2,2),use_voxels=False,interp_type=0))
        foreground = fixed.new_image_like((fixed.numpy()>.05).astype('float32'))
        fixed = normalize(ants.n4_bias_field_correction(fixed,mask=foreground,shrink_factor=2,
            convergence={'iters':[30,20,10],'tol':1e-7},spline_param=150))
        moving = normalize(ants.image_read(str(resources/RESOURCE_NAMES[0])))
        brain = ants.image_read(str(resources/RESOURCE_NAMES[1]))
        common = dict(aff_metric='mattes',aff_sampling=32,aff_random_sampling_rate=.25,
            aff_iterations=(1000,500,250,0),aff_shrink_factors=(6,4,2,1),
            aff_smoothing_sigmas=(3,2,1,0),grad_step=.1,use_legacy_histogram_matching=False)
        rigid = ants.registration(fixed,moving,type_of_transform='Rigid',outprefix=str(folder/'rigid_'),**common)
        affine = ants.registration(fixed,moving,type_of_transform='Affine',initial_transform=rigid['fwdtransforms'],outprefix=str(folder/'affine_'),**common)
        affbrain = ants.apply_transforms(fixed,brain,affine['fwdtransforms'],interpolator='nearestNeighbor')
        metricmask = fixed.new_image_like(binary_dilation(affbrain.numpy()>.5,iterations=2).astype('float32'))
        syn = ants.registration(fixed,moving,type_of_transform='SyNOnly',initial_transform=affine['fwdtransforms'],
            outprefix=str(folder/'syn_'),mask=metricmask,moving_mask=brain,mask_all_stages=True,
            syn_metric='mattes',syn_sampling=32,reg_iterations=(80,40,20),grad_step=.1,
            flow_sigma=3,total_sigma=1,use_legacy_histogram_matching=False)
        reference = nib.load(source)
        for kind in ('cort','sub'):
            labels = ants.image_read(str(resources/f'HarvardOxford-{kind}-maxprob-thr25-1mm.nii.gz'))
            warped = ants.apply_transforms(native,labels,syn['fwdtransforms'],interpolator='genericLabel')
            # Write with the original NIfTI affine and spatial metadata.
            array = np.rint(warped.numpy()).astype(np.int16)
            if not np.isin(array,np.unique(labels.numpy())).all(): raise ValueError('Invalid propagated labels')
            nib.save(nib.Nifti1Image(array,reference.affine,reference.header),folder/f'{kind}_labels.nii.gz')
            for region,(group,ids) in REGIONS.items():
                if group == kind:
                    mask = np.isin(array,ids).astype(np.uint8)
                    if not mask.any(): raise ValueError(f'Empty registered region: {region}')
                    nib.save(nib.Nifti1Image(mask,reference.affine,reference.header),folder/f'mask_{region}.nii.gz')
        metadata = {'schema':SCHEMA,'source_sha256':sha256(source),'resources':fingerprints,
            'regions':REGIONS,'antspyx':ants.__version__,'seed':20261102,
            'grid_mm':[2,2,2],'linear':common,'syn_iterations':[80,40,20],
            'shape':reference.shape,'affine':reference.affine.tolist()}
        tmp = folder/'registration.json.tmp'
        tmp.write_text(json.dumps(metadata,indent=2)+'\n')
        tmp.replace(folder/'registration.json')
    return mask_paths(source, cache)
