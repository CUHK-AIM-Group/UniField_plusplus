# Anatomical conditioning

The low-field volume alone determines the transformation. The moving image is the MNI152NLin6Asym 2 mm T1 template; label images are Harvard-Oxford 1 mm maximum-probability atlases thresholded at 25%.

1. Resample the fixed MRI to a 2 mm working grid. Clip positive finite intensity percentiles 0.5–99.5 and scale to [0,1].
2. Apply N4 bias correction, shrink factor 2, iterations [30,20,10], tolerance 1e-7, spline distance 150 mm. Normalize again.
3. Estimate rigid and affine transforms using Mattes mutual information, 32 bins, 25% sampling, iterations [1000,500,250,0], shrink factors [6,4,2,1], smoothing sigmas [3,2,1,0].
4. Propagate the template brain mask with the affine transform and dilate it by two working-grid voxels. Estimate SyN with this metric mask, Mattes MI (32 bins), iterations [80,40,20], gradient step 0.1, flow sigma 3 and total sigma 1.
5. Apply the composed transforms to discrete atlas labels on the original low-field grid with `genericLabel`. Preserve the low-field NIfTI affine and geometry.

The fixed seed is 20261102. Multithreaded ANTs execution need not be bitwise reproducible. Content-addressed caches store input/resource hashes and reject incompatible resources. Registration requires positive finite image content and nonempty regional output masks.

| Model region | Atlas | Label IDs |
|---|---|---|
| frontal_lobe | cortical | 1,3,4,5,6,7,25,26,27,28,29,33,41 |
| parietal_lobe | cortical | 17,18,19,20,21,30,31,43 |
| temporal_lobe | cortical | 8,9,10,11,12,13,14,15,16,34,35,37,38,39,44,45,46 |
| occipital_lobe | cortical | 22,23,24,32,36,40,47,48 |
| insular_lobe | cortical | 2 |
| ventricles | subcortical | 3,14 |
| white_matter | subcortical | 1,12 |

Lobar groups are custom label sets. `ventricles` denotes bilateral lateral ventricles and `white_matter` denotes cerebral white matter. These masks can overlap and do not form an exhaustive partition. Plan masks are softened using an 11-voxel average-pooling kernel at the default sigma 5. Overlapping focus contributions are combined by maximum in the latent mask. Masks use (D,H,W) order after the same spatial preprocessing, augmentation and temporal sampling as the corresponding MRI.
