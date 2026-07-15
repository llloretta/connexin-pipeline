import numpy as np
import pandas as pd
from skimage.measure import label, regionprops_table


def label_connexin_regions_3d(binary_stack, connectivity=1):
    """
    Label connected regions across the full 3D binary stack (no splitting/watershed).

    Parameters
    ----------
    binary_stack : 3D boolean array (z, y, x), e.g. output of Sauvola thresholding
    connectivity : int, 1-3 (see skimage.measure.label). 1 = 6-connected (strict),
                   3 = 26-connected (permissive). Default 1.
                   Note: connectivity is purely topological (voxel adjacency) and is
                   NOT affected by voxel spacing/anisotropy.

    Returns
    -------
    labeled_3d   : 3D integer array, each connected region has a unique label (0 = background)
    n_components : total number of connected regions found
    """
    print("Labeling connected regions in 3D...")
    labeled_3d = label(binary_stack, connectivity=connectivity)
    n_components = labeled_3d.max()
    print(f"  Connected regions found: {n_components}")

    return labeled_3d, n_components


def get_region_properties_3d(labeled_3d, stack_float, spacing=(1.0, 0.325, 0.325)):
    """
    Build a table of centre coordinates (voxel units), size, and mean intensity
    for each labeled region, accounting for anisotropic voxel spacing for volume.

    Parameters
    ----------
    labeled_3d  : 3D integer array from label_connexin_regions_3d (0 = background)
    stack_float : 3D float array (z, y, x), same shape as labeled_3d, normalized to [0,1]
    spacing     : tuple (z, y, x) physical size of one voxel in µm.
                  Default (1.0, 0.325, 0.325) matches z-step 1 µm, xy pixel 0.325 µm.
                  Only used to compute physical volume (µm^3), not centroid position.

    Returns
    -------
    locations : DataFrame with columns:
                [label, z, y, x, area_voxels, volume_um3, mean_intensity]
                z, y, x are the region's centroid in voxel (array index) units.
    """
    # centroid without spacing -> voxel-index units directly
    props = regionprops_table(
        labeled_3d,
        intensity_image=stack_float,
        properties=('label', 'centroid', 'area', 'mean_intensity')
    )

    locations = pd.DataFrame(props)
    locations = locations.rename(columns={
        'centroid-0': 'z',
        'centroid-1': 'y',
        'centroid-2': 'x',
        'area': 'area_voxels',
    })

    voxel_volume = spacing[0] * spacing[1] * spacing[2]
    locations['volume_um3'] = locations['area_voxels'] * voxel_volume

    locations = locations[[
        'label', 'z', 'y', 'x', 'area_voxels', 'volume_um3', 'mean_intensity'
    ]]

    print(f"  Regions summarized: {len(locations)}")
    return locations