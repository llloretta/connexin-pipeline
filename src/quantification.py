from skimage.measure import label, regionprops
from skimage.feature import peak_local_max
from skimage.morphology import remove_small_objects
import numpy as np
import matplotlib.pyplot as plt
from threshold import adapted_sauvola_threshold
from skimage import measure


# tbd: adjust this functions to fit the localization matrix ! OR have both 

# --- functions to use on image data 
def count_connexin_plaques_within_plaque(img_float, binary, min_distance=2):
    
    # count plaques
    # -> this counts 2 peaks when 2 plaques are touching
    coordinates = peak_local_max(
        img_float,
        min_distance=min_distance,
        threshold_abs=0.05,
        labels=binary
    )
    n_plaques = len(coordinates)
    print(f"Plaques with min distance {min_distance} detected: {n_plaques}")

    return coordinates, n_plaques

# def measure_connexin_regions(binary, img_float):
#     # label connected regions for shape/size info
#     labeled = label(binary)
#     props = regionprops(labeled, intensity_image=img_float)

#     return props, labeled

# -- functions to use on binary dataframe 

def count_connexin_plaques(binary):
    # this function counts the number of connected components in a binary mask
    labeled_array = measure.label(binary)  # label the connected components in the binary mask
    num_regions = labeled_array.max()  # or len(np.unique(labeled_array)) - 
    
    # print(f"Number of connected components (plaques) detected: {num_regions}")

    return num_regions, labeled_array


def remove_regions_above_area(binary, max_area_um2=2.5, pixel_size_um=0.325):
    """Remove segmented regions whose 2D cross-section area exceeds ``max_area_um2`` (µm²).

    Filters per plane, so it accepts either a single 2D plane or a 3D (z, y, x) stack.
    Direction is remove-large (drop blobs bigger than the threshold, keep small plaques);
    use skimage.morphology.remove_small_objects for the opposite.

    Parameters
    ----------
    binary        : 2D or 3D boolean/label array (foreground = nonzero)
    max_area_um2  : float, area threshold in µm² (regions strictly above this are removed)
    pixel_size_um : float, lateral pixel size in µm (x and y, isotropic)

    Returns
    -------
    bool ndarray of the same shape, with oversized regions set to False.
    """
    max_area_px = max_area_um2 / (pixel_size_um ** 2)

    def _filter_plane(plane):
        labeled = measure.label(plane)
        sizes = np.bincount(labeled.ravel())
        remove = np.where(sizes > max_area_px)[0]
        remove = remove[remove != 0]  # keep background label 0
        out = np.asarray(plane, dtype=bool).copy()
        if remove.size:
            out[np.isin(labeled, remove)] = False
        return out

    binary = np.asarray(binary)
    if binary.ndim == 2:
        return _filter_plane(binary)
    out = np.zeros(binary.shape, dtype=bool)
    for z in range(binary.shape[0]):
        out[z] = _filter_plane(binary[z])
    return out
