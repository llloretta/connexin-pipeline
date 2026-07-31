import numpy as np
from skimage.util import view_as_windows
from scipy.ndimage import uniform_filter, generic_filter
from skimage import img_as_float
from skimage.exposure import rescale_intensity
import dask.array as da
from skimage.filters import threshold_sauvola

def adapted_sauvola_threshold(image, window_size=15, k=0.2, r=None):
    """
    Ghaye et al. (2013) adapted Sauvola threshold for fluorescence microscopy.
    Designed for light objects on dark background.
    
    Formula: t(x,y) = m(x,y) + (m(x,y) - 1) * k * (s(x,y)/R - 1)
    
    Parameters
    ----------
    image       : 2D ndarray, should be float in [0,1] or uint8
    window_size : int, size of the local window (odd number recommended)
    k           : float, sensitivity parameter [0.2, 0.5] typical
    r           : float, dynamic range. Defaults to 0.5 for float [0,1] 
                  or 128 for uint8
    Returns
    -------
    threshold   : 2D ndarray of same shape as image
    binary      : 2D boolean array (True = foreground)
    """
    image = image.astype(np.float64)

    # Set R based on image type if not provided
    if r is None:
        if image.max() <= 1.0:
            r = 0.5
        else:
            r = 128.0

    # Compute local mean using uniform filter
    mean = uniform_filter(image, size=window_size)

    # Compute local std: sqrt(E[x^2] - E[x]^2)
    mean_sq = uniform_filter(image ** 2, size=window_size)
    std = np.sqrt(np.clip(mean_sq - mean ** 2, 0, None))

    # Ghaye et al. Eq. 19 — adapted for light-on-dark
    threshold = mean + (mean - 1) * k * (std / r - 1)

    binary = image > threshold

    return threshold, binary



def adapted_sauvola_threshold_3d(image, window_size=(5, 15, 15), k=0.1, r=None):
    """
    3D adaptation of Ghaye et al. (2013) Sauvola threshold.
    
    Parameters
    ----------
    image       : 3D ndarray, float in [0,1] or uint8
    window_size : int or tuple of 3 ints (z, y, x). 
                  Use a tuple if your data is anisotropic —
                  typically smaller in z than xy.
    k           : float, sensitivity parameter
    r           : float, dynamic range
    
    Returns
    -------
    threshold : 3D ndarray
    binary    : 3D boolean array
    """
    image = image.astype(np.float32)  # float32 (not 64) to keep the full-stack 3D filters in memory

    if r is None:
        r = 0.5 if image.max() <= 1.0 else 128.0

    # uniform_filter accepts a tuple size for anisotropic windows
    mean = uniform_filter(image, size=window_size)
    mean_sq = uniform_filter(image ** 2, size=window_size)
    std = np.sqrt(np.clip(mean_sq - mean ** 2, 0, None))

    threshold = mean + (mean - 1) * k * (std / r - 1)
    binary = image > threshold

    return threshold, binary

def normalize_img(image, v_low, v_high):
    """Global normalization of the whole array to [0, 1] with robust percentile clipping.

    Instead of plain min-max (where a single bright outlier voxel maps to 1.0 and
    compresses everything else), the intensity range is clipped to the p_low / p_high
    percentiles of the WHOLE array before rescaling. This is more robust for
    fluorescence z-stacks with hot pixels / bright artefacts. Percentiles are taken
    globally so relative intensity across depth is preserved (dim deep planes stay dim).

    Pass p_low=0, p_high=100 to recover plain global min-max.
    """
    # v_low, v_high = np.percentile(image, [p_low, p_high])

    if v_high <= v_low:                      # flat/degenerate input guard
        v_high = v_low + 1e-8
    normalized_img = img_as_float(
        rescale_intensity(image, in_range=(v_low, v_high), out_range=(0, 1))
    )
    return normalized_img

def savola_3D_image(preprocessed_image_stack,v_low, v_high, window_size, k=0.1, r=0.5, threeD = True): 
    
    # create an empty matrix in the same shape as the preprocessed image to store the binary results
    binary_sauvola_img = np.zeros_like(preprocessed_image_stack, dtype=bool)

    if threeD:
        # normalize and threshold the WHOLE stack at once — no slice loop
        norm_img = normalize_img(preprocessed_image_stack, v_low, v_high)
        thresh, binary_sauvola_img = adapted_sauvola_threshold_3d(
            norm_img, window_size=window_size, k=k, r=r
        )
    else:
        # normalize the WHOLE stack once (global) instead of per-slice, so deep,
        # low-signal planes are not artificially amplified into noise before the
        # per-plane 2D thresholding below
        norm_img = normalize_img(preprocessed_image_stack, v_low, v_high)
        for plane_idx in range(preprocessed_image_stack.shape[0]):
            thresh, binary = adapted_sauvola_threshold(norm_img[plane_idx], window_size, k, r=r)
            binary_sauvola_img[plane_idx] = binary

    return binary_sauvola_img





