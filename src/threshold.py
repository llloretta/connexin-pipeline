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
    image = image.astype(np.float64)

    if r is None:
        r = 0.5 if image.max() <= 1.0 else 128.0

    # uniform_filter accepts a tuple size for anisotropic windows
    mean = uniform_filter(image, size=window_size)
    mean_sq = uniform_filter(image ** 2, size=window_size)
    std = np.sqrt(np.clip(mean_sq - mean ** 2, 0, None))

    threshold = mean + (mean - 1) * k * (std / r - 1)
    binary = image > threshold

    return threshold, binary

def normalize_img(image): 
    # Normalize entire 3D stack at once
    normalized_img = img_as_float(rescale_intensity(image, out_range=(0, 1)))
    return  normalized_img

def savola_3D_image(preprocessed_image_stack, window_size, k=0.1, r=0.5, threeD = True): 
    
    # create an empty matrix in the same shape as the preprocessed image to store the binary results
    binary_sauvola_img = np.zeros_like(preprocessed_image_stack, dtype=bool)

    if threeD:
        # normalize and threshold the WHOLE stack at once — no slice loop
        norm_img = normalize_img(preprocessed_image_stack)
        thresh, binary_sauvola_img = adapted_sauvola_threshold_3d(
            norm_img, window_size=window_size, k=k, r=r
        )
    else: 
        for plane_idx in range(preprocessed_image_stack.shape[0]):
            norm_slice = normalize_img(preprocessed_image_stack[plane_idx])  # per-slice norm
            thresh, binary = adapted_sauvola_threshold(norm_slice, window_size, k, r=0.5)
            binary_sauvola_img[plane_idx] = binary

    return binary_sauvola_img

### adaptation for 3D savola smaller batches with stitching



def _sauvola_chunk(chunk, window_size, k, r):
    thresh = threshold_sauvola(chunk, window_size=window_size, k=k, r=r)
    return chunk > thresh

def sauvola_3d_parallel(
    preprocessed_image_stack,
    window_size,
    k=0.1,
    r=0.5,
    chunk_shape=(64, 512, 512),
):
    norm_img = normalize_img(preprocessed_image_stack).astype(np.float32)

    darr = da.from_array(norm_img, chunks=chunk_shape)

    # handle window_size as either a single int or a per-axis tuple
    if isinstance(window_size, (tuple, list)):
        depth = tuple(w // 2 + 1 for w in window_size)
    else:
        depth = window_size // 2 + 1

    binary = darr.map_overlap(
        _sauvola_chunk,
        depth=depth,
        boundary="reflect",
        window_size=window_size,
        k=k,
        r=r,
        dtype=bool,
    )

    return binary.compute()




