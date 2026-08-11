import numpy as np
from skimage.util import view_as_windows
from scipy.ndimage import uniform_filter, generic_filter
from skimage import img_as_float
from skimage.exposure import rescale_intensity
import dask.array as da
from skimage.filters import threshold_sauvola


def estimate_normalization_bounds(image, low_sigma=3.0, high_percentile=99.5,
                                  max_samples=4_000_000, return_debug=False):
    """Estimate robust per-image normalization bounds (v_low, v_high) from the intensity
    structure, so the normalization adapts across acquisitions instead of using fixed constants.

    Two-step, whole-stack estimate:

    v_low  : robust background floor = median + low_sigma * (1.4826 * MAD). The median/MAD track
             the dominant background peak, so this sits just above the noise regardless of gain.
    v_high : the `high_percentile` (default 99.5) of the intensity distribution of *everything
             above* v_low. Restricting the percentile to above-background voxels makes it land on
             real plaque signal (a plain global percentile fails because the plaques are sparse)
             and scale with each sample's own brightness, so it transfers across samples with
             different absolute max values.

    Large stacks are subsampled (max_samples) for speed; the statistics are unaffected.
    Returns (v_low, v_high) as floats, or (v_low, v_high, debug_dict) when return_debug=True.
    """
    arr = np.asarray(image)
    step = max(1, arr.size // max_samples)
    flat = np.asarray(arr.reshape(-1)[::step], dtype=np.float64)   # subsample the whole stack

    med = np.median(flat)
    mad = np.median(np.abs(flat - med))
    v_low = med + low_sigma * 1.4826 * mad                         # robust background floor

    fg = flat[flat > v_low]                                        # everything above background
    if fg.size < 100:                                              # too little above -> relax
        fg = flat[flat > med]
    v_high = float(np.percentile(fg, high_percentile)) if fg.size else v_low + 1.0

    if v_high <= v_low:
        v_high = v_low + 1.0

    v_low, v_high = float(v_low), float(v_high)
    if return_debug:
        return v_low, v_high, {
            "median": float(med),
            "mad": float(mad),
            "v_low": v_low,
            "v_high": v_high,
            "frac_above_bg": float(fg.size) / float(flat.size),
        }
    return v_low, v_high

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

def adapted_sauvola_3d_blockwise(stack, v_low, v_high, window_size, k=0.15, r=0.5, z_block=32):
    """Memory-safe 3D adapted Sauvola threshold (Ghaye light-on-dark).

    Same result as ``adapted_sauvola_threshold_3d`` but computed in z-blocks with a halo,
    so peak memory stays a few GB even on large stacks (the whole-array version allocates
    ~8 full-volume float copies at once and OOMs on a ~3 GB stack).

    Normalisation uses the fixed absolute bounds ``(v_low, v_high)`` — a per-element
    operation, so block-wise is identical to whole-stack. y and x are processed at full
    width per block (exact); only z is blocked, with a halo of ``wz//2 + 1`` planes so the
    kept interior planes see the correct 3D neighbourhood. Only the true top/bottom of the
    stack reflect, exactly as the whole-array version does.
    """
    ws = tuple(window_size) if isinstance(window_size, (tuple, list)) else (window_size,) * 3
    wz = ws[0]
    pad = wz // 2 + 1
    nz = stack.shape[0]
    inv = 1.0 / max(float(v_high) - float(v_low), 1e-8)
    out = np.zeros(stack.shape, dtype=bool)

    for z0 in range(0, nz, z_block):
        z1 = min(z0 + z_block, nz)
        a = max(0, z0 - pad)          # read the block with a z-halo of real neighbour planes
        b = min(nz, z1 + pad)
        # normalise this block to [0, 1] with the same absolute bounds (== rescale_intensity)
        chunk = np.clip((stack[a:b].astype(np.float32) - v_low) * inv, 0.0, 1.0)
        mean = uniform_filter(chunk, size=ws)
        mean_sq = uniform_filter(chunk * chunk, size=ws)
        std = np.sqrt(np.clip(mean_sq - mean * mean, 0, None))
        threshold = mean + (mean - 1) * k * (std / r - 1)
        binary = chunk > threshold
        out[z0:z1] = binary[z0 - a: z0 - a + (z1 - z0)]   # drop the halo, keep the interior
    return out


def savola_3D_image(preprocessed_image_stack,v_low, v_high, window_size, k=0.1, r=0.5, threeD = True):
    
    # create an empty matrix in the same shape as the preprocessed image to store the binary results
    binary_sauvola_img = np.zeros_like(preprocessed_image_stack, dtype=bool)

    if threeD:
        # 3D adapted Sauvola, computed in z-blocks so the full stack fits in memory
        binary_sauvola_img = adapted_sauvola_3d_blockwise(
            preprocessed_image_stack, v_low, v_high, window_size, k=k, r=r
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





