# functions for PSF centering and deconvolution application
import numpy as np
from skimage.restoration import rolling_ball
from skimage.transform import rescale, resize
from joblib import Parallel, delayed
import numpy as np

def brightest_slice(psf: np.array) -> int:
    """
    Find the slice with the highest total intensity in a PSF stack.

    Parameters
    ----------
    psf_stack : ndarray
        3D PSF array with shape (z, y, x)

    Returns
    -------
    int
        Index of the brightest slice.
    """

    slice_sums = psf.sum(axis=(1,2))
    brightest_slice = np.argmax(slice_sums)

    return brightest_slice

import numpy as np

def center_psf(psf: np.ndarray, brightest_slice: int) -> np.ndarray: 
    """
    Centers a 3D PSF stack along the Z-axis (axis 0) so that the brightest slice
    lands exactly at the geometric center, maintaining the original array shape.
    """
    # 1. Compute the structural center index of the current array
    center_index = psf.shape[0] // 2 
    
    # 2. Calculate the directional shift required
    shift = center_index - brightest_slice 
    print(f"Centering PSF: Shifting peak from slice {brightest_slice} to {center_index} (Shift: {shift:+d} slices)")
    
    # If no shift is needed, return a copy right away
    if shift == 0:
        return psf.copy()
        
    # 3. Create an empty baseline array matching the original size exactly
    psf_centered = np.zeros_like(psf)
    
    # 4. Use slicing to transfer and bound the shifted data safely
    if shift > 0:
        # Shifting downwards (e.g., peak was at 80, moving to 104)
        # Target starts at 'shift' and goes to the end
        # Source starts at 0 and cuts off the last 'shift' elements
        psf_centered[shift:] = psf[:-shift]
    else:
        # Shifting upwards (negative shift value)
        # Target starts at 0 and goes up to the absolute shift boundary
        # Source skips the first 'abs(shift)' elements and goes to the end
        psf_centered[:shift] = psf[-shift:]
        
    return psf_centered

def normalize_psf(psf: np.array) -> np.array:
    """
    Normalize the PSF so that its total intensity sums to 1.

    Parameters
    ----------
    psf : ndarray
        3D PSF array.

    Returns
    -------
    ndarray
        Normalized PSF array.
    """
    # normalize psf
    psf_normalized = psf / psf.sum()
    
    return psf_normalized

def add_zero_padding(psf: np.array, target_z: tuple) -> np.array:
    """
    Add zero-padding to the PSF to match the target shape.

    Parameters
    ----------
    psf : ndarray
        3D PSF array.
    target_z : tuple
        Desired z-dimension for the output PSF.

    Returns
    -------
    ndarray
        Zero-padded PSF array.
    """
    pad_total = target_z - psf.shape[0]   # 510 - 192 = 318 slices that need to be added as black slices
    pad_before = pad_total // 2
    pad_after = pad_total - pad_before

    psf_padded = np.pad(
        psf,
        ((pad_before, pad_after), (0,0), (0,0)),
        mode='constant'
    )

    
    return psf_padded



def remove_background_rolling_ball_3d(stack, radius=30, n_jobs=-1, downsample=1):
    """
    Apply rolling-ball background subtraction to every z-plane independently,
    in parallel across CPU cores.

    Parameters
    ----------
    stack   : 3D array (z, y, x)
    radius  : rolling ball radius, in pixels
    n_jobs  : number of parallel workers (-1 = use all available cores)
    downsample : int >= 1. Speed-up factor. ``skimage.restoration.rolling_ball``
                 cost scales with the number of pixels, so on full-resolution
                 planes (millions of pixels) it is very slow. The background is
                 smooth, so with downsample=f > 1 each plane is shrunk by f,
                 the ball (radius/f) is rolled on the small image, and the
                 background is resized back up before subtraction — typically
                 f^2 faster (e.g. f=4 -> ~16x) with a negligible change to the
                 smooth background estimate. downsample=1 keeps the exact
                 full-resolution behaviour.

    Returns
    -------
    corrected : 3D array, same shape as stack, background-subtracted
    """
    downsample = int(downsample)

    if downsample <= 1:
        def process_plane(plane):
            background = rolling_ball(plane, radius=radius)
            return plane - background
    else:
        def process_plane(plane):
            # estimate the (smooth) background on a shrunk plane, then upsample it
            small = rescale(plane, 1.0 / downsample, order=1,
                            preserve_range=True, anti_aliasing=True)
            background_small = rolling_ball(small, radius=radius / downsample)
            background = resize(background_small, plane.shape, order=1,
                                preserve_range=True)
            corrected = plane.astype(np.float32) - background.astype(np.float32)
            np.clip(corrected, 0, None, out=corrected)          # resize can nudge bg > signal
            return corrected.astype(plane.dtype, copy=False)

    results = Parallel(n_jobs=n_jobs)(
        delayed(process_plane)(stack[z]) for z in range(stack.shape[0])
    )
    return np.stack(results, axis=0)
