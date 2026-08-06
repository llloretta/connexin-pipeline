#!/usr/bin/env python3
"""End-to-end connexin-43 pipeline as a runnable script (replaces the notebooks).

Stages (each: load from disk -> compute -> save -> free memory, so only ~one big
array is in RAM at a time; the heavy ops are already chunked internally):

    1. preprocess : rolling-ball background subtraction + Richardson-Lucy deconvolution
    2. segment    : 3D adapted Sauvola threshold + 3D connected-component labelling
    3. localize   : per-plaque property table (centroid, size, volume, mean intensity)
    4. assign     : match plaques to nuclei-pair (cell-cell contact) via Delaunay graph

All heavy lifting reuses the functions in ``src/``.

Usage
-----
    python run_pipeline.py                         # run all four stages
    python run_pipeline.py --stages segment,localize,assign
    python run_pipeline.py --base-dir /path/to/project

Edit the CONFIG dict below for a new acquisition (crops, radius, spacing, PSF path,
paths, and the optional noise filters are all there).
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
from skimage import io, img_as_ubyte
from skimage.morphology import remove_small_objects

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR / "src"))

from preprocessing import (                      # noqa: E402
    remove_background_rolling_ball_3d,
    brightest_slice,
    center_psf,
    normalize_psf,
)
from threshold import estimate_normalization_bounds, savola_3D_image  # noqa: E402
from localization import label_connexin_regions_3d, get_region_properties_3d  # noqa: E402
from quantification import remove_regions_above_area  # noqa: E402
from nuclei_assignment import (                  # noqa: E402
    get_nuclei_centerpoints,
    build_nuclei_edges,
    match_connexin_to_nuclei_pairs,
)

# --------------------------------------------------------------------------- #
# CONFIG — everything dataset-specific lives here. Adjust for a new acquisition.
# --------------------------------------------------------------------------- #
CONFIG = {
    # ---- inputs ----
    "raw_image":       "data/raw/corrected_images/cnx43.tif",
    "psf_h5":          "data/raw/experimental_PSF/PSF_488nm_cnx.h5",
    "psf_dataset":     "PSF_488/ImageData/Image",           # dataset path inside the .h5
    "nuclei_mask":     "data/raw/nuclei_dataset/nuclei_1_raw_Object Identities_test-exported_data_Input.tiff",
    "nuclei_centers_csv": "data/raw/nuclei_dataset/corrected_nuclei_center.csv",  # used if it exists; else computed

    # ---- intermediate / output files (written and re-read between stages) ----
    "bg_removed":      "data/preprocessed/background_removed_img.tiff",
    "deconvolved":     "data/preprocessed/background_removed_and_deconvolved_img.tiff",
    "binary_mask":     "data/segmented/binary_sauvola3d_w9_21_21_k0_15.tiff",
    "label_mask":      "data/segmented/labelled_sauvola3d_w9_21_21_k0_15.tiff",
    "region_props":    "data/localization_binary_masks/region_properties.csv",
    "matched":         "data/localization_binary_masks/matched_connexin_to_nuclei.csv",

    # ---- preprocessing ----
    "border_crop_yx":  (15, 3),          # cnx_img[:, 15:, 3:] — top/left black border
    "rolling_ball_radius": 10,
    "rl_iterations":   7,
    "rl_method":       "gpu",            # RedLionfish falls back to CPU if no GPU

    # ---- segmentation crop (must match how the nuclei/labels frame is defined) ----
    "seg_z_crop":      (60, 501),        # preprocessed[60:501, :, 23:999]
    "seg_x_crop":      (23, 999),

    # ---- Sauvola ----
    "sauvola_window":  (9, 21, 21),      # (z, y, x); z ~ isotropic given 0.766/0.325 spacing
    "sauvola_k":       0.15,
    "sauvola_r":       0.5,
    "sauvola_3d":      True,             # True -> 3D blockwise; False -> 2D per-plane
    "norm_low_sigma":  3.0,              # v_low  = median + low_sigma * 1.4826 * MAD
    "norm_high_pct":   95.0,             # v_high = high_pct percentile of the Otsu foreground

    # ---- optional noise filters on the binary before labelling (default: off, as in the notebook) ----
    "min_plaque_voxels": None,           # e.g. 3 -> remove_small_objects(min_size=3)
    "max_area_um2":      None,           # e.g. 2.5 -> remove_regions_above_area(2.5)

    # ---- geometry ----
    "spacing":         (0.766, 0.325, 0.325),   # (z, y, x) voxel size in micrometers
    "pixel_size_um":   0.325,                    # lateral, for the area filter

    # ---- nucleus assignment ----
    "nuclei_border_crop_y": 15,          # nuclei_mask[:, 15:, :]
    "max_edge_distance": 52.3,           # um; Delaunay edge cutoff (see thesis: ~cell length)
    "distance_threshold": 50.0,          # um; plaque-to-junction-line cutoff
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _p(rel: str) -> Path:
    """Resolve a CONFIG-relative path against BASE_DIR."""
    return BASE_DIR / rel


def _log(msg: str) -> None:
    print(f"[pipeline] {msg}", flush=True)


def _free(*names_and_values) -> None:
    """Delete large arrays and collect, to keep peak memory low between steps."""
    del names_and_values
    gc.collect()


# --------------------------------------------------------------------------- #
# Stage 1 — preprocess (from 1_1)
# --------------------------------------------------------------------------- #
def stage_preprocess(cfg: dict) -> None:
    import h5py                                   # local: only stage 1 needs these
    import RedLionfishDeconv as rl

    _log("preprocess: loading raw image")
    cnx = io.imread(_p(cfg["raw_image"]))
    cy, cx = cfg["border_crop_yx"]
    cnx = cnx[:, cy:, cx:]                         # remove black border
    _log(f"  raw cropped to {cnx.shape}")

    _log(f"  rolling-ball background subtraction (radius={cfg['rolling_ball_radius']})")
    bg_removed = remove_background_rolling_ball_3d(cnx, radius=cfg["rolling_ball_radius"], n_jobs=-1)
    del cnx; gc.collect()

    _p(cfg["bg_removed"]).parent.mkdir(parents=True, exist_ok=True)
    io.imsave(_p(cfg["bg_removed"]), bg_removed)
    _log(f"  saved {cfg['bg_removed']}")

    # ---- PSF: load, reduce to 3D, centre on the brightest slice, normalise to sum 1 ----
    _log("  preparing PSF")
    with h5py.File(_p(cfg["psf_h5"]), "r") as f:
        psf = f[cfg["psf_dataset"]][:]
    psf = np.squeeze(psf)                          # (1,1,z,y,x) -> (z,y,x)
    while psf.ndim > 3:
        psf = psf[0]
    psf = normalize_psf(center_psf(psf, brightest_slice(psf)))
    _log(f"  PSF shape {psf.shape}, sum={psf.sum():.4f}")

    _log(f"  Richardson-Lucy deconvolution (niter={cfg['rl_iterations']}, method={cfg['rl_method']})")
    image_f32 = bg_removed.astype(np.float32)
    del bg_removed; gc.collect()
    result = rl.doRLDeconvolutionFromNpArrays(
        image_f32, psf.astype(np.float32),
        niter=cfg["rl_iterations"], method=cfg["rl_method"],
    )
    del image_f32; gc.collect()

    _p(cfg["deconvolved"]).parent.mkdir(parents=True, exist_ok=True)
    io.imsave(_p(cfg["deconvolved"]), result)
    _log(f"  saved {cfg['deconvolved']}")
    del result, psf; gc.collect()


# --------------------------------------------------------------------------- #
# Stage 2 — segment (from 2_1)
# --------------------------------------------------------------------------- #
def _crop_seg(stack: np.ndarray, cfg: dict) -> np.ndarray:
    z0, z1 = cfg["seg_z_crop"]
    x0, x1 = cfg["seg_x_crop"]
    return stack[z0:z1, :, x0:x1]


def stage_segment(cfg: dict) -> None:
    _log("segment: loading deconvolved image")
    deconv = _crop_seg(io.imread(_p(cfg["deconvolved"])), cfg)
    _log(f"  cropped segmentation frame {deconv.shape}")

    v_low, v_high = estimate_normalization_bounds(
        deconv, low_sigma=cfg["norm_low_sigma"], high_percentile=cfg["norm_high_pct"])
    _log(f"  normalization bounds: v_low={v_low:.1f}, v_high={v_high:.1f}")

    _log(f"  Sauvola (window={cfg['sauvola_window']}, k={cfg['sauvola_k']}, threeD={cfg['sauvola_3d']})")
    binary = savola_3D_image(
        deconv, v_low=v_low, v_high=v_high,
        window_size=cfg["sauvola_window"], k=cfg["sauvola_k"], r=cfg["sauvola_r"],
        threeD=cfg["sauvola_3d"])
    del deconv; gc.collect()

    # ---- optional noise filters (default off) ----
    if cfg["min_plaque_voxels"]:
        n_before = int(binary.sum())
        binary = remove_small_objects(binary.astype(bool), min_size=cfg["min_plaque_voxels"])
        _log(f"  remove_small_objects(min_size={cfg['min_plaque_voxels']}): "
             f"foreground {n_before} -> {int(binary.sum())} voxels")
    if cfg["max_area_um2"]:
        binary = remove_regions_above_area(
            binary, max_area_um2=cfg["max_area_um2"], pixel_size_um=cfg["pixel_size_um"])
        _log(f"  remove_regions_above_area({cfg['max_area_um2']} um^2) applied")

    _p(cfg["binary_mask"]).parent.mkdir(parents=True, exist_ok=True)
    io.imsave(_p(cfg["binary_mask"]), img_as_ubyte(binary))
    _log(f"  saved {cfg['binary_mask']}")

    _log("  labelling connected regions in 3D")
    labeled_3d, n_components = label_connexin_regions_3d(binary)
    del binary; gc.collect()
    _log(f"  3D connected regions: {n_components}")

    # save labels as int32 (NOT img_as_ubyte, which would corrupt >255 labels)
    io.imsave(_p(cfg["label_mask"]), labeled_3d.astype(np.int32))
    _log(f"  saved {cfg['label_mask']}")
    del labeled_3d; gc.collect()


# --------------------------------------------------------------------------- #
# Stage 3 — localize (from 3)
# --------------------------------------------------------------------------- #
def stage_localize(cfg: dict) -> None:
    _log("localize: loading labels + deconvolved (intensity image)")
    labeled_3d = io.imread(_p(cfg["label_mask"]))
    deconv = _crop_seg(io.imread(_p(cfg["deconvolved"])), cfg)
    if deconv.shape != labeled_3d.shape:
        raise ValueError(f"shape mismatch: labels {labeled_3d.shape} vs deconvolved {deconv.shape}")

    _log("  extracting region properties")
    locations = get_region_properties_3d(labeled_3d, deconv, spacing=cfg["spacing"])
    del labeled_3d, deconv; gc.collect()

    _p(cfg["region_props"]).parent.mkdir(parents=True, exist_ok=True)
    locations.to_csv(_p(cfg["region_props"]), index=False)
    _log(f"  saved {cfg['region_props']} ({len(locations)} plaques)")


# --------------------------------------------------------------------------- #
# Stage 4 — assign (from 4)
# --------------------------------------------------------------------------- #
def stage_assign(cfg: dict) -> None:
    import pandas as pd

    _log("assign: loading region properties")
    df_regions = pd.read_csv(_p(cfg["region_props"]))

    centers_csv = _p(cfg["nuclei_centers_csv"])
    if centers_csv.exists():
        _log(f"  loading nuclei centers from {cfg['nuclei_centers_csv']}")
        df_centers = pd.read_csv(centers_csv)
    else:
        _log("  computing nuclei centers from the nuclei mask")
        mask = io.imread(_p(cfg["nuclei_mask"]))[:, cfg["nuclei_border_crop_y"]:, :]
        df_centers = get_nuclei_centerpoints(mask)
        del mask; gc.collect()

    _log("  building nuclei neighbour graph (Delaunay)")
    edges, nuclei_coords_um = build_nuclei_edges(
        df_centers[["z", "y", "x"]].values,
        spacing=cfg["spacing"], max_edge_distance=cfg["max_edge_distance"])

    _log("  matching plaques to nuclei pairs")
    matched = match_connexin_to_nuclei_pairs(
        df_regions, edges, nuclei_coords_um,
        spacing=cfg["spacing"], distance_threshold=cfg["distance_threshold"])

    _p(cfg["matched"]).parent.mkdir(parents=True, exist_ok=True)
    matched.to_csv(_p(cfg["matched"]), index=False)
    n_assigned = int(matched["nucleus_1"].notna().sum())
    _log(f"  saved {cfg['matched']} ({n_assigned}/{len(matched)} plaques assigned)")


# --------------------------------------------------------------------------- #
STAGES = {
    "preprocess": stage_preprocess,
    "segment":    stage_segment,
    "localize":   stage_localize,
    "assign":     stage_assign,
}


def main() -> None:
    global BASE_DIR
    parser = argparse.ArgumentParser(description="Run the connexin-43 pipeline.")
    parser.add_argument("--stages", default="preprocess,segment,localize,assign",
                        help="comma-separated subset of: preprocess,segment,localize,assign")
    parser.add_argument("--base-dir", default=None,
                        help="project root (defaults to the directory of this script)")
    args = parser.parse_args()

    if args.base_dir:
        BASE_DIR = Path(args.base_dir).resolve()

    requested = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = [s for s in requested if s not in STAGES]
    if unknown:
        parser.error(f"unknown stage(s): {unknown}; choose from {list(STAGES)}")

    for name in requested:
        _log(f"=== stage: {name} ===")
        STAGES[name](CONFIG)
    _log("done.")


if __name__ == "__main__":
    main()
