#!/usr/bin/env python3
"""End-to-end connexin-43 pipeline as a runnable script (replaces the notebooks).

Stages (each: load from disk -> compute -> save -> free memory, so only ~one big
array is in RAM at a time; the heavy ops are already chunked internally):

    1. preprocess : rolling-ball background subtraction + Richardson-Lucy deconvolution
    2. segment    : 3D adapted Sauvola threshold + 3D connected-component labelling
       figures    : RAW / preprocessed / Sauvola comparison figure for one plane
    3. localize   : per-plaque property table (centroid, size, volume, mean intensity)
    4. assign     : match plaques to nuclei-pair (cell-cell contact) via Delaunay graph

All heavy lifting reuses the functions in ``src/``.

Multiple tissue samples
-----------------------
Each acquisition is one entry in the ``SAMPLES`` dict below (its input files, its
crops, and its tuning). Pick one with ``--sample``; every output for that sample is
written *inside that sample's folder* (``out_dir``), so samples never overwrite each
other. ``sample_1`` reproduces the original notebook paths exactly.

Usage
-----
    python run_pipeline.py --sample sample_1                    # all four stages
    python run_pipeline.py --sample sample_2                    # a new acquisition
    python run_pipeline.py --sample sample_2 --stages segment,localize,assign
    python run_pipeline.py --sample sample_3 --base-dir /path/to/project

For a NEW acquisition: add/complete its entry in ``SAMPLES`` (raw image + labelled
nuclei filenames and the per-sample crops), then run with ``--sample <name>``.
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
# COMMON — shared across every acquisition (same microscope / same algorithm
# settings). Override any of these per sample inside SAMPLES if one differs.
# --------------------------------------------------------------------------- #
COMMON = {
    # ---- PSF (shared: same objective / channel for every acquisition) ----
    "psf_h5":          "data/raw/experimental_PSF/PSF_488nm_cnx.h5",
    "psf_dataset":     "PSF_488/ImageData/Image",           # dataset path inside the .h5

    # ---- preprocessing ----
    "rolling_ball_radius": 10,
    "rolling_ball_downsample": 4,        # 1 = full res (slow); f>1 estimates bg on a 1/f plane (~f^2 faster)
    "rl_iterations":   7,
    "rl_method":       "gpu",            # RedLionfish falls back to CPU if no GPU

    # ---- Sauvola ----
    "sauvola_window":  (9, 21, 21),      # (z, y, x); z ~ isotropic given 0.766/0.325 spacing
    "sauvola_k":       0.15,
    "sauvola_r":       0.5,
    "sauvola_3d":      True,             # True -> 3D blockwise; False -> 2D per-plane
    "norm_low_sigma":  3.0,              # v_low  = median + low_sigma * 1.4826 * MAD
    "norm_high_pct":   99.5,             # v_high = high_pct percentile of the above-background signal

    # ---- optional noise filters on the binary before labelling (default: off, as in the notebook) ----
    "min_plaque_voxels": None,           # e.g. 3 -> remove_small_objects(min_size=3)
    "max_area_um2":      None,           # e.g. 2.5 -> remove_regions_above_area(2.5)

    # ---- geometry ----
    "spacing":         (0.766, 0.325, 0.325),   # (z, y, x) voxel size in micrometers
    "pixel_size_um":   0.325,                    # lateral, for the area filter

    # ---- nucleus assignment (matches notebook 4 cell 17) ----
    "max_edge_distance": 140.0,          # um; Delaunay edge cutoff (~ one cell length)
    "distance_threshold": 12.0,          # um; plaque-to-junction-midpoint cutoff

    # ---- QC figure ----
    "figure_plane":     160,             # z-plane (in the FULL deconvolved/raw frame) for the
                                         # RAW / preprocessed / segmented comparison figure
    "figure_vmax_pct":  99.5,            # robust display vmax percentile for the grayscale panels
}

# Output layout, written *relative to each sample's out_dir* so everything for a
# sample stays inside that sample's folder.
OUTPUTS = {
    "bg_removed":   "preprocessed/background_removed_img.tiff",
    "deconvolved":  "preprocessed/background_removed_and_deconvolved_img.tiff",
    "binary_mask":  "segmented/binary_sauvola3d_w9_21_21_k0_15.tiff",
    "label_mask":   "segmented/labelled_sauvola3d_w9_21_21_k0_15.tiff",
    "region_props": "localization_binary_masks/region_properties.csv",
    "matched":      "localization_binary_masks/matched_connexin_to_nuclei.csv",
    "edges":        "localization_binary_masks/nuclei_edges.csv",
    "seg_figure":   "figures/segmentation_comparison_plane{plane}.png",
    "summary":      "summary.txt",
}

# --------------------------------------------------------------------------- #
# SAMPLES — one entry per tissue sample. Crops are per-sample; a crop of None
# (or (0, 0) for the raw border) means "no crop on that axis".
#
# Crop conventions:
#   border_crop_yx : (y0, x0) removed from the RAW connexin stack -> cnx[:, y0:, x0:]
#   seg_{z,y,x}_crop : (start, stop) or None, applied to the DECONVOLVED stack
#   nuclei_crop    : (z, y, x) each (start, stop) or None, applied to the nuclei mask
# The segmentation crop and the nuclei crop MUST bring the connexin and the nuclei
# into the same voxel frame (same z/y/x extent) — that is what the assignment relies on.
#
# Optional per-sample overrides (add to a sample dict only if needed):
#   "v_low"/"v_high"   : fixed normalization bounds; if both set, auto-estimation is skipped.
#   "norm_high_pct", "norm_low_sigma" : override the estimator for one sample.
# --------------------------------------------------------------------------- #
SAMPLES = {
    # ---- original acquisition: reproduces the notebook paths exactly ----
    "sample_1": {
        "out_dir":            "data/sample_1",
        "raw_image":          "data/raw/corrected_images/cnx43.tif",
        "nuclei_mask":        "data/raw/nuclei_dataset/nuclei_1_raw_Object Identities_test-exported_data_Input.tiff",
        "nuclei_centers_csv": "data/raw/nuclei_dataset/corrected_nuclei_center.csv",  # used if it exists; else computed
        "border_crop_yx":     (15, 3),              # cnx[:, 15:, 3:]
        "seg_z_crop":         (60, 501),            # deconvolved[60:501, :, 23:999]
        "seg_y_crop":         None,
        "seg_x_crop":         (23, 999),
        "nuclei_crop":        (None, (15, None), None),   # nuclei_mask[:, 15:, :]
    },

    # ---- new acquisition #2 (raw connexin + labelled nuclei in data/raw/sample_2) ----
    "sample_2": {
        "out_dir":            "data/raw/sample_2",
        "raw_image":          "data/raw/sample_2/cnx43_1st_1X_pos1_raw_cropped.tif",   # TODO: exact filename
        "nuclei_mask":        "data/raw/sample_2/sample_01_labels_all.tif", # TODO: exact filename
        "nuclei_centers_csv": None,                  # computed from the labelled nuclei mask
        "border_crop_yx":     (0, 0),                # TODO: raw connexin border crop, if any
        "seg_z_crop":         (70,470),                  # TODO: (z0, z1) to match the nuclei frame
        "seg_y_crop":         None,                  # TODO: (y0, y1)
        "seg_x_crop":         None,                  # TODO: (x0, x1)
        "nuclei_crop":        ((70,470), None, None),    # TODO: crop the nuclei mask into the same frame
    },

    # ---- new acquisition #3 (raw connexin + labelled nuclei in data/raw/sample_3) ----
    "sample_3": {
        "out_dir":            "data/raw/sample_3",
        "raw_image":          "data/raw/sample_3/cnx43_1st_1X_pos2_cropped.tif",   # TODO: exact filename
        "nuclei_mask":        "data/raw/sample_3/sample_02_labels_all.tif", # TODO: exact filename
        "nuclei_centers_csv": None,
        "border_crop_yx":     (0, 0),                # TODO
        "seg_z_crop":         (70,470),                  # TODO
        "seg_y_crop":         None,                  # TODO
        "seg_x_crop":         None,                  # TODO
        "nuclei_crop":        ((70,470), None, None),    # TODO
    },
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _p(rel: str) -> Path:
    """Resolve a project-relative path against BASE_DIR."""
    return BASE_DIR / rel


def _out(cfg: dict, key: str, **fmt) -> Path:
    """Resolve an output path inside this sample's out_dir.

    Any ``{name}`` placeholders in the OUTPUTS template are filled from **fmt
    (e.g. _out(cfg, "seg_figure", plane=160)).
    """
    rel = OUTPUTS[key].format(**fmt) if fmt else OUTPUTS[key]
    return BASE_DIR / cfg["out_dir"] / rel


def _log(msg: str) -> None:
    print(f"[pipeline] {msg}", flush=True)


def _slice(spec) -> slice:
    """Turn a (start, stop) tuple (or None) into a slice; None -> full axis."""
    if spec is None:
        return slice(None)
    start, stop = spec
    return slice(start, stop)


def _apply_crop(arr: np.ndarray, zc=None, yc=None, xc=None) -> np.ndarray:
    """Crop a (z, y, x) array; each of zc/yc/xc is a (start, stop) tuple or None."""
    return arr[_slice(zc), _slice(yc), _slice(xc)]


def resolve_config(sample: str) -> dict:
    """Merge COMMON + the chosen sample's overrides into a single flat config."""
    if sample not in SAMPLES:
        raise KeyError(f"unknown sample {sample!r}; choose from {list(SAMPLES)}")
    cfg = dict(COMMON)
    cfg.update(SAMPLES[sample])
    cfg["sample"] = sample
    return cfg


# --------------------------------------------------------------------------- #
# Stage 1 — preprocess (from 1_1)
# --------------------------------------------------------------------------- #
def stage_preprocess(cfg: dict, summary: dict) -> None:
    import h5py                                   # local: only stage 1 needs these
    import RedLionfishDeconv as rl

    _log("preprocess: loading raw image")
    cnx = io.imread(_p(cfg["raw_image"]))
    cy, cx = cfg["border_crop_yx"]
    cnx = cnx[:, cy:, cx:]                         # remove black border
    _log(f"  raw cropped to {cnx.shape}")

    _log(f"  rolling-ball background subtraction (radius={cfg['rolling_ball_radius']}, "
         f"downsample={cfg['rolling_ball_downsample']})")
    bg_removed = remove_background_rolling_ball_3d(
        cnx, radius=cfg["rolling_ball_radius"], n_jobs=-1,
        downsample=cfg["rolling_ball_downsample"])
    del cnx; gc.collect()

    _out(cfg, "bg_removed").parent.mkdir(parents=True, exist_ok=True)
    io.imsave(_out(cfg, "bg_removed"), bg_removed)
    _log(f"  saved {_out(cfg, 'bg_removed')}")

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

    _out(cfg, "deconvolved").parent.mkdir(parents=True, exist_ok=True)
    io.imsave(_out(cfg, "deconvolved"), result)
    _log(f"  saved {_out(cfg, 'deconvolved')}")
    del result, psf; gc.collect()


# --------------------------------------------------------------------------- #
# Stage 2 — segment (from 2_1)
# --------------------------------------------------------------------------- #
def _crop_seg(stack: np.ndarray, cfg: dict) -> np.ndarray:
    return _apply_crop(stack, cfg["seg_z_crop"], cfg.get("seg_y_crop"), cfg["seg_x_crop"])


def stage_segment(cfg: dict, summary: dict) -> None:
    _log("segment: loading deconvolved image")
    deconv = _crop_seg(io.imread(_out(cfg, "deconvolved")), cfg)
    _log(f"  cropped segmentation frame {deconv.shape}")

    if cfg.get("v_low") is not None and cfg.get("v_high") is not None:
        v_low, v_high = float(cfg["v_low"]), float(cfg["v_high"])
        _log(f"  normalization bounds (manual override): v_low={v_low:.1f}, v_high={v_high:.1f}")
    else:
        v_low, v_high, dbg = estimate_normalization_bounds(
            deconv, low_sigma=cfg["norm_low_sigma"], high_percentile=cfg["norm_high_pct"],
            return_debug=True)
        _log(f"  median={dbg['median']:.1f}, above-background fraction={dbg['frac_above_bg']:.3f}")
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

    _out(cfg, "binary_mask").parent.mkdir(parents=True, exist_ok=True)
    io.imsave(_out(cfg, "binary_mask"), img_as_ubyte(binary))
    _log(f"  saved {_out(cfg, 'binary_mask')}")

    _log("  labelling connected regions in 3D")
    labeled_3d, n_components = label_connexin_regions_3d(binary)
    del binary; gc.collect()
    _log(f"  3D connected regions (candidate plaques): {n_components}")
    summary["n_plaques_3d"] = int(n_components)

    # save labels as int32 (NOT img_as_ubyte, which would corrupt >255 labels)
    io.imsave(_out(cfg, "label_mask"), labeled_3d.astype(np.int32))
    _log(f"  saved {_out(cfg, 'label_mask')}")
    del labeled_3d; gc.collect()


# --------------------------------------------------------------------------- #
# Figures — per-sample QC (RAW vs preprocessed vs Sauvola segmentation, one plane)
# --------------------------------------------------------------------------- #
def stage_figures(cfg: dict, summary: dict) -> None:
    import tifffile
    import matplotlib
    matplotlib.use("Agg")               # file output only, no display
    import matplotlib.pyplot as plt

    plane = int(cfg["figure_plane"])
    z0 = cfg["seg_z_crop"][0] if cfg["seg_z_crop"] else 0

    # cropped-frame depth of the mask, read cheaply from the file header
    with tifffile.TiffFile(_out(cfg, "binary_mask")) as tf:
        n_cropped = tf.series[0].shape[0]

    # figure_plane is in the FULL (raw/deconvolved) frame; clamp it to the segmented range
    plane = int(np.clip(plane, z0, z0 + n_cropped - 1))
    mask_idx = plane - z0
    if plane != cfg["figure_plane"]:
        _log(f"figures: requested plane {cfg['figure_plane']} outside segmented range "
             f"[{z0}, {z0 + n_cropped - 1}]; using plane {plane}")

    yc, xc = cfg.get("seg_y_crop"), cfg["seg_x_crop"]
    cy, cx = cfg["border_crop_yx"]

    # single-plane reads (avoid loading whole stacks)
    raw_plane = tifffile.imread(_p(cfg["raw_image"]), key=plane)[cy:, cx:]     # border crop
    raw_plane = raw_plane[_slice(yc), _slice(xc)]                             # segmentation crop
    deconv_plane = tifffile.imread(_out(cfg, "deconvolved"), key=plane)[_slice(yc), _slice(xc)]
    seg_plane = tifffile.imread(_out(cfg, "binary_mask"), key=mask_idx) > 0

    pct = cfg["figure_vmax_pct"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(raw_plane, cmap="gray", vmin=0, vmax=np.percentile(raw_plane, pct))
    axes[0].set_title(f"RAW (plane {plane})")
    axes[1].imshow(deconv_plane, cmap="gray", vmin=0, vmax=np.percentile(deconv_plane, pct))
    axes[1].set_title("Preprocessed (bg-removed + deconvolved)")
    axes[2].imshow(deconv_plane, cmap="gray", vmin=0, vmax=np.percentile(deconv_plane, pct))
    axes[2].contour(seg_plane, levels=[0.5], colors="red", linewidths=0.5)
    axes[2].set_title("Sauvola segmentation (red)")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"{cfg['sample']} — segmentation comparison, plane {plane}")
    fig.tight_layout()

    out_png = _out(cfg, "seg_figure", plane=plane)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log(f"  saved {out_png}")


# --------------------------------------------------------------------------- #
# Stage 3 — localize (from 3)
# --------------------------------------------------------------------------- #
def stage_localize(cfg: dict, summary: dict) -> None:
    _log("localize: loading labels + deconvolved (intensity image)")
    labeled_3d = io.imread(_out(cfg, "label_mask"))
    deconv = _crop_seg(io.imread(_out(cfg, "deconvolved")), cfg)
    if deconv.shape != labeled_3d.shape:
        raise ValueError(f"shape mismatch: labels {labeled_3d.shape} vs deconvolved {deconv.shape}")

    _log("  extracting region properties")
    locations = get_region_properties_3d(labeled_3d, deconv, spacing=cfg["spacing"])
    del labeled_3d, deconv; gc.collect()

    _out(cfg, "region_props").parent.mkdir(parents=True, exist_ok=True)
    locations.to_csv(_out(cfg, "region_props"), index=False)
    _log(f"  saved {_out(cfg, 'region_props')} ({len(locations)} plaques)")
    summary["n_plaques_3d"] = int(len(locations))


# --------------------------------------------------------------------------- #
# Stage 4 — assign (from 4)
# --------------------------------------------------------------------------- #
def stage_assign(cfg: dict, summary: dict) -> None:
    import pandas as pd

    _log("assign: loading region properties")
    df_regions = pd.read_csv(_out(cfg, "region_props"))
    summary.setdefault("n_plaques_3d", int(len(df_regions)))

    centers_csv = _p(cfg["nuclei_centers_csv"]) if cfg["nuclei_centers_csv"] else None
    if centers_csv is not None and centers_csv.exists():
        _log(f"  loading nuclei centers from {cfg['nuclei_centers_csv']}")
        df_centers = pd.read_csv(centers_csv)
    else:
        _log("  computing nuclei centers from the labelled nuclei mask")
        mask = _apply_crop(io.imread(_p(cfg["nuclei_mask"])), *cfg["nuclei_crop"])
        df_centers = get_nuclei_centerpoints(mask)
        del mask; gc.collect()
    summary["n_nuclei"] = int(len(df_centers))
    _log(f"  nuclei: {summary['n_nuclei']}")

    _log("  building nuclei neighbour graph (Delaunay)")
    edges, nuclei_coords_um = build_nuclei_edges(
        df_centers[["z", "y", "x"]].values,
        spacing=cfg["spacing"], max_edge_distance=cfg["max_edge_distance"])

    _out(cfg, "edges").parent.mkdir(parents=True, exist_ok=True)
    edges.to_csv(_out(cfg, "edges"), index=False)
    summary["n_edges"] = int(len(edges))
    _log(f"  saved {_out(cfg, 'edges')} ({len(edges)} candidate cell-cell edges)")

    _log("  matching plaques to nuclei pairs")
    matched = match_connexin_to_nuclei_pairs(
        df_regions, edges, nuclei_coords_um,
        spacing=cfg["spacing"], distance_threshold=cfg["distance_threshold"])

    _out(cfg, "matched").parent.mkdir(parents=True, exist_ok=True)
    matched.to_csv(_out(cfg, "matched"), index=False)
    n_assigned = int(matched["nucleus_1"].notna().sum())
    summary["n_assigned"] = n_assigned
    _log(f"  saved {_out(cfg, 'matched')} ({n_assigned}/{len(matched)} plaques assigned)")


# --------------------------------------------------------------------------- #
STAGES = {
    "preprocess": stage_preprocess,
    "segment":    stage_segment,
    "figures":    stage_figures,
    "localize":   stage_localize,
    "assign":     stage_assign,
}


def _report_summary(cfg: dict, summary: dict) -> None:
    """Print and save the requested per-sample numbers."""
    lines = [
        "==================== pipeline summary ====================",
        f"Tissue sample              : {summary.get('sample', cfg['sample'])}",
        f"Number of nuclei           : {summary.get('n_nuclei', 'n/a')}",
        f"Number of 3D candidate plaques : {summary.get('n_plaques_3d', 'n/a')}",
    ]
    if "n_edges" in summary:
        lines.append(f"Candidate cell-cell edges  : {summary['n_edges']}")
    if "n_assigned" in summary:
        lines.append(f"Plaques assigned to a pair : {summary['n_assigned']}")
    lines.append("=========================================================")
    text = "\n".join(lines)
    print(text, flush=True)

    out = _out(cfg, "summary")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")
    _log(f"summary written to {out}")


def main() -> None:
    global BASE_DIR
    parser = argparse.ArgumentParser(description="Run the connexin-43 pipeline.")
    parser.add_argument("--sample", default="sample_1",
                        help=f"which acquisition to run; choose from {list(SAMPLES)}")
    parser.add_argument("--stages", default="preprocess,segment,figures,localize,assign",
                        help="comma-separated subset of: preprocess,segment,figures,localize,assign")
    parser.add_argument("--base-dir", default=None,
                        help="project root (defaults to the directory of this script)")
    args = parser.parse_args()

    if args.base_dir:
        BASE_DIR = Path(args.base_dir).resolve()

    if args.sample not in SAMPLES:
        parser.error(f"unknown sample {args.sample!r}; choose from {list(SAMPLES)}")
    cfg = resolve_config(args.sample)

    requested = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = [s for s in requested if s not in STAGES]
    if unknown:
        parser.error(f"unknown stage(s): {unknown}; choose from {list(STAGES)}")

    summary = {"sample": args.sample}
    _log(f"### tissue sample: {args.sample} | out_dir: {cfg['out_dir']} ###")
    for name in requested:
        _log(f"=== stage: {name} ===")
        STAGES[name](cfg, summary)

    _report_summary(cfg, summary)
    _log("done.")


if __name__ == "__main__":
    main()
