# Connexin-43 pipeline

Quantifies connexin-43 (Cx43) gap-junction plaques in 3D cardiac microscopy stacks and assigns each
plaque to the nucleus–nucleus junction (intercalated disc) it sits closest to.

The whole analysis exists in two equivalent forms:

- **`run_pipeline.py`** — one command that runs the full chain end to end on a sample, memory-safely.
- **`notebooks/`** — the same steps broken up, for looking at each stage in detail, tuning parameters,
  and QC. Use these to *evaluate* a single step; use the script to *run* everything.

All the real work lives in `src/` (`preprocessing`, `threshold`, `localization`, `quantification`,
`nuclei_assignment`); both the script and the notebooks call the same functions.

---

## 1. Setup

The project uses a conda environment named **`cardiac_analysis`**:

```bash
conda env create -f environment.yml
conda activate cardiac_analysis
```

The **preprocess** stage (rolling-ball background subtraction + Richardson–Lucy deconvolution) needs
`h5py` (to read the PSF) and `RedLionfish` for the deconvolution. RedLionfish uses the GPU when one is
available and falls back to CPU otherwise. If a machine cannot run the deconvolution, do the
preprocessing elsewhere and run the pipeline from the **segment** stage onward on the already
background-removed + deconvolved image (see stages below).

---

## 2. Where the data goes

**Data is not committed** — `data/` and `results/` are git-ignored (as are `*.tif`, `*.tiff`, `*.h5`).
You create `data/` locally and drop the raw files in. Each sample is one entry in the `SAMPLES` dict
at the top of `run_pipeline.py`, and that entry says exactly where its files are.

Expected layout (paths come straight from `SAMPLES` / `COMMON`):

```
data/
├── raw/
│   ├── experimental_PSF/
│   │   └── PSF_488nm_cnx.h5                       # PSF, shared by all samples (488 nm channel)
│   ├── corrected_images/
│   │   └── cnx43.tif                              # sample_1 raw connexin stack
│   ├── nuclei_dataset/
│   │   └── nuclei_1_raw_..._Input.tiff            # sample_1 labelled nuclei mask
│   ├── sample_2/
│   │   ├── <raw connexin>.tif                     # sample_2 raw connexin stack
│   │   └── <labelled nuclei>.tif                  # sample_2 labelled nuclei mask
│   └── sample_3/
│       ├── <raw connexin>.tif
│       └── <labelled nuclei>.tif
├── sample_1/                                      # sample_1 OUTPUTS (its out_dir)
├── raw/sample_2/                                  # sample_2 outputs also land inside its folder
└── raw/sample_3/
```

Two inputs per sample: the **raw connexin stack** and the **labelled nuclei mask** (an integer/label
image where each nucleus is a connected component). The **PSF** is shared across samples.

Everything a run produces is written **inside that sample's `out_dir`**, so samples never overwrite
each other:

```
<out_dir>/
├── preprocessed/
│   ├── background_removed_img.tiff
│   └── background_removed_and_deconvolved_img.tiff
├── segmented/
│   ├── binary_sauvola3d_w9_21_21_k0_15.tiff       # binary mask
│   └── labelled_sauvola3d_w9_21_21_k0_15.tiff     # 3D-labelled plaques (int32)
├── localization_binary_masks/
│   ├── region_properties.csv                      # one row per plaque (centroid, size, intensity)
│   ├── nuclei_edges.csv                           # candidate cell–cell edges (Delaunay)
│   └── matched_connexin_to_nuclei.csv             # each plaque + the nucleus pair it was assigned to
├── figures/
│   └── segmentation_comparison_plane<N>.png       # RAW vs preprocessed vs segmentation
└── summary.txt                                    # nuclei, plaques, single-voxel %, v_low/v_high, ...
```

---

## 3. Running the pipeline

```bash
conda activate cardiac_analysis

python run_pipeline.py --sample sample_1                       # all stages, sample_1
python run_pipeline.py --sample sample_2                       # a different sample
python run_pipeline.py --sample sample_2 --stages segment,localize,assign   # skip preprocess
python run_pipeline.py --help                                 # options
```

The pipeline runs five stages in order; each **loads from disk → computes → saves → frees memory**, so
only about one big array is in RAM at a time. Because every stage reads what the previous one wrote,
you can stop and resume, or re-run just the stages you're tuning with `--stages`.

| Stage        | Does                                                              | Reads → Writes |
|--------------|------------------------------------------------------------------|----------------|
| `preprocess` | rolling-ball background subtraction + Richardson–Lucy deconvolution | raw + PSF → `preprocessed/…` |
| `segment`    | 3D adapted-Sauvola threshold + 3D connected-component labelling   | deconvolved → `segmented/…` |
| `figures`    | RAW / preprocessed / segmentation comparison figure for one plane | above → `figures/…` |
| `localize`   | per-plaque property table (centroid, size, volume, mean intensity) | labels + deconvolved → `region_properties.csv` |
| `assign`     | match each plaque to the closest nucleus-pair midpoint (Delaunay) | region props + nuclei mask → `nuclei_edges.csv`, `matched_connexin_to_nuclei.csv` |

At the end it prints and writes `summary.txt`: tissue sample, number of nuclei (from the **original**
nuclei image, before any crop), number of 3D candidate plaques, single-voxel plaque fraction,
`v_low`/`v_high`, image max and shape, candidate edges, and plaques assigned.

### Adding / configuring a sample

Edit the `SAMPLES` dict in `run_pipeline.py`. Each entry needs its input filenames and its **crops**
(so the connexin and nuclei end up in the same voxel frame — the assignment relies on this):

- `border_crop_yx` — `(y0, x0)` trimmed from the raw connexin stack (`cnx[:, y0:, x0:]`).
- `seg_z_crop` / `seg_y_crop` / `seg_x_crop` — `(start, stop)` or `None`, applied to the deconvolved stack.
- `nuclei_crop` — `(z, y, x)`, each `(start, stop)` or `None`, applied to the nuclei mask.

Shared algorithm settings (Sauvola window/k, normalization percentile, spacing, assignment distances)
live in `COMMON` and apply to every sample; add the same key to a sample dict to override it there.
Optional per-sample overrides: `v_low`/`v_high` (fixed normalization bounds, skips auto-estimation) and
`figure_plane` (which plane the comparison figure uses).

---

## 4. Notebooks — step-by-step evaluation

The notebooks are for inspecting and tuning **one stage at a time** (parameter sweeps, QC plots,
validation against manual annotation). They read/write the same files the pipeline does, so you can run
a stage in a notebook, look at it, then let `run_pipeline.py` do the rest. Run them in the
`cardiac_analysis` env.

| Notebook | Pipeline stage | Use it to… |
|----------|----------------|------------|
| `0_sample_planes` | — | explore per-plane intensity / depth zones of a stack |
| `1_1_preprocessing` | preprocess | run/inspect rolling-ball + deconvolution on one image |
| `1_2_preprocessing_full_stack_test` | preprocess | QC preprocessing across the full stack |
| `1_4_connexin_size_and_radius_check` | preprocess | pick the rolling-ball radius from plaque sizes |
| `2_1_segmentation` | segment | tune Sauvola (window/k), compare settings on a plane |
| `2_2_validation` | segment | compare segmentation to the manual annotation (plane 160) |
| `2_3_segmentation_full_stack_test` | segment | QC segmentation across depth |
| `3_localization` | localize | build/inspect the per-plaque property table |
| `4_connect_nuclei` | assign | Delaunay graph + plaque→nucleus-pair assignment, 3D plots |
| `5_sample_comparison` | (post) | compare samples: paired/unpaired plaques & edges, plaques vs depth |
| `5_analysis_spatial_distribution` | (post) | spatial-distribution analysis of the plaques |
| `6_segmentation_stats_per_sample` | (post) | per-sample v_low/v_high, max, nuclei, regions, single-voxel |

> Notebooks re-run and re-save their own outputs, so committing them can produce large diffs/merge
> conflicts. Clear outputs before committing if you want clean diffs.

---

## 5. Repository layout

```
run_pipeline.py     end-to-end pipeline (edit SAMPLES / COMMON here)
environment.yml     conda environment (cardiac_analysis)
src/                the algorithms (imported by the script and the notebooks)
  preprocessing.py    rolling ball, PSF prep
  threshold.py        normalization bounds + adapted (Ghaye) Sauvola
  localization.py     3D labelling + region properties
  quantification.py   plaque counts, size filters, single-voxel count
  nuclei_assignment.py  nuclei centers, Delaunay edges, plaque→pair matching
  analysis.py         spatial-distribution helpers
notebooks/          per-stage evaluation & QC (see table above)
data/  results/      local only — git-ignored
```
