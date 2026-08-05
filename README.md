# grain-morph

Classical computer-vision morphometry + QC pipeline for backlit silhouette
grain images (CAMSIZER X2). Given a directory of raw frames, it detects
individual grains, measures their size and shape (Feret diameters, EFD
harmonics, Wadell roundness/sphericity, curvature entropy), flags -- but
never silently drops -- anything that looks unreliable (defocused, border-
clipped, a sliver, a probable agglomerate), and produces per-sample summary
tables plus QC review figures.

## Install

Requires Python >= 3.11. `wadell_rs` (Wadell roundness/sphericity) installs
directly from its git repository -- there is no `fast_rs`/pure-Python
fallback in this codebase; that dependency must build successfully for
`grain-morph` to install at all.

**uv** (recommended -- this repo ships a pinned `uv.lock`):

```bash
uv sync                 # runtime dependencies only
uv sync --extra dev      # + pytest, ruff, mypy, psutil
```

**pip**, in a virtualenv:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .          # runtime dependencies only
pip install -e ".[dev]"   # + pytest, ruff, mypy, psutil
```

Either path installs the `grain-morph` console script (`grain_morph.cli:app`,
built with Typer).

## Quickstart

Every command reads a config via `--config` (deep-merged on top of the
packaged `configs/default.yaml`; see the reference table below). **Per-camera
calibration (`calibration.um_per_px.basic` / `.zoom`) is required** -- it
defaults to `null` in the packaged config, and `detect` aborts loudly
(`KeyError`) before processing a single frame if any camera among the
discovered frames has no calibration set. Put your real values in a small
override YAML, e.g.:

```yaml
# my_config.yaml
calibration:
  um_per_px: { basic: 20.0, zoom: 8.0 }
```

Three-command pipeline:

```bash
# Stage 1: detect + measure + QC-flag every object in every frame under FRAMES
grain-morph detect FRAMES OUT --config my_config.yaml

# Stage 2: fold the per-grain table into per-(sample, camera) summaries
grain-morph aggregate OUT/grains OUT --config my_config.yaml

# Stage 3: render QC review figures (contact sheets, focus scatter, etc.)
grain-morph report OUT/grains FRAMES OUT --config my_config.yaml
```

`detect` is resumable: re-running it over the same `OUT` only reprocesses
frames that are new, changed, or missing from the manifest (`--force`
reprocesses everything from a clean slate). Add `--jobs N` to override
`runtime.n_jobs` for one run.

A fourth command, `grain-morph make-fixtures SRC DEST --factor N`,
anti-aliased-downsamples a directory of frames by an integer factor
(filenames preserved) -- used to build the small committed test fixtures
under `tests/fixtures/real/`; not part of the analysis pipeline itself.

## Config reference

Every key in `configs/default.yaml`. Any subset can be overridden via a user
YAML passed to `--config` (deep-merged; unspecified keys keep their default).
An unrecognized key raises immediately (config models forbid extra fields).

| Key | Description |
|---|---|
| `calibration.um_per_px.basic` / `.zoom` | Micrometers per pixel for each camera. Required -- `null` (the default) makes `detect` fail loudly before processing any frame. |
| `filename.sample_regex` | Regex applied to a frame's filename stem, capturing `sample`, `cam`, and either `frame` (a frame index) or the literal `back_token`. |
| `filename.camera_map` | Maps the single-letter camera code captured by `cam` (default `b`/`z`) to a full camera name (`basic`/`zoom`), matching the keys under `calibration.um_per_px`. |
| `filename.back_token` | Literal token (default `back`) identifying a blank/background frame in the filename. |
| `flatfield.method` | `auto` (blank-division when a paired blank exists, else morphological), `blank` (require a blank, error if missing), or `morphological` (grey-closing background estimate from the image itself). |
| `flatfield.morph_kernel_px` | Structuring-element size (px) for the morphological background estimate. |
| `threshold.method` | `half_max` (default) or `otsu`. |
| `threshold.half_max_fraction` | Fraction of the way from the estimated core intensity back up to background (1.0) used as the `half_max` threshold level. |
| `threshold.core_percentile` | Low percentile of Otsu-dark pixels used to estimate the opaque-object core intensity for `half_max`. |
| `detect.min_area_px` | Minimum connected-component area (px) kept as a candidate object. |
| `detect.fill_holes` | Whether interior holes are filled before labeling. |
| `measure.efd_order` | Number of elliptic Fourier descriptor harmonics computed per grain. |
| `measure.efd_resample_n` | Number of boundary points the polygon is resampled to before computing EFDs. |
| `measure.wadell_smoothing` | Boundary-smoothing strength (`alpha_ratio`) passed to `wadell_rs` for Wadell roundness/sphericity. |
| `measure.curvature_resample_n` | Number of boundary points the polygon is resampled to before computing curvature entropy. |
| `measure.curvature_smoothing` | Gaussian smoothing sigma (in resampled-point units) applied to the boundary before differentiating for curvature entropy. |
| `measure.curvature_bins` | Number of histogram bins used to estimate the curvature distribution for curvature entropy. |
| `qc.min_ecd_px` | Minimum equivalent circular diameter (px); below this, `flag_too_small`. |
| `qc.defocus_edge_width_px` | 10-90% boundary intensity-rise distance (px) above which `flag_defocus` is set. |
| `qc.defocus_contrast_min` | Local core/background contrast below which `flag_defocus` is set. |
| `qc.sliver_aspect_ratio` | Aspect ratio above which a small object may be flagged `flag_sliver`. |
| `qc.sliver_max_ecd_px` | Only objects at or below this ECD (px) are eligible for `flag_sliver`. |
| `qc.agglomerate_solidity_max` | Solidity below which `flag_possible_agglomerate` is set. |
| `qc.disqualifying_flags` | Flag names that `aggregate` treats as disqualifying (i.e. `qc_pass = False`) by default. |
| `output.format` | Tabular output format: `parquet`, `csv`, or `feather`. Parquet/feather are lossless; CSV is a lossy, text-based inspection fallback. |
| `output.save_contours` | Whether subpixel boundary polygons (WKT) are persisted alongside the per-grain measurements. |
| `output.partition` | Whether parquet output is written hive-partitioned by `(sample_id, camera)`. |
| `runtime.n_jobs` | Worker processes `joblib.Parallel` uses to process frames (`-1` = all cores). Overridden per call by `--jobs` / `run_detect(n_jobs=...)`. |
| `runtime.chunk_size` | Grain rows buffered in memory before each incremental write, bounding peak RSS on long runs. |

## How the QC gate works, and how to tune it

QC is **flag, never drop**: every detected object is measured and every flag
computed and recorded, no matter how bad it looks. Nothing is discarded at
`detect` time. Rejection only happens downstream, at `aggregate`, as a
recomputed function of `cfg.qc.disqualifying_flags` -- so you can rerun
`aggregate` with a stricter or looser gate against the same `detect` output
without ever re-running detection.

**Thresholding (finding the grain silhouette).** The default `half_max`
method estimates where the opaque grain "ends" and background "begins" on a
flat-fielded frame (background normalized to ~1.0, objects darker). It first
finds the Otsu split between foreground and background, takes the darkest
`threshold.core_percentile`% of pixels below that split as an estimate of
the fully-opaque object core, then places the actual object/background
boundary a fraction (`threshold.half_max_fraction`, default 0.5 -- literally
"half max") of the way back up from that core toward background. Using an
Otsu split to isolate the dark pixel population (rather than, say, the whole
frame's median) keeps the core estimate accurate even under a residual
illumination gradient or when the object covers only a small part of the
frame.

**Flat-fielding (removing illumination gradients).** Real backlit frames are
rarely uniformly lit. `flatfield.method: auto` (the default) divides each
frame by a per-pixel illumination-field estimate: a measured blank/background
frame when one is paired to the data frame (`blank` division), or -- when no
blank is available -- a morphological estimate computed from the frame
itself via grey closing (`morphological`, tuned by `morph_kernel_px`), which
fills in dark compact objects while preserving the background level. Force
one method or the other with `flatfield.method: blank` / `morphological`
instead of `auto` if you want to guarantee which is used (`blank` raises if
no blank frame is available).

**Defocus (the sharpness gate).** `flag_defocus` is the pipeline's most
important discriminator between a trustworthy grain measurement and a blurry
out-of-plane blob. It's driven by two metrics, both measured on the *raw*
(pre-flat-field) image, since raw sensor values carry the true optical blur
signature:

- `edge_width_px` -- median 10%-90% intensity-rise distance, sampled along
  outward normals at points around the object boundary. A sharp edge rises
  over a couple of pixels; a defocused edge rises gradually over many more.
  Flagged when it exceeds `qc.defocus_edge_width_px` (default 3.0 px).
- `contrast` -- `(local_background_mean - core_mean) / local_background_mean`,
  from a small eroded core and a dilated ring just outside the object.
  Flagged when it falls below `qc.defocus_contrast_min` (default 0.90).

An object is flagged defocused if *either* condition trips. If you're seeing
too many sharp grains flagged, loosen (raise `defocus_edge_width_px` and/or
lower `defocus_contrast_min`); if blurry grains are slipping through,
tighten in the opposite direction. `report`'s `focus_scatter.png` plots
every grain's `edge_gradient_norm` vs. `contrast` colored by `flag_defocus`,
which is the fastest way to see where your current threshold sits relative
to the actual data.

**Other flags:** `flag_border` (object touches the frame edge -- likely
truncated), `flag_too_small` (`ecd_px` below `qc.min_ecd_px`),
`flag_sliver` (aspect ratio above `qc.sliver_aspect_ratio` *and* ECD at or
below `qc.sliver_max_ecd_px` -- catches small elongated debris/fibers, not
large elongated grains), `flag_possible_agglomerate` (solidity below
`qc.agglomerate_solidity_max` -- a concave outline suggests multiple grains
stuck together), and `flag_no_polygon` (contour extraction failed for this
detection).

**Tuning the pass/fail gate.** `qc.disqualifying_flags` (default: all of
`flag_defocus`, `flag_border`, `flag_too_small`, `flag_sliver`,
`flag_no_polygon` -- notably **not** `flag_possible_agglomerate`, which is
informational by default) lists which flags make `qc_pass = False` when
`aggregate` recomputes it. Drop a flag from this list to stop it from
rejecting grains (it's still recorded and still visible in `summary`'s
per-flag rejection counts), or add `flag_possible_agglomerate` to reject
those too.

**Seeing the effect of your thresholds.** `aggregate` writes
`rejection_by_ecd` -- rejection rate binned into Wentworth-scale ECD classes
(very-fine sand through granule and up), per `(sample_id, camera)` -- so a
size-dependent gate (e.g. small grains disproportionately flagged defocused)
is visible in the numbers, not just implied. `report` renders the same idea
as `rejection_vs_ecd.png`, alongside `focus_scatter.png` and per-flag
contact sheets (`contact_sheet_rejected_<flag>_NN.png`, tiled crops of the
actual rejected grains) and `contact_sheet_accepted.png` (an ECD-stratified
sample of what passed) -- the fastest way to confirm the gate is rejecting
the right things before trusting `summary`'s D10/D50/D90.

## Descriptors

Beyond the geometric basics (area, perimeter, Feret max/min, major/minor
axis, aspect ratio, solidity, convexity, circularity, extent, eccentricity),
three richer shape descriptors are computed per grain:

- **Elliptic Fourier descriptors (EFD)** (`measure.efd_order` harmonics,
  via `pyefd`, normalized for rotation/starting-point/scale invariance) --
  a full boundary shape signature, plus `fourier_power_cum_90`, the smallest
  harmonic index whose cumulative power reaches 90% of the total (a
  complexity summary in one number).
- **Wadell (1932) roundness and sphericity** (`wadell_roundness`,
  `wadell_sphericity`), via `wadell_rs`, the classical sedimentology
  angularity measures -- roundness compares corner curvature to the
  inscribed circle; sphericity compares the object to its area-equivalent
  circle.
- **Curvature entropy** -- the Shannon entropy of the boundary's local
  (signed) curvature distribution, normalized to `[0, 1]`. A smooth, regular
  outline concentrates curvature into a few histogram bins (low entropy); a
  rough, crenulated outline spreads it across many (high entropy). A
  scale-free boundary-complexity scalar that complements EFD and Wadell
  roundness rather than duplicating them.

## Parallelization

`grain-morph detect` fans frames out across **process**-based workers via
`joblib.Parallel(backend="loky", return_as="generator")` (not threads --
each worker does real CPU-bound image work). Control worker count with
`--jobs` (CLI) or `runtime.n_jobs` (config; `-1` = all cores; `1` runs
everything in the calling process, useful for deterministic tests/debugging).
Each worker process pins its own BLAS/OpenMP thread count to 1
(`joblib.parallel_config(inner_max_num_threads=1)`) so N worker processes
don't each also spawn their own internal thread pools and oversubscribe the
machine.

Memory stays bounded on long runs: each frame's grain rows are written to
disk as soon as that frame completes, keyed only by that frame's own
identity -- never accumulated across the whole run. This is also what makes
`detect` resumable: re-running it skips any frame already recorded in
`manifest.<ext>` with a matching content fingerprint, and reprocessing a
frame (`--force`, or because its file changed) overwrites exactly that
frame's own output rather than appending a duplicate. A run interrupted
partway through can simply be re-invoked with the same arguments to pick up
where it left off.
