# grain-morph — design spec (v0.1)

Date: 2026-08-04
Status: **draft, pending user review**

## 1. Purpose

A standalone, classical-computer-vision, command-line pipeline that turns
directories of backlit silhouette images (from a Microtrac CAMSIZER X2 dynamic
image analyzer, or any similar transmitted-light instrument) into a **per-grain
table** of size and shape descriptors with **QC flags**, plus a small set of
**QC review artifacts** and the **subpixel grain-boundary geometry** for
re-analysis.

The value is not segmentation (trivial for backlit silhouettes) but: careful
flat-fielding, subpixel boundary extraction, a defensible focus/QC gate, and
persisting geometry so morphometry can be re-derived without re-segmenting tens
of GB of frames.

**Assemble from maintained packages. Write glue, config, CLI, and tests — not
algorithms.** No deep learning, no training, no GPU.

This package is the image-based sibling of `camsizer-io` (which reads the
instrument's *processed* CSV / X-Plorer outputs). The two are **decoupled** —
no shared code, no import dependency.

## 2. Input data — measured facts

Profiled from real frames on `/Volumes/LEXAR/Camsizer/PPX/images_P_01_cs` and
`images_P_17_cs`:

- **Format:** 8-bit grayscale Windows BMP, **2048 rows × 2040 cols**, ~4.2 MB
  each. Dispatch on extension via `imageio` — do not hard-code BMP (runs may use
  PNG/TIFF).
- **Optics:** transmitted-light silhouettes. Grains are near-opaque dark objects
  (interior intensity ~8–25) on a bright background (~125–210).
- **Illumination is NOT uniform.** Background varies ~125→208 within a frame
  (vignetting/gradient). Flat-field correction is **mandatory**.
- **Two cameras**, both exported: `_b_` = **CCD Basic** (coarse), `_z_` =
  **CCD Zoom** (fine). They have **different µm/px scales** and overlapping size
  ranges. `camera` is a first-class dimension.
- **Blank/background frames exist** per (sample, run, camera):
  `..._{cam}_back.bmp` (e.g. `P_17_cs_002_b_back.bmp`). These are cleaner and
  brighter (higher floor) — ideal for **direct** flat-fielding by division.
- **Sparse loading:** ~3–5 particles/frame in samples; must handle 0 to ~200
  without special-casing.
- **The BMP resolution header is meaningless** (print-DPI field). Scale comes
  from config; fail loudly if the camera in a frame has no calibration.
- **Large runs:** thousands of frames per run, tens of GB. Stream and
  parallelize; never load a whole run into memory.

### Filename convention (this export)

```
P_01_cs_003_b_0004258.bmp    sample tokens | run | camera(b/z) | frame index
P_17_cs_002_z_back.bmp       ...            | run | camera(z)   | "back" = blank
```

The **instrument tokens** — camera (`_b_`/`_z_`), the `_back` blank suffix, and
the trailing numeric **frame index** — are fixed exporter convention (still
configurable, but with a working default). Everything the sample-regex captures
from the *remainder* is the `sample_id`. **Do not hard-code `P_{NN}_cs`** — the
sample naming is user-defined and varies between studies.

### Empirical good/bad separation (2 frames, raw 8-bit; sanity targets, NOT thresholds)

| Object | ECD px | Solidity | Mean edge grad | Core/bg contrast | Truth |
|---|---|---|---|---|---|
| A | 193 | 0.937 | 19.5 | 0.937 | in focus |
| B | 136 | 0.924 | 32.8 | 0.937 | in focus |
| C | 70 | 0.942 | 27.7 | 0.940 | in focus |
| D | 60 | 0.962 | 30.1 | 0.946 | border-clipped |
| E | 18 | 0.902 | 19.0 | 0.725 | debris, AR 4.2 |
| F | 188 | 0.956 | 29.6 | 0.938 | in focus |
| G | 96 | **0.975** | **7.2** | **0.853** | **defocused** |
| H | 27 | 0.877 | 17.2 | 0.770 | sliver, AR 6.4 |

Defocused grain G has the *highest* solidity — solidity alone won't catch it.
The discriminating signals are **edge sharpness** and **contrast**. In-focus
contrast clusters 0.937–0.946; defocused drops to 0.853.

### Non-grain objects to catch

1. **Defocused particles** (finite depth of field): soft ramped edges, inflated
   area, upward-biased roundness. The single most important thing to catch.
2. **Border-truncated** particles clipped by the frame edge.
3. **Debris / fibers / fines**: small, elongated, low-contrast.
4. **Agglomerates** (touching grains): flag only in v1.

## 3. Decisions (where this differs from the original build prompt)

1. **Flat-field default = blank division.** Auto-discover `..._{cam}_back.bmp`
   and pair it to every frame of that `(sample, run, camera)`; divide and
   normalize. Morphological-opening estimate (large kernel, ~201 px) is the
   **fallback** when no blank exists. Config: `flatfield.method ∈
   {blank, morphological, auto}`, default `auto`. Log which was used per frame.
2. **Camera is first-class.** `camera ∈ {basic, zoom}` is its own column;
   `um_per_px` looked up **per camera** from config; **fail loudly** if a
   frame's camera has no calibration entry. Aggregation groups by
   `(sample_id, camera)`; **no cross-camera distribution stitching in v1** —
   Basic and Zoom are reported separately (stitching is downstream).
3. **sample_id from a configurable capture regex** applied after stripping
   instrument tokens. Permissive default; no `P_{NN}_cs` assumption.
4. **Threshold = half-maximum by default** (midpoint between local background
   and opaque core), Otsu selectable via `threshold.method`. Rationale: for a
   ramped edge the 50 % level is the least-biased estimator of the true
   geometric boundary and is the optical-sizing convention.
5. **Subpixel polygons persisted ON by default.** The boundary of record is the
   subpixel contour, not a raster mask (mask perimeters are quantized and
   over-estimate by up to 4/π, biasing every circularity/roundness metric).
   Stored in a **separate geometry Parquet keyed on `grain_uid`**: exterior ring
   + holes as WKT in **pixel coordinates**, with `um_per_px` alongside so it is
   self-describing. Masks are rasterized from the polygon on demand; never
   stored. This enables re-analysis without re-segmentation.
6. **Focus gate:** `edge_width_px` (10–90 % intensity rise along boundary
   normals) is the **primary**, scale-free criterion; `edge_gradient_norm` +
   `contrast` are the cross-check/fallback. Store both continuous metrics and
   booleans for every object.

## 4. Architecture

Two decoupled stages — segmentation is the expensive step and must never be
re-run to re-measure.

```
src/grain_morph/
  __init__.py
  config.py     # pydantic models, YAML load/validate, config_hash
  io.py         # frame discovery, filename parse, blank pairing, manifest, resume
  flatfield.py  # blank division | morphological estimate
  detect.py     # threshold, label, fill holes, subpixel contour -> shapely.Polygon
  measure.py    # polygon + raster morphometry, pyefd, wadell_rs
  qc.py         # per-object QC metrics + flag logic
  aggregate.py  # per-grain table -> per (sample_id, camera) summaries
  report.py     # QC contact sheets, focus scatter, rejection-vs-ECD, field render
  cli.py        # typer: detect | aggregate | report
tests/
  synth.py      # synthetic frame generator (ground truth)
  test_*.py
configs/
  default.yaml
README.md
pyproject.toml  # + uv lockfile
```

Target **< 1200 LOC** source (excluding tests). A module past ~250 LOC is a
signal something a dependency already does is being reimplemented.

### Stage 1 — `grain-morph detect`

Input: a directory/glob of frames. Output: per-grain Parquet (chunked; one row
per detected object) + geometry Parquet + run artifacts.

Per frame: read `uint8` grayscale → flat-field (blank/morphological) → threshold
(half-max default) → label + fill holes + drop `< min_area_px` → subpixel
`find_contours` on the **flat-fielded grayscale** at the threshold level →
`shapely.Polygon` (largest exterior ring; interior rings = holes; associate to
labels by centroid/point-in-polygon) → measure (§5) → QC metrics + flags (§6) →
emit rows + geometry.

- Parallelize across frames with `joblib.Parallel` (process-based, one frame per
  task; frames independent).
- **Resume:** manifest of processed frames (path + mtime + size hash). Skip
  completed unless `--force`.
- Empty frames (0 objects): write no grain rows, still record in manifest.
- Never fail the whole run on one bad frame: catch per-frame exceptions, record
  in `errors.parquet`, continue.
- Blank (`_back`) frames are consumed as flat-field references, never processed
  as data.

### Stage 2 — `grain-morph aggregate`

Reads per-grain Parquet, applies QC flags as **configurable** filters, emits per
`(sample_id, camera)` summaries: N accepted, N rejected by each reason,
D10/D50/D90 of ECD and of Feret-min (sieve-analogous axis), mean/SD of each
shape descriptor, and the accepted-grain table joined to sample metadata.
**Rejection rate is reported binned by ECD** so a size-dependent focus gate is
visible, not silent.

### `grain-morph report`

QC artifacts (§7).

## 5. Morphometry (polygon space via shapely; raster fallback flagged)

- **Size:** `area_px`, `area_um2`, `ecd_um`, `feret_max_um`, `feret_min_um`
  (rotating calipers / min rotated rectangle; report alongside ECD),
  `major_axis_um`, `minor_axis_um` (fitted ellipse), `perimeter_um` (polygon) +
  `perimeter_crofton_px` (regionprops, for discrepancy diagnostics).
- **First-order shape:** `aspect_ratio` (major/minor), `solidity`, `convexity`
  (hull perim / perim), `circularity` (4πA/P², documented resolution-sensitive),
  `extent`, `eccentricity`, `orientation`.
- **Roundness/sphericity:** `wadell_roundness`, `wadell_sphericity` via
  `wadell_rs` (fallback `PaPieta/fast_rs`, noted in README). `wadell_smoothing`
  recorded as a column on every row.
- **Outline harmonics:** normalized EFD via `pyefd` (order default 15), contour
  resampled to fixed 256 points; stored as `efd_*` columns. Plus
  `fourier_power_cum_90` diagnostic.
- **Geometry:** subpixel contour persisted to `contours.parquet` keyed on
  `grain_uid` (WKT, pixel coords, `um_per_px`). ON by default (§3.5).

## 6. QC metrics and flags

Store **both** continuous metric and boolean flag per object:

| Metric | Definition | Flag |
|---|---|---|
| `edge_width_px` | 10–90 % rise along boundary normals | `flag_defocus` (primary) |
| `edge_gradient` | mean Sobel over ±3 px band, raw image | — |
| `edge_gradient_norm` | `edge_gradient` / local background | `flag_defocus` (cross-check) |
| `contrast` | (local bg − eroded-core mean) / local bg | `flag_defocus` (cross-check) |
| touches border | bbox intersects frame edge | `flag_border` |
| `ecd_px` | — | `flag_too_small` (`< min_ecd_px`) |
| `aspect_ratio` | — | `flag_sliver` (with small size + low contrast) |
| solidity, convex defect count/depth | — | `flag_possible_agglomerate` |
| contour status | — | `flag_no_polygon` |

`qc_pass` = none of the disqualifying flags set (which flags disqualify is
configurable). Normalize gradient by local background (illumination not uniform).

**Agglomerates:** flag only in v1. `--split-agglomerates` opt-in runs
distance-transform + watershed on flagged objects and marks products with
`split_from_uid`. Default off.

## 7. Outputs

`grain_uid` is deterministic: `f"{frame_id}:{label}"` (frame_id = filename stem).

**Output format is configurable.** `output.format` selects the tabular writer
for the per-grain table, geometry table, and aggregates: **`parquet` (default)**,
`csv`, or `feather`. Parquet is recommended (columnar, partitioned, typed);
`csv` is the portable/inspectable fallback; `feather` for fast round-trips. A
single `write_table(df, path, fmt)` helper centralizes this so no module hard-
codes a format. `sample_id`/`camera` **partitioning applies to Parquet only**;
for `csv`/`feather` the same split is expressed as one file per partition
(`{sample_id}__{camera}.csv`). Geometry WKT survives all three formats. Run
artifacts stay fixed: `run_config.yaml` (YAML), `summary.json` (JSON);
`manifest`/`errors` follow `output.format`.

**Per-grain table** (partition by `sample_id`, then `camera`), one row per
detected object, always:

```
grain_uid  frame_id  frame_path  sample_id  camera  run  label
centroid_x  centroid_y  um_per_px
<all §5 measurements>  <all §6 metrics + flags>  qc_pass
wadell_smoothing  pipeline_version  config_hash
```

**Geometry Parquet** `contours.parquet` keyed on `grain_uid` (WKT exterior +
holes, pixel coords, `um_per_px`).

**Run artifacts:** `manifest.parquet` (every frame: status, object count,
timing, flatfield method used), `errors.parquet` (per-frame failure +
traceback), `run_config.yaml` (fully resolved config), `summary.json` (counts,
rejection breakdown, wall time).

**QC review artifacts (`report`):**
1. **Contact sheets** — grain crops tiled with `grain_uid` + key metrics: (a) all
   rejected, grouped by flag; (b) stratified random sample of accepted across the
   size range. Cap N per sheet, paginate. Most valuable trust-building output.
2. **Focus scatter** — `edge_gradient_norm` vs `contrast`, colored by flag, with
   threshold lines. In-focus grains should cluster; gate should visibly separate.
3. **Rejection rate vs ECD bin** bar chart.
4. **Illumination field** rendered for one frame per run (vignetting drift).

## 8. Testing

Three test tiers, by purpose:

1. **Synthetic (precision, CI).** `tests/synth.py` generates frames with known
   ground truth (polygons of known area/perimeter/AR on a bright background;
   known multiplicative gradient; configurable Gaussian blur for defocus;
   edge-overlapping polygons; Poisson/Gaussian noise). All numeric **acceptance
   criteria below run here** — only synthetic data has exact ground truth, and
   only full-resolution edges support sub-1 % area / focus-recall assertions.
2. **Downsampled real (integration, CI, committed).** A small set of real frames
   downsampled (anti-aliased, integer factor, target ≲ 300 KB each) committed
   under `tests/fixtures/real/`: at least one **Basic** + one **Zoom** data
   frame plus their `_{cam}_back.bmp` blanks. These drive **integration/smoke**
   tests — filename parsing, blank discovery + pairing, per-camera dispatch,
   end-to-end `detect → aggregate → report`, correct schema/partitioning, and
   "grains actually detected". A `tests/fixtures/real/GENERATION.md` records the
   source path, frame indices, and downsample factor; fixture config scales
   `um_per_px` by that factor. **Downsampled frames are NOT used for precision
   or focus-gate assertions** (downsampling degrades the edges those measure).
3. **Full-res real (opt-in, local only).** Tests that run against the frames on
   `/Volumes/LEXAR` and their `_back` blanks **when present locally** — the only
   place the real defocus separation (e.g. the profiled grain G) is validated.
   Skipped in CI.

**Full-resolution BMPs are NOT committed** (multi-MB, tens of GB per run); only
the downsampled integration fixtures are.

Acceptance criteria:

1. Zero blur, no gradient: `area_um2` within **1 %**, `perimeter_um` within
   **2 %** of ground truth.
2. 40 % multiplicative gradient: identical polygon in bright vs dark corner
   agrees within **2 %** (the flat-field test — fails loudly if flat-fielding
   breaks).
3. Blur σ ≥ 4 px flagged `flag_defocus` with recall ≥ **0.95**; FP rate on
   unblurred ≤ **0.02**.
4. All edge-intersecting polygons flagged `flag_border`; none missed.
5. `perimeter_um` (polygon) measurably lower than raster perimeter and closer to
   ground truth — asserted explicitly (guards against regression to mask-based).
6. Zero-particle frame → zero rows, one manifest entry, no exception.
7. `--resume` on a completed directory does no work, byte-identical output.
8. 500 synthetic frames process without memory growth (peak-RSS bound asserted).

## 9. Stack & style

Core (pip-installable, permissive): `numpy scipy scikit-image pandas pyarrow
shapely pyefd wadell_rs imageio typer pydantic PyYAML joblib tqdm matplotlib
pytest`. `wadell_rs` fallback: `PaPieta/fast_rs` (note substitution in README).
`pyproject.toml` + `uv`, pinned lockfile.

- Python ≥ 3.11, type hints throughout, `from __future__ import annotations`,
  `ruff` + `mypy` clean.
- Single YAML config, pydantic-validated, **every threshold exposed, no magic
  numbers in source**. Ship `configs/default.yaml` with §2 starting values.
- `logging` at INFO; one line per frame at DEBUG.
- README: install, 3-command quickstart, config reference table, and a "how the
  QC gate works and how to tune it" section for a non-author reader.

## 10. Non-goals (v1)

No GUI/web/napari. No deep learning / training / GPU. No database (Parquet on
disk). No custom watershed / Fourier / roundness (use libraries). No
sieve-equivalence or mass-conversion modelling. No cross-camera stitching.
**Flag, never drop** — every detected object gets a row; rejection is a
reversible filter at aggregation, never a deletion at detection.

## 11. v0.2 addendum (2026-08-04) — curvature entropy + parallelization

Two requirements added mid-implementation. Both are recorded here as decisions.

### 11.1 Curvature entropy (new morphometric descriptor)

**What:** a scale-free measure of boundary complexity/roughness — the Shannon
entropy of the distribution of local boundary curvature. A smooth, regular
outline concentrates curvature in few bins (low entropy); a rough, irregular,
or crenulated outline spreads curvature across many bins (high entropy). It
complements the EFD harmonics (frequency-domain) and Wadell roundness
(corner-scale) with a single distribution-shape scalar, and is cheap.

**Method (decided):**
1. Take the grain's subpixel exterior ring (the same polygon used everywhere
   else) and resample it to `measure.curvature_resample_n` evenly-spaced points
   (closed curve).
2. Gaussian-smooth the closed coordinate sequences with
   `measure.curvature_smoothing` (sigma in resampled-point units, `mode="wrap"`)
   to suppress pixel-quantization noise before differentiating — curvature is a
   second-derivative quantity and is noise-sensitive; the smoothing scale is a
   free parameter and **travels with the data as a column**, like
   `wadell_smoothing`.
3. Signed curvature κ = (x'y'' − y'x'') / (x'² + y'²)^{3/2} via periodic finite
   differences (`np.gradient` on the wrapped arrays).
4. Histogram κ into `measure.curvature_bins` bins; normalize to a probability
   vector p; `curvature_entropy = −Σ p_i ln p_i / ln(bins)` (normalized to
   [0, 1] so it is comparable across bin counts). Empty/degenerate boundaries →
   `NaN`, with `flag_no_polygon` already covering the no-polygon case.

**Columns:** `curvature_entropy` (float, per grain) and `curvature_smoothing`
(float, the sigma used — travels with the data). Config section under `measure`:
`curvature_resample_n` (default 256), `curvature_smoothing` (default 2.0),
`curvature_bins` (default 32). Lives in `measure.py` as
`measure_curvature_entropy(poly, resample_n, smoothing, bins) -> dict`.

### 11.2 Parallelization (across images)

The architecture is already frame-parallel; this makes it explicit and
optimized, and sets the standard the whole codebase already meets.

**Decisions:**
- **Unit of parallelism = one frame.** Frames are fully independent (no shared
  state); `process_frame(spec, cfg, ...)` is a pure function returning a small,
  picklable `FrameResult` (lists of dicts + WKT strings). This is already true
  of every module written (config/io/flatfield/detect/measure/qc are pure
  per-frame functions with no global mutable state), so no rewrite of Tasks
  1–11 is needed — the requirement is satisfied by construction and enforced at
  the pipeline layer.
- **Never ship images across the process boundary.** Workers receive a
  `FrameSpec` (paths only), read their own frame + blank from disk, do all
  compute, and return only small row/geometry dicts. No large array crosses the
  loky pickle boundary.
- **Process-based joblib** (`backend="loky"`) over frames. `n_jobs` is
  configurable (`runtime.n_jobs`, default `-1` = all cores) and overridable via
  the CLI `--jobs`.
- **Streaming, memory-bounded.** Consume results as they complete
  (`joblib.Parallel(return_as="generator")`) and write rows incrementally in
  chunks (`runtime.chunk_size`) rather than materializing a whole run in RAM —
  this is what keeps peak RSS bounded (acceptance criterion 8) at tens-of-GB
  run scale.
- **No thread oversubscription.** With N worker processes each calling
  numpy/scipy/skimage (which spawn their own BLAS/OpenMP threads), cap inner
  threads to 1 per worker via `joblib.parallel_config(inner_max_num_threads=1)`
  so N processes don't each spawn N threads. No new dependency.
- The synthetic generator's supersampling (test-only) is not on the production
  parallel path; its per-object full-frame supersample stays a test concern.
