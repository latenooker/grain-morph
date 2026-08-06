# QC & focus gate — reference

> **Status: work in progress, subject to revision.** Companion to
> `morphometrics.md` / `delineation.md`; same custom-vs-imported treatment.

How `grain-morph` decides which grains are reliable. Two steps, deliberately
separated: **metrics** (continuous, threshold-free, computed once at detection)
and **flags + gate** (thresholds applied to those metrics). Source:
`src/grain_morph/qc.py`. Defaults: `configs/default.yaml` → `qc:`.

## The key design decision: record at detection, gate at aggregation

`qc_flags` is computed and stored per grain at detection time, **but the
accept/reject decision is not baked in.** `qc_pass` is **recomputed** at
aggregation time from `cfg.qc.disqualifying_flags` (see `aggregation.md`). So a
stricter or looser gate is explored by re-running `aggregate` with a different
config — **no re-detection, no re-measurement.** The per-grain table always
carries the raw flags; the gate is a late, cheap, reversible filter.

## QC metrics (`qc_metrics`, on the raw un-flat-fielded frame)

Continuous, threshold-free. Focus/contrast formulas are detailed in
`morphometrics.md` → *Focus / contrast metrics*; summarized here with provenance:

| Metric | Meaning | Custom / Imported |
|---|---|---|
| `edge_width_px` | Median 10–90% intensity-rise distance along boundary normals (profiles via `scipy.ndimage.map_coordinates`) | custom (sampling imported) |
| `contrast` | `(bg_mean − core_mean)/bg_mean`, ring vs eroded interior | custom (morphology imported) |
| `edge_gradient` | Mean `skimage.filters.sobel` magnitude in a boundary band | imported (custom band) |
| `edge_gradient_norm` | `edge_gradient / local_bg_mean` | custom |
| `ecd_px`, `qc_aspect_ratio`, `qc_solidity` | raster `regionprops` geometry (`_mask_geometry`) | imported + custom ECD |
| `touches_border`, `has_polygon` | booleans | custom |

Sampling constants (`qc.py`): `_N_BOUNDARY_SAMPLES = 48` normals, step
`_NORMAL_STEP_PX = 0.5`, normal half-length `0.6·(half object size)` clamped to
`[4, 15]` px, contrast margin `5` px, edge band `±3` px, crop pad `20` px.

## Flags (`qc_flags`, applying `cfg.qc`)

Pure custom thresholding on the metrics above. Defaults from
`configs/default.yaml`:

| Flag | Rule | Default threshold | Disqualifying by default? |
|---|---|---|---|
| `flag_defocus` | `edge_width_px > defocus_edge_width_px` **or** `contrast < defocus_contrast_min` | 5.25 px; 0.70 | **yes** |
| `flag_border` | `touches_border` (bbox on frame edge) | — | **yes** |
| `flag_too_small` | `ecd_px < min_ecd_px` | 10 px | **yes** |
| `flag_sliver` | `aspect_ratio > sliver_aspect_ratio` **and** `ecd_px ≤ sliver_max_ecd_px` | 3.0; 40 px | **yes** |
| `flag_possible_agglomerate` | `solidity < agglomerate_solidity_max` | 0.90 | no (diagnostic only) |
| `flag_no_polygon` | contour extraction failed | — | **yes** |

`qc_pass = not any(disqualifying_flags set)`. Default `disqualifying_flags =
[flag_defocus, flag_border, flag_too_small, flag_sliver, flag_no_polygon]` —
note `flag_possible_agglomerate` is **recorded but not enforced**.

## Gotchas

- **Flags threshold on the raster geometry**, not the reported polygon columns:
  `flag_sliver`/`flag_possible_agglomerate` use `qc_aspect_ratio`/`qc_solidity`
  (from `_mask_geometry`), which the row persists as `qc_solidity`/
  `qc_aspect_ratio`. The reported `solidity`/`aspect_ratio` are the *polygon*
  versions and differ by up to ~0.17. **To reproduce a flag, use the `qc_*`
  column** (see `morphometrics.md` → Dual geometry).
- **`edge_width_px` is camera-scale-dependent** — the single global
  `defocus_edge_width_px` is basic-tuned and over-rejects sharp zoom grains; and
  zoom acceptability tracks *size*, not edge width. Per-camera defocus/size gates
  are proposed in `followups.md` (with hand-label evidence).
- **A low-`curvature_entropy` `flag_debris`** (border-guarded) is proposed in
  `followups.md` to catch fiber/organic debris that currently slips the gate.

## Provenance summary

Imported: `scipy.ndimage` (morphology, `map_coordinates`), `skimage.filters.sobel`,
`skimage.measure.regionprops`. Custom: the 10–90% rise logic, contrast ratio,
boundary-normal construction, and **all** flag thresholding (`qc_flags` is pure
config-driven comparisons — no library involved).
