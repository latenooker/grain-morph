# Morphometric calculations — reference

> **Status: work in progress, subject to revision.** This is one part of a
> broader documentation set still being written. Companion documents to come
> cover the **grain-delineation pipeline** (flat-fielding → thresholding →
> connected components → subpixel marching-squares boundary → polygon repair),
> the **QC/focus gate**, and **aggregation**. Formulas, provenance, and the
> known-issue notes here may change as those land and as the flagged bugs (esp.
> `curvature_entropy`) are resolved.

Every per-grain quantity `grain-morph` persists, its exact calculation, and
whether the arithmetic is **custom** (written in this repo) or **imported**
(delegated to a third-party library). Written to make the numbers auditable and
to make clear where a result depends on an external implementation we do not
control.

Sources: `src/grain_morph/measure.py` (size, shape, spectral, boundary) and
`src/grain_morph/qc.py` (focus/contrast + the QC-side geometry). Assembly into
one row: `pipeline._build_row`.

## Conventions

- **Coordinates:** polygons are in pixel space, `x` = column, `y` = row. The
  boundary polygon is the subpixel marching-squares contour from
  `detect.detect_objects` (`Detection.polygon`); the *mask* is the integer
  label raster for the same object.
- **Units:** raw geometry is in **pixels**; physical columns (`*_um`, `*_um2`)
  multiply by `um_per_px` (per camera). `um_per_px` is currently a **placeholder
  calibration** (see `followups.md`) — every `*_um`/`*_um2` value is
  placeholder-scaled; all pixel-native, ratio, and shape metrics are
  calibration-independent.
- **Two boundaries, deliberately:** most size/shape metrics use the **subpixel
  polygon** (shapely); a few ellipse-fit quantities have no shapely equivalent
  and use `skimage.measure.regionprops` on a **rasterization** of that polygon
  (`_rasterize_polygon`). Where both a polygon- and a raster-derived version of
  the same quantity exist, both are persisted under distinct names (see
  [Dual geometry](#dual-geometry-solidity--aspect-ratio)).

## Provenance at a glance

| Library | Version role | What it computes here |
|---|---|---|
| `shapely` | imported | polygon `area`, `length`, `convex_hull` (area+length), `interpolate` (arc-length resampling) |
| `skimage.measure.regionprops` | imported | ellipse fit (`axis_major/minor_length`, `eccentricity`, `orientation`), `extent`, raster `perimeter`, `perimeter_crofton`, raster `solidity` |
| `skimage.draw.polygon`, `skimage.measure.label` | imported | rasterize polygon → mask; connected-component labeling |
| `pyefd` | imported | elliptic Fourier descriptor coefficients (Kuhl & Giardina 1982) |
| `wadell_rs` | imported | Wadell (1932) roundness & sphericity |
| `scipy.ndimage` | imported | binary morphology (dilate/erode), `map_coordinates` (profile sampling), `gaussian_filter1d` |
| `skimage.filters.sobel` | imported | edge gradient magnitude |
| `edt` | imported | Euclidean distance transform (input to `wadell_rs`) |
| **this repo** | **custom** | ECD, circularity, solidity, convexity, Feret max/min, curvature entropy, edge width, contrast, cumulative-power harmonic index, all unit conversions and ratio metrics |

---

## Size metrics (`measure_polygon`)

| Column | Calculation | Custom / Imported |
|---|---|---|
| `area_px` | `poly.area` — subpixel polygon area | imported (shapely) |
| `area_um2` | `area_px · um_per_px²` | custom |
| `ecd_um` | Equivalent circular diameter: `2·√(area_um2 / π)` | custom |
| `ecd_px` (in `qc_metrics`) | `2·√(poly.area / π)` | custom |
| `perimeter_um` | `poly.length · um_per_px` (shapely `length` = sum of all ring lengths) | imported + custom scale |
| `feret_max_um` | Max pairwise distance between convex-hull vertices, `·um_per_px`. Brute-force `O(N²)` over hull vertices (`_feret_max_px`) | custom (hull from shapely) |
| `feret_min_um` | Min hull width via rotating calipers: for each hull edge, project all vertices onto the edge normal, width = span; take the min over edges (`_feret_min_px`) | custom |
| `major_axis_um` / `minor_axis_um` | `regionprops.axis_major_length` / `axis_minor_length` of the equal-second-moment ellipse, on the rasterized polygon, `·um_per_px` | imported (skimage) |
| `perimeter_crofton_px` | `regionprops.perimeter_crofton` — Crofton raster perimeter (accurate diagnostic) | imported |

`ecd_um` and `ecd_px` are **exactly consistent**: both derive from `poly.area`,
so `ecd_um = ecd_px · um_per_px` (verified to 1e-13 across all grains).

**Note:** `circularity` uses the subpixel polygon perimeter, which carries a
small (~0.3%) marching-squares over-length; this biases `circularity` slightly
low (and is why `raster_perimeter_px`, a coarse Freeman chain code that
*over*-estimates, is kept only as a raster-inferiority baseline, not used in any
metric).

## First-order shape metrics (`measure_polygon`)

| Column | Calculation | Custom / Imported |
|---|---|---|
| `circularity` | `4π · area_px / perimeter_px²` (Cox / isoperimetric quotient; 1 = circle) | custom |
| `solidity` | `area_px / convex_hull.area` (both subpixel polygon) | custom (hull from shapely) |
| `convexity` | `convex_hull.length / perimeter_px` | custom (hull from shapely) |
| `aspect_ratio` | `major_axis / minor_axis` (ratio of the regionprops ellipse axes) | custom ratio of imported values |
| `extent` | `regionprops.extent` = object area / bounding-box area | imported |
| `eccentricity` | `regionprops.eccentricity` of the equal-second-moment ellipse | imported |
| `orientation` | `regionprops.orientation` (radians, major-axis angle) | imported |

Verified physical bounds across all grains: `circularity ∈ [0.02, 0.98]`,
`convexity ∈ [0.51, 1.00]`, `feret_min ≤ feret_max`, `minor ≤ major` (no
violations).

## Spectral & boundary-shape metrics

| Column | Calculation | Custom / Imported |
|---|---|---|
| `efd_{h}_{a,b,c,d}` (h = 1..order) | `pyefd.elliptic_fourier_descriptors(points, order, normalize=True)` on the arc-length-resampled boundary. `normalize=True` makes coefficients invariant to rotation, start point, and scale (so they complement, not duplicate, the size columns) | imported (pyefd); custom resample + normalize choice |
| `fourier_power_cum_90` | Smallest 1-indexed harmonic `h` at which `cumsum(power)/total ≥ 0.90`, where `power_h = a²+b²+c²+d²` | custom |
| `wadell_roundness` | `wadell_rs.roundness.calculate_roundness(...)`; `cfg.measure.wadell_smoothing → alpha_ratio`, other knobs fixed (`_WADELL_*`) | imported (wadell_rs) |
| `wadell_sphericity` | `wadell_rs.sphericity.calculate_sphericity(method="area")` | imported (wadell_rs) |
| `curvature_entropy` | Normalized Shannon entropy of the boundary's signed-curvature histogram — see below | **custom** |

### `curvature_entropy` in full (`measure_curvature_entropy`)

1. Resample the exterior ring to `curvature_resample_n` arc-length-even points.
2. Smooth the wrapped `x`, `y` sequences: `gaussian_filter1d(..., sigma, mode="wrap")`.
3. Signed curvature per point: `k = (x'·y'' − y'·x'') / (x'² + y'²)^{3/2}`
   (denominator guarded against 0), with derivatives from `np.gradient`.
4. Histogram `k` into `bins` bins, normalize to a probability vector `p`.
5. `curvature_entropy = −Σ pᵢ·ln pᵢ / ln(bins)` over non-zero `pᵢ` → `[0, 1]`.

⚠️ **Two caveats — see [Known issues](#known-issues--gotchas):** the histogram
range is each grain's *own* min–max (step 4), which makes the metric behave as a
boundary-**regularity** score (high = smooth/round, low = angular/filamentary —
opposite to the current docstring wording) and outlier-sensitive; and
`np.gradient` (step 3) is not truly periodic at the ring seam.

## Focus / contrast metrics (`qc.py`, on the raw un-flat-fielded frame)

| Column | Calculation | Custom / Imported |
|---|---|---|
| `edge_width_px` | Median, over boundary-normal intensity profiles, of the distance between the 10% and 90% levels of the `core_mean → bg_mean` rise. Profiles sampled with `scipy.ndimage.map_coordinates` (bilinear) along outward normals; 10/90 crossings via `np.interp` (`_edge_width_px`) | custom (sampling imported) |
| `contrast` | `(bg_mean − core_mean) / bg_mean`, where `bg_mean` = mean raw intensity in a dilated ring just outside the mask, `core_mean` = mean in the eroded interior | custom (morphology imported) |
| `edge_gradient` | Mean `skimage.filters.sobel` magnitude in a `±_EDGE_BAND_PX` band around the boundary | imported (sobel); custom band |
| `edge_gradient_norm` | `edge_gradient / local_bg_mean` | custom |
| `qc_solidity`, `qc_aspect_ratio`, `ecd_px` | `regionprops` on the local raster mask (`_mask_geometry`); `ecd_px = 2·√(area/π)` | imported + custom ECD |

`touches_border` and `has_polygon` are booleans, not measurements.

---

## Dual geometry (solidity & aspect ratio)

`solidity`/`aspect_ratio` are persisted **twice**, from two different geometries,
under different names (`_build_row:410–421`):

- **`solidity` / `aspect_ratio`** — from `measure_polygon`, on the **subpixel
  polygon** (solidity = polygon area / polygon convex-hull area). These are the
  reported shape descriptors.
- **`qc_solidity` / `qc_aspect_ratio`** — from `qc_metrics`, on the **raster
  mask** via `regionprops`. **These are what `qc_flags` actually thresholds on.**

They differ by up to **0.17** (mean 0.02) across grains. **Consequence:**
`flag_possible_agglomerate` fires on `qc_solidity < agglomerate_solidity_max`,
*not* on the reported `solidity` column — so filtering the output table on
`solidity` will **not** reproduce the flag. Use `qc_solidity` to reproduce QC
decisions.

## Known issues & gotchas

1. **`curvature_entropy` direction is inverted vs. its docstring.** The
   implementation scores *smooth/round* grains **high** (measured
   `corr(curvature_entropy, circularity) = +0.77`) and *angular/filamentary*
   grains **low** — the reverse of the docstring's "smooth → low". Root cause:
   `np.histogram` uses each grain's own curvature min–max as the range, so a few
   sharp-corner spikes collapse everything else into one bin. Net: it is a
   *boundary-regularity* score, outlier-sensitive and per-grain-relative, not an
   absolute roughness measure. (The `flag_debris` proposal in `followups.md`
   relies on this current behavior — low CE = fiber/debris — so changing the
   computation would require re-validating that.)
2. **`curvature_entropy` gradient is not periodic** despite the docstring:
   `np.gradient` uses one-sided differences at the ring seam, so 1–2 of
   `resample_n` points have slightly wrong curvature. Small effect.
3. **Dual solidity/aspect ratio** — see above; the flag and the reported column
   are different numbers.
4. **`*_um` columns are placeholder-scaled** until real `um_per_px` lands; the
   defocus/size QC cuts and all shape metrics are pixel- or ratio-based and are
   unaffected.
5. **Feret helpers reinvent library functions.** `_feret_max_px` (brute-force
   `O(N²)`) duplicates `skimage.measure.regionprops.feret_diameter_max`, and
   `_feret_min_px` (rotating calipers) duplicates the min width from
   `shapely.minimum_rotated_rectangle`. Both are correct, but per the project's
   "assemble, don't reinvent" rule they should be replaced with the library
   calls. (These are the only two morphometric *algorithms* written from scratch
   where a maintained equivalent already exists; the other custom entries are
   either one-line standard formulas or genuinely bespoke QC metrics.)

See `followups.md` for the QC-tuning and performance follow-ups
(per-camera defocus/size gates, `flag_debris`, disk-bound processing).
