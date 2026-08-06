# Grain delineation — reference

> **Status: work in progress, subject to revision.** Companion to
> `morphometrics.md`; same custom-vs-imported treatment. Covers how a raw frame
> becomes a set of delineated grains (flat-field → threshold → label →
> subpixel boundary → repair). QC and aggregation docs still to come.

"Delineation" is everything upstream of measurement: turning a backlit
silhouette frame into, per grain, an integer **label mask** and a **subpixel
boundary polygon**. Source: `src/grain_morph/flatfield.py` and
`src/grain_morph/detect.py`; driven per frame by `pipeline.process_frame`.

## What "delineated" produces

Each detected object carries two representations, used differently downstream:

- **Label mask** — integer connected-component raster (`label_image == label`).
  Pixel-precise but blocky. Source of the ellipse-fit and raster QC geometry.
- **Subpixel polygon** (`Detection.polygon`) — a marching-squares iso-intensity
  contour at the threshold level, as a shapely `Polygon` (with holes). Source of
  area/perimeter/Feret/EFD/curvature. Always either `None` or a valid,
  positive-area polygon.

The two are produced **independently** (mask from thresholding+labeling, polygon
from contour tracing) and then associated — see [ring→label](#4-ring-to-label-association).

## Provenance at a glance

| Stage | Algorithm | Custom / Imported |
|---|---|---|
| Flat-field (morphological) | `scipy.ndimage.grey_closing` | imported |
| Flat-field (blank) + normalize | divide by field, floor at 1.0, divide by median | custom glue |
| Threshold `otsu` | `skimage.filters.threshold_otsu` | imported |
| Threshold `half_max` | percentile core + fraction-to-background | **custom heuristic** |
| Fill holes | `scipy.ndimage.binary_fill_holes` | imported |
| Connected components | `skimage.measure.label` + `regionprops` | imported |
| Min-area filter | drop `region.area < min_area_px`, relabel | custom glue |
| Subpixel contours | `skimage.measure.find_contours` (marching squares) | imported |
| Contour → polygon | `shapely.Polygon` + `buffer(0)` | imported (custom wrap) |
| Ring → label association | `scipy.ndimage.distance_transform_edt(return_indices)` nearest-label | imported + custom glue |
| Polygon-with-holes assembly | largest ring = exterior, rest = holes (shapely) | custom glue |
| Repair | `shapely.make_valid` + pick largest valid polygon | imported + custom glue |

**Reading:** every non-trivial algorithm is imported; the custom code is the
`half_max` threshold heuristic plus glue (normalization, association, assembly,
repair orchestration).

## Pipeline, stage by stage

### 1. Flat-field correction (`flatfield.apply_flatfield`)

Illumination is non-uniform, so an estimated illumination field is divided out
before thresholding. Field estimator selected by `cfg.flatfield.method`:

- **`blank`** (or `auto` with a paired `_{cam}_back.bmp`): the measured blank
  frame is the field. Preferred.
- **`morphological`** (or `auto` with no blank): grey **closing**
  (`scipy.ndimage.grey_closing`, size `morph_kernel_px`) estimates the bright
  background by filling in dark compact objects smaller than the kernel.
  (Closing, not opening — closing removes dark features; opening would remove
  bright ones and leave the grains.)

Then (custom): `field = max(field, 1.0)` (the backlit background is bright, so a
near-zero field pixel is degenerate, not real — floored without a tunable
threshold), `corrected = image / field`, clipped ≥0, then divided by its median
so **background ≈ 1.0** and objects sit below 1.0.

### 2. Threshold level (`detect.threshold_level`)

- **`half_max`** (custom): take "opaque" pixels darker than
  `1 - min_object_depth`, estimate the object core as `core_percentile` of those,
  and put the level `half_max_fraction` of the way from the core back up to the
  background (1.0): `level = core + half_max_fraction·(1 − core)`. A domain
  heuristic for dark-object/bright-background silhouettes.
- **`otsu`** (imported): `skimage.filters.threshold_otsu`.

### 3. Foreground, labeling, size filter (`detect.detect_objects`)

`foreground = corrected < level`; optional `binary_fill_holes`
(`cfg.detect.fill_holes`). Connected components via `skimage.measure.label`;
`regionprops` gives per-label area/bbox/centroid. Components with
`area < min_area_px` are dropped and the image relabeled. (Imported algorithms;
the keep/drop/relabel is glue.)

### 4. Ring-to-label association (`_assign_rings_to_labels`)

`find_contours(corrected, level)` traces subpixel iso-level rings independently
of the label mask, so each ring must be matched to a label. Each ring's centroid
is looked up in the label image; if it lands on background, the **nearest labeled
pixel** is used via `scipy.ndimage.distance_transform_edt(background,
return_indices=True)`. Per label, the **largest ring becomes the exterior** and
any remaining rings become **holes** (`_polygon_with_holes`). (EDT imported; the
association/assembly is custom glue.)

### 5. Polygon repair (`_repair_polygon`)

A coarse marching-squares trace can self-intersect or produce a hole larger than
its exterior (negative/zero area). `shapely.make_valid` repairs it; if it returns
a collection, the **largest positive-area polygon** is kept. Guarantees
`Detection.polygon` is `None` or a valid, positive-area `Polygon` — so one bad
contour degrades a single grain's geometry, never the frame.

## Conventions & caveats

- **Contours are traced on the flat-fielded `corrected` image** at the same
  `level` used for the mask — so polygon and mask refer to the same threshold,
  but the polygon is subpixel while the mask is pixel-quantized.
- **Marching-squares over-length:** the subpixel contour follows per-pixel edge
  crossings and carries a small (~0.3%) perimeter over-estimate — this slightly
  biases `circularity` low (see `morphometrics.md`).
- **`half_max` assumptions:** dark-object-on-bright-background, objects opaque
  enough to clear `min_object_depth`. Sparse or very-low-contrast frames may want
  `otsu`.
- **Independent mask vs polygon** is why some metrics have both a polygon and a
  raster version (see `morphometrics.md` → Dual geometry).

## Custom-vs-imported, honestly

Delineation is **almost entirely imported algorithms + custom glue**, matching
the project rule ("assemble from maintained packages; write glue, config, CLI,
tests — not algorithms"). The only custom *algorithm* here is the `half_max`
threshold heuristic; everything else custom is orchestration. For the
measurement side's accounting — including two Feret helpers that reinvent
`skimage.regionprops.feret_diameter_max` / `shapely.minimum_rotated_rectangle`
and should be replaced — see `morphometrics.md`.
