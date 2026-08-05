"""Foreground detection: threshold level, labeling, subpixel contours.

Objects in a flat-fielded frame are dark silhouettes on a background near
1.0 (see :mod:`grain_morph.flatfield`). This module picks a threshold level
separating object from background, labels connected foreground components,
and — separately — extracts *subpixel* boundary polygons for each label via
`skimage.measure.find_contours` at that same level. The two are combined by
associating each contour to the label whose centroid falls inside it, which
is what lets downstream morphometry (Task 9+) work from a smooth polygon
rather than the blocky pixel mask.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shapely
from scipy import ndimage
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from skimage.filters import threshold_otsu
from skimage.measure import find_contours, label, regionprops

from grain_morph.config import Config


@dataclass
class Detection:
    """A single detected object: its pixel mask summary and subpixel polygon.

    Attributes:
        label: Connected-component label id in the returned label image.
        polygon: Subpixel boundary polygon (exterior ring plus any interior
            holes), or `None` if no contour could be associated with this
            label, or if the associated contour was invalid and
            unrecoverable (see `_repair_polygon`). Always a valid,
            positive-area `Polygon` when not `None`.
        mask_bbox: Pixel bounding box `(min_row, min_col, max_row, max_col)`
            from `skimage.measure.regionprops`, `max_row`/`max_col`
            exclusive.
        centroid_xy: Pixel-mask centroid as `(x, y)` = `(col, row)`.
        contour_ok: `True` if `polygon` was successfully recovered (and,
            if necessary, repaired into a valid polygon).
    """

    label: int
    polygon: Polygon | None
    mask_bbox: tuple[int, int, int, int]
    centroid_xy: tuple[float, float]
    contour_ok: bool


def threshold_level(corrected: np.ndarray, cfg: Config) -> float:
    """Compute the foreground/background intensity level for a flat-fielded frame.

    Objects are darker than the returned level (`corrected < level`).

    - `"half_max"` (default): defines the object-core pixel set *absolutely*
      as those darker than `1.0 - cfg.threshold.min_object_depth` (backlit
      grains are near-opaque, far below the 1.0 background), estimates the
      core intensity as a low percentile (`cfg.threshold.core_percentile`) of
      that set, and places the level a fraction
      (`cfg.threshold.half_max_fraction`) of the way from the core back up to
      background. An absolute opacity cut — rather than a relative Otsu split
      — is used deliberately: Otsu *always* partitions the histogram, so on a
      frame with no real objects it splits the sensor noise and plants the
      "core" just below background, and even with a few real grains among
      heavy noise the noise dominates the split; either way the level rides up
      into the noise and traces hundreds of specks. The absolute cut is also
      robust to residual illumination gradient (a few-percent background
      spread never reaches the opacity cut). If fewer than
      `cfg.detect.min_area_px` pixels are opaque, the frame is treated as
      object-free and a level below every pixel is returned (detects nothing).
    - `"otsu"`: `skimage.filters.threshold_otsu` on the raw intensities.

    Args:
        corrected: Flat-fielded frame, background ~1.0, objects darker.
        cfg: Resolved pipeline configuration; consumes `cfg.threshold`.

    Returns:
        The scalar threshold level, in the same units as `corrected`.

    Raises:
        ValueError: If `cfg.threshold.method` is not `"half_max"` or
            `"otsu"`.
    """
    if cfg.threshold.method == "half_max":
        # Genuinely-opaque object-core pixels, by an absolute cut (see the
        # docstring for why absolute rather than an Otsu split).
        opaque = corrected[corrected < 1.0 - cfg.threshold.min_object_depth]
        # Object-presence gate: fewer than one minimum object's worth of
        # opaque pixels -> treat the frame as object-free and return a level
        # below every pixel so `corrected < level` selects nothing.
        if opaque.size < cfg.detect.min_area_px:
            return float(corrected.min()) - 1.0
        core = float(np.percentile(opaque, cfg.threshold.core_percentile))
        return core + cfg.threshold.half_max_fraction * (1.0 - core)
    if cfg.threshold.method == "otsu":
        return float(threshold_otsu(corrected))
    raise ValueError(f"unknown threshold method: {cfg.threshold.method!r}")


def _contour_to_polygon(contour: np.ndarray) -> Polygon | None:
    """Convert a `find_contours` (row, col) ring into an (x, y) shapely polygon.

    Args:
        contour: `(N, 2)` array of `(row, col)` points from
            `skimage.measure.find_contours`.

    Returns:
        A shapely `Polygon` with `x = col`, `y = row`, or `None` if the ring
        has too few points to form a valid polygon.
    """
    if len(contour) < 3:
        return None
    xy = contour[:, ::-1]  # (row, col) -> (x, y) = (col, row)
    poly = Polygon(xy)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly if isinstance(poly, Polygon) and not poly.is_empty else None


def _assign_rings_to_labels(
    rings: list[Polygon], label_image: np.ndarray, region_labels: list[int]
) -> dict[int, list[Polygon]]:
    """Group contour rings by the label whose region they belong to.

    A ring is assigned to the label of the pixel at its centroid; if that
    pixel falls outside every region (e.g. a contour hugging the boundary
    of a thin object), the ring is assigned to the nearest labeled pixel
    instead.

    Args:
        rings: Candidate contour polygons (exterior or interior rings, not
            yet distinguished).
        label_image: Integer label image from `skimage.measure.label`.
        region_labels: The set of label ids present in `label_image`.

    Returns:
        Mapping from label id to the list of rings assigned to it.
    """
    assigned: dict[int, list[Polygon]] = {lbl: [] for lbl in region_labels}
    if not rings:
        return assigned

    # Precompute nearest-labeled-pixel lookup once, for rings whose centroid
    # lands on background (e.g. a ring hugging a hole or thin neck).
    background = label_image == 0
    _, nearest_idx = ndimage.distance_transform_edt(background, return_indices=True)

    for ring in rings:
        cx, cy = ring.centroid.x, ring.centroid.y
        row = int(round(cy))
        col = int(round(cx))
        row = min(max(row, 0), label_image.shape[0] - 1)
        col = min(max(col, 0), label_image.shape[1] - 1)
        lbl = int(label_image[row, col])
        if lbl == 0:
            nrow, ncol = nearest_idx[0, row, col], nearest_idx[1, row, col]
            lbl = int(label_image[nrow, ncol])
        if lbl in assigned:
            assigned[lbl].append(ring)
    return assigned


def _polygon_with_holes(rings: list[Polygon]) -> Polygon:
    """Combine a label's rings into one polygon: largest ring exterior, rest holes.

    Args:
        rings: Contour rings assigned to a single label (at least one).

    Returns:
        A `Polygon` whose exterior is the largest-area ring and whose
        interior rings are the remaining rings' boundaries.
    """
    rings_by_area = sorted(rings, key=lambda p: p.area, reverse=True)
    exterior = rings_by_area[0]
    holes = [list(r.exterior.coords) for r in rings_by_area[1:]]
    return Polygon(exterior.exterior.coords, holes=holes)


def _repair_polygon(poly: Polygon) -> Polygon | None:
    """Repair a self-intersecting or negative-area polygon via `make_valid`.

    On coarse/downsampled real frames, `_polygon_with_holes` can combine
    an exterior ring and its assigned interior rings into an invalid
    `Polygon` two ways: a self-intersecting exterior ring (from a coarse
    marching-squares trace), which drives `poly.area` to `0` or an
    incorrect value; or an interior "hole" ring that is mis-associated /
    oversized relative to its exterior, which drives `poly.area`
    (`exterior_area - hole_area`) negative. Both are confirmed failure
    modes on real CAMSIZER frames.

    `shapely.make_valid` repairs both, at the cost of returning a
    `Polygon`, `MultiPolygon`, or `GeometryCollection` depending on the
    input's topology. This extracts the largest-area `Polygon` component
    of that result -- confirmed, on the real failing frames, to recover
    the correct object outline. Already-valid, positive-area polygons are
    returned unchanged (same object) so callers that only measure valid
    input see no change in behavior.

    Args:
        poly: Candidate object polygon, possibly invalid.

    Returns:
        A valid, positive-area `Polygon` -- `poly` itself if it was
        already valid, otherwise the largest-area polygonal component of
        its repair -- or `None` if no component of the repair has
        positive area (the contour is unrecoverable).
    """
    if poly.is_valid and poly.area > 0.0:
        return poly

    repaired = shapely.make_valid(poly)
    if isinstance(repaired, Polygon):
        candidates = [repaired]
    elif isinstance(repaired, (MultiPolygon, GeometryCollection)):
        candidates = [g for g in repaired.geoms if isinstance(g, Polygon)]
    else:
        candidates = []

    if not candidates:
        return None
    largest = max(candidates, key=lambda p: p.area)
    return largest if largest.area > 0.0 else None


def detect_objects(corrected: np.ndarray, cfg: Config) -> tuple[np.ndarray, list[Detection]]:
    """Threshold, label, and extract subpixel contour polygons for objects.

    Foreground is `corrected < threshold_level(corrected, cfg)`. Interior
    holes are optionally filled (`cfg.detect.fill_holes`) before connected
    components are labeled; components smaller than `cfg.detect.min_area_px`
    are dropped. Independently, `skimage.measure.find_contours` traces
    subpixel iso-intensity rings at the same level; each ring is associated
    with the label whose pixel mask contains its centroid (falling back to
    the nearest labeled pixel), the largest ring per label becomes the
    polygon exterior, and any remaining rings for that label become holes.
    On coarse/downsampled frames this combined polygon can come out
    invalid (self-intersecting exterior, or a mis-associated/oversized
    hole); such polygons are repaired via `_repair_polygon` before being
    stored, so `Detection.polygon` is always either `None` or a valid,
    positive-area `Polygon`.

    Args:
        corrected: Flat-fielded frame, background ~1.0, objects darker.
        cfg: Resolved pipeline configuration; consumes `cfg.threshold` and
            `cfg.detect`.

    Returns:
        A `(label_image, detections)` tuple. `label_image` is the integer
        connected-component label image (0 = background) after small
        objects have been dropped. `detections` has one `Detection` per
        surviving label, in label order.
    """
    level = threshold_level(corrected, cfg)
    foreground = corrected < level
    if cfg.detect.fill_holes:
        foreground = ndimage.binary_fill_holes(foreground)

    label_image = label(foreground)
    regions = regionprops(label_image)
    keep_labels = [r.label for r in regions if r.area >= cfg.detect.min_area_px]

    if len(keep_labels) < len(regions):
        drop_mask = ~np.isin(label_image, keep_labels)
        label_image = label_image.copy()
        label_image[drop_mask] = 0
        label_image = label(label_image > 0)
        regions = regionprops(label_image)

    region_labels = [r.label for r in regions]
    contours = find_contours(corrected, level)
    rings = [poly for c in contours if (poly := _contour_to_polygon(c)) is not None]
    rings_by_label = _assign_rings_to_labels(rings, label_image, region_labels)

    detections: list[Detection] = []
    for region in regions:
        min_row, min_col, max_row, max_col = region.bbox
        centroid_row, centroid_col = region.centroid
        label_rings = rings_by_label.get(region.label, [])
        polygon = _polygon_with_holes(label_rings) if label_rings else None
        if polygon is not None:
            polygon = _repair_polygon(polygon)
        detections.append(
            Detection(
                label=region.label,
                polygon=polygon,
                mask_bbox=(min_row, min_col, max_row, max_col),
                centroid_xy=(centroid_col, centroid_row),
                contour_ok=polygon is not None,
            )
        )
    return label_image, detections
