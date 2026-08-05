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
from scipy import ndimage
from shapely.geometry import Polygon
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
            label.
        mask_bbox: Pixel bounding box `(min_row, min_col, max_row, max_col)`
            from `skimage.measure.regionprops`, `max_row`/`max_col`
            exclusive.
        centroid_xy: Pixel-mask centroid as `(x, y)` = `(col, row)`.
        contour_ok: `True` if `polygon` was successfully recovered.
    """

    label: int
    polygon: Polygon | None
    mask_bbox: tuple[int, int, int, int]
    centroid_xy: tuple[float, float]
    contour_ok: bool


def threshold_level(corrected: np.ndarray, cfg: Config) -> float:
    """Compute the foreground/background intensity level for a flat-fielded frame.

    Objects are darker than the returned level (`corrected < level`).

    - `"half_max"` (default): estimates the opaque-object core intensity as
      a low percentile (`cfg.threshold.core_percentile`) of the *dark*
      pixels, then places the level a fraction
      (`cfg.threshold.half_max_fraction`) of the way from that core back up
      to the background level of 1.0. The dark subset is the foreground side
      of an Otsu split (`corrected < threshold_otsu(corrected)`) rather than
      a fixed cut like the frame median: under a residual illumination
      gradient a large share of *background* pixels can fall below the
      median, which would drag the "core" estimate toward background and
      inflate the recovered level/area. Otsu locates the object/background
      valley itself, so the dark subset stays foreground-only regardless of
      gradient — and regardless of how little of the frame the object
      covers, which a whole-frame low percentile would otherwise miss
      entirely.
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
        otsu_split = float(threshold_otsu(corrected))
        dark = corrected[corrected < otsu_split]
        # A blank/background-only frame has no pixels below the Otsu split;
        # fall back to a whole-frame percentile so this can't crash or
        # divide by an empty selection.
        percentile_source = dark if dark.size else corrected
        core = float(np.percentile(percentile_source, cfg.threshold.core_percentile))
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
