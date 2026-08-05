"""Per-object QC metrics and flags: the focus/border/size/shape gate.

This is the pipeline's most important discriminator — it decides which
detections are trustworthy grain measurements versus defocused blobs,
border-clipped fragments, slivers/debris, or probable agglomerates. Per the
design doc, **nothing is ever silently dropped**: every object gets its
continuous metrics and boolean flags recorded, and `qc_pass` is just a
configurable aggregate of which flags are "disqualifying"
(`cfg.qc.disqualifying_flags`) — so a stricter or looser gate can be applied
downstream without re-running detection.

Two functions, matching the two halves of that job:

- :func:`qc_metrics` computes the continuous, observational quantities:
  `edge_width_px` (10-90% intensity-rise distance sampled along boundary
  normals — the *primary*, scale-free focus signal), `edge_gradient` /
  `edge_gradient_norm` and `contrast` (the raw-image gradient/contrast
  cross-checks), and `ecd_px` / `aspect_ratio` / `solidity` (size/shape). It
  also records two pass-through booleans (`touches_border`, `has_polygon`)
  that :func:`qc_flags` needs but has no other way to recover, since its own
  signature is deliberately just `(metrics, cfg)`.
- :func:`qc_flags` applies `cfg.qc` thresholds to those metrics and returns
  every named flag plus `qc_pass`.

Focus metrics are measured on the **raw** image (before flat-fielding):
flat-fielding rescales intensities for thresholding, but the raw sensor
values are what actually carry the optical blur signature. Geometry
(`ecd_px`, `aspect_ratio`, `solidity`) comes from the polygon/mask instead,
which are themselves already derived from the flat-fielded frame upstream
(`grain_morph.detect`).
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage
from shapely.affinity import translate
from shapely.geometry import Polygon
from skimage.filters import sobel
from skimage.measure import find_contours, regionprops
from skimage.measure import label as sk_label

from grain_morph.config import Config

# --- Private tuning constants -------------------------------------------
#
# None of these are exposed on `Config.qc` (that schema is fixed by the
# interfaces Task 12 consumes: min_ecd_px, defocus_edge_width_px,
# defocus_contrast_min, sliver_aspect_ratio, sliver_max_ecd_px,
# agglomerate_solidity_max, disqualifying_flags). They control *how* a
# metric is measured, not a pass/fail threshold, so they live here as named
# constants instead.

# Half-width (px) of the band straddling the boundary within which
# `edge_gradient` (mean Sobel magnitude) is measured — the brief's "within a
# ±3 px band around the boundary."
_EDGE_BAND_PX = 3

# Erosion depth (px) carving the "core" interior and dilation depth (px)
# carving the "local background" ring for `contrast`. Kept equal so core and
# ring sit at a comparable distance from the boundary on either side.
_CONTRAST_MARGIN_PX = 5

# Number of boundary points sampled (evenly by arc length) for the
# `edge_width_px` normal-profile measurement.
_N_BOUNDARY_SAMPLES = 48

# Step size (px) along each boundary-normal intensity profile.
_NORMAL_STEP_PX = 0.5

# Each normal profile's half-length is capped at this many px...
_NORMAL_HALF_LEN_MAX_PX = 15.0
# ...but also capped at this fraction of the object's own local half-width,
# so a normal probing "inward" on a small object can't tunnel past the
# opposite boundary and sample the far side as if it were the core.
_NORMAL_RADIUS_FRACTION = 0.6
# ...with this floor, so very small objects still get *some* signal rather
# than a zero-length profile (they are typically also caught by
# `flag_too_small` regardless).
_NORMAL_HALF_LEN_MIN_PX = 4.0

# Small outward step (px) used to test which of the two candidate boundary
# normal directions points away from the object (into background).
_OUTWARD_PROBE_PX = 2.0

# 10-90% rise fractions defining `edge_width_px`.
_RISE_LO_FRAC = 0.10
_RISE_HI_FRAC = 0.90

# Padding (px) added around an object's bbox before cropping the frame for
# all local morphology/normal-sampling below. Generous enough to contain the
# largest of the margins above (ring dilation, edge band, normal half-length)
# with room to spare, and to keep unrelated nearby objects out of the crop.
_CROP_PAD_PX = 20


def _crop_bounds(
    bbox: tuple[int, int, int, int], frame_shape: tuple[int, int], pad: int
) -> tuple[int, int, int, int]:
    """Padded crop window around an object's bbox, clipped to the frame.

    Args:
        bbox: `(min_row, min_col, max_row, max_col)`, `max_row`/`max_col`
            exclusive (as produced by `regionprops`/`Detection.mask_bbox`).
        frame_shape: `(height, width)` of the full frame.
        pad: Padding (px) added on every side before clipping.

    Returns:
        `(row0, col0, row1, col1)` crop bounds, `row1`/`col1` exclusive.
    """
    min_row, min_col, max_row, max_col = bbox
    height, width = frame_shape
    row0 = max(0, min_row - pad)
    col0 = max(0, min_col - pad)
    row1 = min(height, max_row + pad)
    col1 = min(width, max_col + pad)
    return row0, col0, row1, col1


def _largest_labeled_mask(mask: np.ndarray) -> np.ndarray:
    """Restrict a boolean mask to its largest connected component.

    Defensive against a mask that (after cropping) ends up with stray
    disconnected pixels; mirrors `grain_morph.measure._largest_component`.

    Args:
        mask: Boolean mask.

    Returns:
        Same-shape boolean mask; unchanged if `mask` has no foreground
        pixels at all.
    """
    if not mask.any():
        return mask
    labeled = sk_label(mask.astype(np.uint8))
    regions = regionprops(labeled)
    largest = max(regions, key=lambda r: r.area)
    return labeled == largest.label


def _mask_geometry(mask: np.ndarray) -> tuple[float, float, float]:
    """`(ecd_px, aspect_ratio, solidity)` from a raster mask via `regionprops`.

    Used as the geometry fallback when no polygon is available, and as the
    source of `aspect_ratio`/`solidity` even when one is (mirroring
    `measure.measure_polygon`'s own use of a raster ellipse fit for
    `aspect_ratio`, since shapely has no native ellipse-fit equivalent).

    Args:
        mask: Boolean object mask; assumed single connected component.

    Returns:
        `(ecd_px, aspect_ratio, solidity)`. All zero/degenerate if `mask`
        has no foreground pixels.
    """
    if not mask.any():
        return 0.0, 1.0, 1.0
    region = regionprops(sk_label(mask.astype(np.uint8)))[0]
    area_px = float(region.area)
    ecd_px = 2.0 * math.sqrt(area_px / math.pi)
    minor = float(region.axis_minor_length)
    major = float(region.axis_major_length)
    aspect_ratio = major / minor if minor > 0 else float("inf")
    return ecd_px, aspect_ratio, float(region.solidity)


def _contrast_metrics(raw_crop: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    """`(contrast, local_bg_mean, core_mean)` from a raw-intensity crop.

    `local_bg_mean` is the mean raw intensity in a ring just outside the
    mask (`dilate(mask) & ~mask`); `core_mean` is the mean in an eroded
    interior (`erode(mask)`). `contrast = (local_bg_mean - core_mean) /
    local_bg_mean`.

    Args:
        raw_crop: Raw-image crop, float, same shape as `mask`.
        mask: Boolean object mask, single connected component.

    Returns:
        `(contrast, local_bg_mean, core_mean)`.
    """
    dilated = ndimage.binary_dilation(mask, iterations=_CONTRAST_MARGIN_PX)
    eroded = ndimage.binary_erosion(mask, iterations=_CONTRAST_MARGIN_PX)
    ring = dilated & ~mask
    core = eroded if eroded.any() else mask

    if ring.any():
        bg_mean = float(raw_crop[ring].mean())
    else:
        # Degenerate (object fills the whole padded crop): fall back to
        # whatever background pixels remain in the crop.
        background = ~mask
        bg_mean = float(raw_crop[background].mean()) if background.any() else float(raw_crop.mean())
    core_mean = float(raw_crop[core].mean())
    contrast = (bg_mean - core_mean) / bg_mean if bg_mean != 0 else 0.0
    return contrast, bg_mean, core_mean


def _edge_gradient(raw_crop: np.ndarray, mask: np.ndarray) -> float:
    """Mean Sobel gradient magnitude in a `±_EDGE_BAND_PX` band around the boundary.

    Args:
        raw_crop: Raw-image crop, float, same shape as `mask`.
        mask: Boolean object mask, single connected component.

    Returns:
        Mean Sobel magnitude (`skimage.filters.sobel` on `raw_crop`) within
        the band, or `nan` if the band is empty.
    """
    dilated = ndimage.binary_dilation(mask, iterations=_EDGE_BAND_PX)
    eroded = ndimage.binary_erosion(mask, iterations=_EDGE_BAND_PX)
    band = dilated & ~eroded
    if not band.any():
        band = dilated & ~mask  # object too small to erode; band = outer half only
    if not band.any():
        return float("nan")
    grad = sobel(raw_crop)
    return float(grad[band].mean())


def _boundary_points_and_normals(
    poly: Polygon | None, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """Evenly-spaced boundary points and outward unit normals.

    Points come from `poly.exterior`, resampled by arc length, when a
    polygon is available; otherwise from a raster contour of `mask` (a
    coarser but workable fallback for the rare case `detect_objects`
    couldn't associate a contour with this label).

    Args:
        poly: Object boundary polygon, or `None`.
        mask: Boolean object mask, same coordinate frame as `poly`.

    Returns:
        `(points, normals)`, each `(N, 2)` arrays of `(x, y)`, or `None` if
        no boundary could be established at all (empty mask).
    """
    if poly is not None and poly.exterior.length > 0:
        ring = poly.exterior
        distances = np.linspace(0.0, ring.length, _N_BOUNDARY_SAMPLES, endpoint=False)
        pts = np.array([ring.interpolate(float(d)).coords[0] for d in distances])
    else:
        contours = find_contours(mask.astype(float), 0.5)
        if not contours:
            return None
        largest = max(contours, key=len)
        idx = np.linspace(0, len(largest), _N_BOUNDARY_SAMPLES, endpoint=False).astype(int)
        idx = idx % len(largest)
        pts = largest[idx][:, ::-1]  # (row, col) -> (x, y)

    tangent = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
    normals = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normals = normals / norms

    height, width = mask.shape
    for i in range(len(pts)):
        probe = pts[i] + normals[i] * _OUTWARD_PROBE_PX
        col = min(max(int(round(probe[0])), 0), width - 1)
        row = min(max(int(round(probe[1])), 0), height - 1)
        if mask[row, col]:
            normals[i] = -normals[i]
    return pts, normals


def _edge_width_px(
    raw_crop: np.ndarray,
    poly: Polygon | None,
    mask: np.ndarray,
    core_mean: float,
    bg_mean: float,
) -> float:
    """Median 10-90% intensity-rise distance along boundary normals.

    For each sampled boundary point, an intensity profile is interpolated
    along the outward normal (`scipy.ndimage.map_coordinates`) and the
    distance between where it crosses 10% and 90% of the way from
    `core_mean` to `bg_mean` is measured. Using those two *global*,
    already-computed levels (rather than each profile's own local extremes)
    means the profile's sampling window only needs to bracket the
    transition itself, not reach a flat plateau — which matters because a
    window wide enough to reach a flat plateau on a strongly blurred edge
    could otherwise tunnel past a small object's opposite boundary.

    Args:
        raw_crop: Raw-image crop, float.
        poly: Object boundary polygon in `raw_crop`-local coordinates, or
            `None`.
        mask: Boolean object mask, same coordinate frame as `raw_crop`.
        core_mean: Mean raw intensity of the eroded interior (from
            `_contrast_metrics`).
        bg_mean: Mean raw intensity of the local background ring (from
            `_contrast_metrics`).

    Returns:
        The median rise distance (px) across sampled boundary points, or
        `nan` if it could not be measured (no usable boundary, or
        `bg_mean <= core_mean`).
    """
    if bg_mean <= core_mean:
        return float("nan")
    result = _boundary_points_and_normals(poly, mask)
    if result is None:
        return float("nan")
    pts, normals = result

    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return float("nan")
    half_dim = 0.5 * min(rows.max() - rows.min() + 1, cols.max() - cols.min() + 1)
    half_len_raw = _NORMAL_RADIUS_FRACTION * half_dim
    half_len = float(np.clip(half_len_raw, _NORMAL_HALF_LEN_MIN_PX, _NORMAL_HALF_LEN_MAX_PX))
    offsets = np.arange(-half_len, half_len + _NORMAL_STEP_PX, _NORMAL_STEP_PX)

    lo_level = core_mean + _RISE_LO_FRAC * (bg_mean - core_mean)
    hi_level = core_mean + _RISE_HI_FRAC * (bg_mean - core_mean)

    widths = []
    for pt, nrm in zip(pts, normals, strict=True):
        xs = pt[0] + offsets * nrm[0]
        ys = pt[1] + offsets * nrm[1]
        profile = ndimage.map_coordinates(raw_crop, [ys, xs], order=1, mode="nearest")
        if profile[-1] <= profile[0]:
            continue  # not a clean dark-core -> bright-background rise; skip
        lo_pos = float(np.interp(lo_level, profile, offsets))
        hi_pos = float(np.interp(hi_level, profile, offsets))
        width = hi_pos - lo_pos
        if width > 0:
            widths.append(width)
    return float(np.median(widths)) if widths else float("nan")


def qc_metrics(
    raw_image: np.ndarray,
    corrected: np.ndarray,
    poly: Polygon | None,
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    frame_shape: tuple[int, int],
    cfg: Config,
) -> dict[str, float | bool]:
    """Compute continuous QC metrics for one detected object.

    Focus/contrast (`edge_width_px`, `edge_gradient`, `edge_gradient_norm`,
    `contrast`) are measured on `raw_image`; geometry (`ecd_px`,
    `aspect_ratio`, `solidity`) comes from `poly`/`mask`. `corrected` is
    accepted for interface parity with the rest of the per-object pipeline
    (this is the exact signature `grain_morph.aggregate` will call per
    detection) but is not itself read: nothing here needs flat-fielded
    intensities.

    Also records two booleans, `touches_border` and `has_polygon`, that
    :func:`qc_flags` needs but cannot otherwise derive from a `(metrics,
    cfg)`-only signature.

    Args:
        raw_image: Full, un-flat-fielded frame (`f.image.astype(float)`),
            same shape as `frame_shape`.
        corrected: Flat-fielded frame, same shape; unused (see above).
        poly: Object boundary polygon (`grain_morph.detect.Detection.
            polygon`), or `None` if contour extraction failed.
        mask: Boolean object mask over the full frame (e.g. `labels ==
            d.label`).
        bbox: `(min_row, min_col, max_row, max_col)` pixel bbox, `max_row`/
            `max_col` exclusive (`Detection.mask_bbox`).
        frame_shape: `(height, width)` of the full frame.
        cfg: Resolved pipeline configuration (unused directly here — the
            metrics are threshold-free; `cfg.qc` is consumed by
            `qc_flags`). Kept in the signature for a uniform per-object
            call shape alongside `qc_flags`.

    Returns:
        Dict with keys `edge_width_px, edge_gradient, edge_gradient_norm,
        contrast, ecd_px, aspect_ratio, solidity, touches_border,
        has_polygon`.
    """
    min_row, min_col, max_row, max_col = bbox
    height, width = frame_shape[0], frame_shape[1]
    touches_border = min_row <= 0 or min_col <= 0 or max_row >= height or max_col >= width

    row0, col0, row1, col1 = _crop_bounds(bbox, (height, width), _CROP_PAD_PX)
    mask_crop = _largest_labeled_mask(mask[row0:row1, col0:col1])
    raw_crop = raw_image[row0:row1, col0:col1].astype(np.float64)
    poly_crop = translate(poly, xoff=-col0, yoff=-row0) if poly is not None else None

    if poly is not None and poly.area > 0:
        ecd_px = 2.0 * math.sqrt(float(poly.area) / math.pi)
        _, aspect_ratio, solidity = _mask_geometry(mask_crop)
    else:
        ecd_px, aspect_ratio, solidity = _mask_geometry(mask_crop)

    contrast, local_bg_mean, core_mean = _contrast_metrics(raw_crop, mask_crop)
    edge_gradient = _edge_gradient(raw_crop, mask_crop)
    edge_gradient_norm = edge_gradient / local_bg_mean if local_bg_mean else float("nan")
    edge_width_px = _edge_width_px(raw_crop, poly_crop, mask_crop, core_mean, local_bg_mean)

    return {
        "edge_width_px": edge_width_px,
        "edge_gradient": edge_gradient,
        "edge_gradient_norm": edge_gradient_norm,
        "contrast": contrast,
        "ecd_px": ecd_px,
        "aspect_ratio": aspect_ratio,
        "solidity": solidity,
        "touches_border": touches_border,
        "has_polygon": poly is not None,
    }


def qc_flags(metrics: dict[str, float | bool], cfg: Config) -> dict[str, bool]:
    """Apply `cfg.qc` thresholds to `qc_metrics` output and derive `qc_pass`.

    Args:
        metrics: Output of :func:`qc_metrics`.
        cfg: Resolved pipeline configuration; consumes `cfg.qc`.

    Returns:
        Dict with keys `flag_defocus, flag_border, flag_too_small,
        flag_sliver, flag_possible_agglomerate, flag_no_polygon, qc_pass`.
        `qc_pass` is `True` iff none of `cfg.qc.disqualifying_flags` are
        set.
    """
    qc = cfg.qc
    edge_width_px = float(metrics["edge_width_px"])
    contrast = float(metrics["contrast"])
    ecd_px = float(metrics["ecd_px"])
    aspect_ratio = float(metrics["aspect_ratio"])
    solidity = float(metrics["solidity"])

    is_defocused = edge_width_px > qc.defocus_edge_width_px or contrast < qc.defocus_contrast_min
    flags: dict[str, bool] = {
        "flag_defocus": is_defocused,
        "flag_border": bool(metrics["touches_border"]),
        "flag_too_small": ecd_px < qc.min_ecd_px,
        "flag_sliver": aspect_ratio > qc.sliver_aspect_ratio and ecd_px < qc.sliver_max_ecd_px,
        "flag_possible_agglomerate": solidity < qc.agglomerate_solidity_max,
        "flag_no_polygon": not bool(metrics["has_polygon"]),
    }
    flags["qc_pass"] = not any(flags[name] for name in qc.disqualifying_flags)
    return flags
