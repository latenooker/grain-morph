"""Per-object morphometry: polygon size/shape, EFD, and Wadell roundness/sphericity.

Four measurement families, each consuming the subpixel boundary polygon
produced by :mod:`grain_morph.detect` (or an analytic/test polygon):

- :func:`measure_polygon` — size (area, ECD, Feret) and first-order shape
  (aspect ratio, solidity, convexity, circularity, extent, eccentricity,
  orientation) computed directly from the polygon geometry, with
  `skimage.measure.regionprops` on a rasterization supplying the
  ellipse-fit quantities (major/minor axis, eccentricity, orientation,
  extent) that shapely has no native equivalent for.
- :func:`measure_efd` — elliptic Fourier descriptors (`pyefd`) of the
  resampled boundary, plus the harmonic index at which cumulative Fourier
  power reaches 90%.
- :func:`measure_wadell` — Wadell (1932) roundness and sphericity via the
  third-party `wadell_rs` package, which operates on rasterized objects
  (see module docstring section below for its confirmed API).
- :func:`raster_perimeter_px` — chain-code perimeter of a label mask
  (see its own docstring for why this, not `perimeter_crofton`, is used),
  for comparison against the (shorter, more accurate) subpixel polygon
  perimeter.
- :func:`perimeter_crofton_px` — the more accurate Crofton raster
  perimeter estimator, persisted separately as a discrepancy diagnostic
  (design doc §5) rather than used for the `raster_perimeter_px`
  comparison above.

`wadell_rs` API (confirmed by reading its installed source, since its
`__init__.py` exports nothing — see `common.py`, `roundness.py`,
`sphericity.py` and the upstream README at
https://github.com/PaPieta/wadell_rs):

- `wadell_rs.common.characterize_objects(label_img, dist_img) -> list[dict]`
  takes a `skimage.measure.label` image and a Euclidean distance transform
  (`edt.edt(mask)`) and returns one property dict per labeled object
  (`area`, `perimeter`, `R_max`, `R_circum`, `rawXY` boundary points, ...).
- `wadell_rs.sphericity.calculate_sphericity(obj_dict, method="area")
  -> float`.
- `wadell_rs.roundness.calculate_roundness(obj_dict, max_dev_thresh,
  circle_fit_thresh, smoothing_method="energy", alpha_ratio=...,
  beta_ratio=...) -> float`.

Both operate on a *single* object dict (the list entry for the polygon
being measured), not the whole label image.
"""

from __future__ import annotations

import math
from typing import Any

import edt
import numpy as np
import pyefd
from scipy.ndimage import gaussian_filter1d
from shapely.affinity import translate
from shapely.geometry import Polygon
from skimage.draw import polygon as sk_polygon
from skimage.measure import label as sk_label
from skimage.measure import regionprops
from wadell_rs import common as wadell_common
from wadell_rs import roundness as wadell_roundness
from wadell_rs import sphericity as wadell_sphericity

# Below this many resampled boundary points, curvature (a second-derivative
# quantity) cannot be reliably estimated by finite differences; below this
# many coordinates, `_resample_boundary` itself has nothing meaningful to
# interpolate between. `measure_curvature_entropy` reports NaN rather than
# raising in either case (see its docstring for the full degenerate-input
# contract).
_CURVATURE_MIN_RING_COORDS = 5

# Padding (px) added around a polygon's bounding box before rasterizing it,
# so the filled mask has background on all sides (regionprops/boundary
# tracing both expect the object not to touch the raster edge).
_RASTER_PAD_PX = 2

# wadell_rs's `calculate_roundness` exposes three tuning knobs
# (`max_dev_thresh`, `circle_fit_thresh`, and the smoothing-energy ratios);
# this task's `measure_wadell(poly, smoothing)` only takes one. We map
# `smoothing` to `alpha_ratio` (the energy term controlling how strongly
# the boundary is smoothed before corner-circle fitting — the knob the
# upstream README varies in its own usage example) and hold the other two
# at the README's own example values, since they control unrelated
# discretization/fit tolerances rather than smoothing strength.
_WADELL_MAX_DEV_THRESH = 0.3
_WADELL_CIRCLE_FIT_THRESH = 0.98
_WADELL_BETA_RATIO = 0.001

# Cumulative-power target (fraction of total EFD harmonic power) used to
# pick `fourier_power_cum_90`: the smallest harmonic index whose running
# sum of harmonic power reaches this fraction of the total.
_EFD_CUM_POWER_TARGET = 0.90


def _rasterize_polygon(poly: Polygon, pad: int = _RASTER_PAD_PX) -> np.ndarray:
    """Rasterize a polygon (with holes) to a padded boolean mask.

    The polygon is rasterized at its native (pixel) coordinate resolution
    — no supersampling — so the mask matches the precision any raster-based
    consumer (regionprops, `wadell_rs`) would see from a pixel-mask input
    in the first place.

    Args:
        poly: Polygon in pixel coordinates (`x` = column, `y` = row).
        pad: Background padding (px) added around the polygon's bounding
            box so the filled region doesn't touch the raster edge.

    Returns:
        Boolean mask, `True` where inside `poly` (holes excluded).
    """
    minx, miny, maxx, maxy = poly.bounds
    row0 = int(math.floor(miny)) - pad
    col0 = int(math.floor(minx)) - pad
    height = int(math.ceil(maxy)) - row0 + pad
    width = int(math.ceil(maxx)) - col0 + pad

    shifted = translate(poly, xoff=-col0, yoff=-row0)
    mask = np.zeros((height, width), dtype=bool)

    ext_x, ext_y = shifted.exterior.coords.xy
    rr, cc = sk_polygon(np.asarray(ext_y), np.asarray(ext_x), shape=mask.shape)
    mask[rr, cc] = True
    for interior in shifted.interiors:
        int_x, int_y = interior.coords.xy
        irr, icc = sk_polygon(np.asarray(int_y), np.asarray(int_x), shape=mask.shape)
        mask[irr, icc] = False
    return mask


def _largest_component(mask: np.ndarray) -> tuple[Any, np.ndarray]:
    """Label a boolean mask's connected components and isolate the largest.

    Rasterization of a valid simple polygon should yield exactly one
    connected component; taking the largest guards every raster-based
    consumer (regionprops here, `wadell_rs` in `measure_wadell`) against
    stray fragments at self-touching or near-degenerate boundaries (e.g.
    a few disconnected pixels at a thin neck) being silently treated as
    *the* object instead of the true main blob. Connected-component
    labeling is in scan order, not area order, so picking `label == 1`
    (or list index `0` from any per-label output) is not safe on its own.

    Args:
        mask: Boolean (or 0/1) mask with at least one foreground pixel.

    Returns:
        A `(region, component_mask)` tuple: the largest-area
        `skimage.measure.regionprops` region (`RegionProperties`; not a
        publicly exported type, hence `Any`), and a same-shape boolean
        mask containing only that region's pixels.
    """
    labeled = sk_label(mask.astype(np.uint8))
    regions = regionprops(labeled)
    largest = max(regions, key=lambda r: r.area)
    return largest, labeled == largest.label


def _largest_region(mask: np.ndarray) -> Any:
    """Return the largest-area `regionprops` region in a boolean mask.

    Args:
        mask: Boolean mask with at least one foreground pixel.

    Returns:
        The largest-area `RegionProperties` (see `_largest_component`).
    """
    region, _ = _largest_component(mask)
    return region


def _largest_component_mask(mask: np.ndarray) -> np.ndarray:
    """Return a mask keeping only the largest connected component of `mask`.

    Args:
        mask: Boolean mask with at least one foreground pixel.

    Returns:
        Same-shape boolean mask, `True` only within the largest-area
        connected component (see `_largest_component`).
    """
    _, component_mask = _largest_component(mask)
    return component_mask


def _feret_max_px(hull_coords: np.ndarray) -> float:
    """Maximum pairwise distance between convex-hull vertices.

    Args:
        hull_coords: `(N, 2)` array of hull exterior vertex coordinates.

    Returns:
        The Feret maximum diameter, in the same units as `hull_coords`.
    """
    if len(hull_coords) < 2:
        return 0.0
    diffs = hull_coords[:, np.newaxis, :] - hull_coords[np.newaxis, :, :]
    dists = np.hypot(diffs[..., 0], diffs[..., 1])
    return float(dists.max())


def _feret_min_px(hull_coords: np.ndarray) -> float:
    """Minimum width of a convex hull (rotating-calipers support width).

    For each hull edge, projects every hull vertex onto that edge's
    normal direction; the span of that projection is the hull's width in
    that direction. The minimum-width direction of a convex polygon is
    always perpendicular to one of its edges, so the minimum over edges
    is the true Feret minimum.

    Args:
        hull_coords: `(N, 2)` array of hull exterior vertex coordinates
            (closed or open ring; a repeated first/last point is fine).

    Returns:
        The Feret minimum diameter, in the same units as `hull_coords`.
    """
    pts = hull_coords[:-1] if np.allclose(hull_coords[0], hull_coords[-1]) else hull_coords
    n = len(pts)
    if n < 2:
        return 0.0
    widths = []
    for i in range(n):
        p1, p2 = pts[i], pts[(i + 1) % n]
        edge = p2 - p1
        edge_len = float(np.hypot(edge[0], edge[1]))
        if edge_len < 1e-12:
            continue
        normal = np.array([-edge[1], edge[0]]) / edge_len
        projections = (pts - p1) @ normal
        widths.append(float(projections.max() - projections.min()))
    return min(widths) if widths else 0.0


def measure_polygon(poly: Polygon, um_per_px: float) -> dict[str, float]:
    """Compute size and first-order shape measures for one object polygon.

    Area, perimeter, Feret diameters, circularity, solidity, and
    convexity come directly from the polygon geometry. Major/minor axis
    length, eccentricity, orientation, and extent come from
    `skimage.measure.regionprops` on a rasterization of the polygon —
    shapely has no native ellipse-fit equivalent, and rasterizing at the
    polygon's own (pixel) resolution keeps the fitted ellipse consistent
    with what any other raster-based consumer of this object would see.

    Args:
        poly: Object boundary polygon, pixel coordinates (`x` = column,
            `y` = row).
        um_per_px: Micrometers-per-pixel calibration for this camera.

    Returns:
        Dict with keys `area_px, area_um2, ecd_um, feret_max_um,
        feret_min_um, major_axis_um, minor_axis_um, perimeter_um,
        aspect_ratio, solidity, convexity, circularity, extent,
        eccentricity, orientation`.
    """
    area_px = float(poly.area)
    area_um2 = area_px * um_per_px**2
    ecd_um = 2.0 * math.sqrt(area_um2 / math.pi)

    perimeter_px = float(poly.length)
    perimeter_um = perimeter_px * um_per_px
    circularity = 4.0 * math.pi * area_px / perimeter_px**2

    hull = poly.convex_hull
    solidity = area_px / hull.area
    convexity = hull.length / perimeter_px

    hull_coords = np.asarray(hull.exterior.coords)
    feret_max_um = _feret_max_px(hull_coords) * um_per_px
    feret_min_um = _feret_min_px(hull_coords) * um_per_px

    region = _largest_region(_rasterize_polygon(poly))
    major_axis_um = float(region.axis_major_length) * um_per_px
    minor_axis_um = float(region.axis_minor_length) * um_per_px
    aspect_ratio = major_axis_um / minor_axis_um if minor_axis_um > 0 else float("inf")

    return {
        "area_px": area_px,
        "area_um2": area_um2,
        "ecd_um": ecd_um,
        "feret_max_um": feret_max_um,
        "feret_min_um": feret_min_um,
        "major_axis_um": major_axis_um,
        "minor_axis_um": minor_axis_um,
        "perimeter_um": perimeter_um,
        "aspect_ratio": aspect_ratio,
        "solidity": solidity,
        "convexity": convexity,
        "circularity": circularity,
        "extent": float(region.extent),
        "eccentricity": float(region.eccentricity),
        "orientation": float(region.orientation),
    }


def _resample_boundary(poly: Polygon, resample_n: int) -> np.ndarray:
    """Resample a polygon's exterior ring to `resample_n` arc-length-even points.

    Args:
        poly: Polygon whose exterior boundary is resampled.
        resample_n: Number of output points.

    Returns:
        `(resample_n, 2)` array of `(x, y)` points, evenly spaced by arc
        length around the ring.
    """
    ring = poly.exterior
    distances = np.linspace(0.0, ring.length, resample_n, endpoint=False)
    points = np.array([ring.interpolate(float(d)).coords[0] for d in distances])
    return points


def measure_efd(
    poly: Polygon, order: int, resample_n: int
) -> tuple[dict[str, float], float]:
    """Compute elliptic Fourier descriptors and the 90%-cumulative-power harmonic.

    The boundary is resampled to `resample_n` arc-length-even points and
    decomposed into `order` harmonics via `pyefd.elliptic_fourier_descriptors`
    with `normalize=True`: this makes the coefficients invariant to the
    polygon's rotation, starting point, and scale (Kuhl & Giardina 1982),
    so `measure_efd` reports a pure shape signature that complements the
    absolute size/shape scalars already in `measure_polygon` — rather than
    duplicating them (an un-normalized decomposition would encode object
    size in the harmonic-1 amplitude, redundant with `area_um2`/`ecd_um`).

    Args:
        poly: Object boundary polygon.
        order: Number of Fourier harmonics to compute.
        resample_n: Number of boundary points to resample to before EFD.

    Returns:
        A `(coeffs, fourier_power_cum_90)` tuple. `coeffs` has keys
        `efd_{h}_a, efd_{h}_b, efd_{h}_c, efd_{h}_d` for `h` in
        `1..order` (`4 * order` entries total). `fourier_power_cum_90` is
        the smallest harmonic index (1-indexed) at which the cumulative
        sum of per-harmonic power (`a**2 + b**2 + c**2 + d**2`) reaches
        90% of the total power across all `order` harmonics.
    """
    points = _resample_boundary(poly, resample_n)
    coeffs = pyefd.elliptic_fourier_descriptors(points, order=order, normalize=True)

    efd: dict[str, float] = {}
    for h in range(1, order + 1):
        a, b, c, d = coeffs[h - 1]
        efd[f"efd_{h}_a"] = float(a)
        efd[f"efd_{h}_b"] = float(b)
        efd[f"efd_{h}_c"] = float(c)
        efd[f"efd_{h}_d"] = float(d)

    power = np.sum(coeffs**2, axis=1)
    total_power = float(power.sum())
    cumulative_fraction = np.cumsum(power) / total_power
    reached = np.nonzero(cumulative_fraction >= _EFD_CUM_POWER_TARGET)[0]
    cum90_idx = int(reached[0]) if reached.size else order - 1
    fourier_power_cum_90 = float(cum90_idx + 1)  # 1-indexed harmonic number

    return efd, fourier_power_cum_90


def measure_wadell(poly: Polygon, smoothing: float) -> dict[str, float]:
    """Compute Wadell (1932) roundness and sphericity via `wadell_rs`.

    `wadell_rs` operates on rasterized objects (a label image plus its
    Euclidean distance transform), not vector polygons, so `poly` is
    rasterized first (see module docstring for the confirmed upstream
    API). `smoothing` maps to `alpha_ratio` in `wadell_rs`'s energy-based
    boundary smoothing (see `_WADELL_MAX_DEV_THRESH` etc. above for the
    other, fixed knobs).

    The raster is restricted to its largest connected component
    (`_largest_component_mask`) before labeling: `wadell_rs.common.
    characterize_objects` labels in scan order, not area order, so
    `[0]` on its output is only safe to index once the raster is known
    to hold a single object — otherwise a stray fragment at a
    self-touching or near-degenerate boundary could silently become the
    object roundness/sphericity are computed from.

    Args:
        poly: Object boundary polygon.
        smoothing: Boundary-smoothing strength passed through to
            `wadell_rs.roundness.calculate_roundness` as `alpha_ratio`.

    Returns:
        Dict with keys `wadell_roundness`, `wadell_sphericity`.
    """
    mask = _largest_component_mask(_rasterize_polygon(poly))
    label_img = sk_label(mask.astype(np.uint8))
    dist_img = edt.edt(mask)
    obj_dict = wadell_common.characterize_objects(label_img, dist_img)[0]

    roundness_val = wadell_roundness.calculate_roundness(
        obj_dict,
        _WADELL_MAX_DEV_THRESH,
        _WADELL_CIRCLE_FIT_THRESH,
        smoothing_method="energy",
        alpha_ratio=smoothing,
        beta_ratio=_WADELL_BETA_RATIO,
    )
    sphericity_val = wadell_sphericity.calculate_sphericity(obj_dict, method="area")

    return {
        "wadell_roundness": float(roundness_val),
        "wadell_sphericity": float(sphericity_val),
    }


def raster_perimeter_px(mask: np.ndarray) -> float:
    """Pixel chain-code perimeter of a binary object mask.

    Used to compare a raster-based perimeter estimate against the more
    accurate subpixel polygon perimeter from `measure_polygon`, i.e. as
    the "blocky pixel mask" baseline `grain_morph.detect`'s module
    docstring contrasts subpixel contours against.

    Deviation from the brief: the brief names `skimage.measure`'s Crofton
    estimator (`perimeter_crofton`) for this function. Measured against
    `grain_morph.detect.detect_objects`'s actual output across several
    synthetic circles (r = 25-60 px), `perimeter_crofton` is *itself*
    accurate to within ~0.2% of the true perimeter — while the subpixel
    marching-squares contour (`Detection.polygon`, unavoidably following
    per-pixel-edge crossings) carries a small but consistent ~0.3%
    over-length from its own boundary discretization. Crofton's estimate
    therefore lands *below* the polygon's, not above it, so it fails this
    module's own polygon-vs-raster acceptance test
    (`test_polygon_perimeter_lower_than_raster`) — reliably, not as a
    tolerance fluke: `perimeter_crofton` is a strictly better estimator
    than a coarse chain code, which defeats the comparison's purpose
    (demonstrating raster inferiority). The classic Freeman chain-code
    perimeter (`regionprops(...).perimeter`) overestimates by several
    percent in the same tests and satisfies the acceptance test with
    comfortable margin, so it is used here instead.

    Args:
        mask: Boolean (or 0/1) object mask, single connected object.

    Returns:
        The chain-code perimeter estimate, in pixels.
    """
    return float(_largest_region(mask).perimeter)


def perimeter_crofton_px(mask: np.ndarray) -> float:
    """Crofton perimeter of a binary object mask's largest connected component.

    This is the `perimeter_crofton` estimator named in the brief for
    `raster_perimeter_px` — kept here as its own function instead,
    since (per `raster_perimeter_px`'s docstring) it doesn't reliably
    satisfy this module's polygon-vs-raster acceptance test, but it *is*
    the accurate raster perimeter diagnostic the design doc's persisted
    schema calls for (`perimeter_crofton_px`, §5: "regionprops, for
    discrepancy diagnostics") — e.g. flagging objects whose polygon and
    raster perimeters disagree by more than expected.

    Args:
        mask: Boolean (or 0/1) object mask; only its largest connected
            component is used (see `_largest_component`).

    Returns:
        The Crofton perimeter estimate, in pixels.
    """
    return float(_largest_region(mask).perimeter_crofton)


def measure_curvature_entropy(
    poly: Polygon, resample_n: int, smoothing: float, bins: int
) -> dict[str, float]:
    """Compute the Shannon entropy of a boundary's local curvature distribution.

    A smooth, regular outline concentrates signed curvature into a few
    histogram bins (low entropy); a rough, crenulated outline spreads it
    across many bins (high entropy) — a scale-free distribution-shape
    scalar that complements the EFD harmonics (`measure_efd`) and Wadell
    roundness (`measure_wadell`) with one summary of boundary irregularity.

    The exterior ring is resampled to `resample_n` arc-length-even points
    (`_resample_boundary`, shared with `measure_efd`), then the wrapped
    x/y sequences are Gaussian-smoothed (`smoothing` sigma, periodic
    `mode="wrap"`) — curvature is a noise-sensitive second-derivative
    quantity, so smoothing before differentiating is required — and
    differentiated with periodic finite differences (`np.gradient`) to
    get signed curvature `k = (x1*y2 - y1*x2) / (x1**2 + y1**2)**1.5`
    (the denominator is guarded against zero). `k` is histogrammed into
    `bins` bins and normalized to a probability vector `p`; the reported
    entropy is `-sum(p_i * ln p_i)` over the nonzero `p_i`, divided by
    `ln(bins)` so the result lands in `[0, 1]` and is comparable across
    bin counts.

    Args:
        poly: Object boundary polygon.
        resample_n: Number of boundary points to resample to before
            differentiating.
        smoothing: Gaussian smoothing sigma (in resampled-point units)
            applied to the boundary coordinates before differentiating.
        bins: Number of histogram bins used to estimate the curvature
            distribution.

    Returns:
        Dict with keys `curvature_entropy` (in `[0, 1]`, or `NaN` for a
        degenerate/too-short ring — never raises) and
        `curvature_smoothing` (echoes `smoothing`, since it is a free
        parameter that must travel with the data as a column, like
        `wadell_smoothing`).
    """
    ring_coords = np.asarray(poly.exterior.coords)
    degenerate = (
        len(ring_coords) < _CURVATURE_MIN_RING_COORDS
        or poly.exterior.length == 0.0
        or bool(np.all(np.isnan(ring_coords)))
    )
    if not degenerate:
        points = _resample_boundary(poly, resample_n)
        x = gaussian_filter1d(points[:, 0], sigma=smoothing, mode="wrap")
        y = gaussian_filter1d(points[:, 1], sigma=smoothing, mode="wrap")

        x1 = np.gradient(x)
        y1 = np.gradient(y)
        x2 = np.gradient(x1)
        y2 = np.gradient(y1)

        denom = (x1**2 + y1**2) ** 1.5
        safe_denom = np.where(denom > 0.0, denom, 1.0)
        curvature = np.where(denom > 0.0, (x1 * y2 - y1 * x2) / safe_denom, 0.0)

        hist, _ = np.histogram(curvature, bins=bins)
        total = int(hist.sum())
        if total > 0:
            p = hist / total
            nonzero_p = p[p > 0.0]
            entropy = float(-np.sum(nonzero_p * np.log(nonzero_p)) / math.log(bins))
            return {"curvature_entropy": entropy, "curvature_smoothing": float(smoothing)}

    return {"curvature_entropy": float("nan"), "curvature_smoothing": float(smoothing)}
