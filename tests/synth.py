"""Synthetic frame generator with ground-truth polygons.

Renders dark objects (``"ellipse"``, ``"rect"``, or randomized ``"blob"``
shapes) on a bright background at a known pixel geometry. Each object's
ground-truth boundary is a :class:`shapely.Polygon` built directly in
full-resolution pixel coordinates; the raster silhouette is then produced by
rasterizing that same polygon on a supersampled grid and downscaling with
area-averaging (anti-aliasing). This makes the rendered edge a smooth ramp
whose 50 %-intensity crossing coincides with the ground-truth polygon
boundary — the property downstream pipeline tests rely on when they recover
area/perimeter from a subpixel ``find_contours`` run at the half-maximum
level (see ``docs/superpowers/specs/2026-08-04-grain-morph-design.md``, and
the task-6 brief's "CRITICAL design guidance").

Optional per-object Gaussian blur (``blur_sigma``) simulates defocus and is
applied *after* rasterization, on that object's own intensity layer, before
compositing. A multiplicative illumination gradient and Gaussian read noise
can be applied to the whole frame. Everything is deterministic given
``seed``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage
from shapely.geometry import Polygon
from skimage.draw import polygon as _sk_polygon
from skimage.transform import downscale_local_mean

# Nominal dark-object / bright-background intensities (uint8 levels).
_DARK = 15.0
_BRIGHT = 200.0

# Supersampling factor used to anti-alias object silhouettes so their
# 50%-intensity crossing lands on the true polygon boundary.
_SUPERSAMPLE = 8

# Number of vertices used to approximate ellipse/blob outlines.
_N_OUTLINE_PTS = 128


@dataclass
class SynthObject:
    """A single synthetic object with known ground-truth geometry.

    Attributes:
        polygon: Ground-truth boundary in full-resolution pixel coordinates
            (x, y), i.e. the geometry the rendered silhouette's
            50%-intensity crossing is designed to coincide with.
        blurred: True if this object was rendered with ``blur_sigma > 0``.
        touches_border: True if the polygon's bounding box reaches or
            crosses the frame edge.
    """

    polygon: Polygon
    blurred: bool
    touches_border: bool


@dataclass
class SynthFrame:
    """A rendered synthetic frame plus its per-object ground truth.

    Attributes:
        image: Rendered grayscale frame, shape ``(height, width)``, dtype
            ``uint8``.
        objects: Ground-truth :class:`SynthObject` for each requested
            object, in the same order as the input ``objects`` list.
        background_level: Nominal background intensity before the
            illumination gradient and noise were applied.
    """

    image: np.ndarray
    objects: list[SynthObject]
    background_level: float


def ground_truth_area(obj: SynthObject) -> float:
    """Ground-truth area of an object's polygon.

    Args:
        obj: Synthetic object.

    Returns:
        Polygon area in px^2.
    """
    return float(obj.polygon.area)


def ground_truth_perimeter(obj: SynthObject) -> float:
    """Ground-truth perimeter of an object's polygon.

    Args:
        obj: Synthetic object.

    Returns:
        Polygon perimeter (boundary length) in px.
    """
    return float(obj.polygon.length)


def _rotate(x: np.ndarray, y: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
    """Rotate (x, y) point arrays about the origin by ``angle`` radians."""
    ca, sa = np.cos(angle), np.sin(angle)
    return x * ca - y * sa, x * sa + y * ca


def _ellipse_polygon(cx: float, cy: float, a: float, b: float, angle: float) -> Polygon:
    """Build an ellipse polygon (semi-axes ``a``, ``b``) centered at (cx, cy).

    Args:
        cx: Center x (column) coordinate, px.
        cy: Center y (row) coordinate, px.
        a: Semi-axis along local x before rotation, px.
        b: Semi-axis along local y before rotation, px.
        angle: Rotation, radians.

    Returns:
        Closed shapely polygon approximating the ellipse.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, _N_OUTLINE_PTS, endpoint=False)
    ex, ey = a * np.cos(theta), b * np.sin(theta)
    rx, ry = _rotate(ex, ey, angle)
    return Polygon(np.column_stack([cx + rx, cy + ry]))


def _rect_polygon(cx: float, cy: float, a: float, b: float, angle: float) -> Polygon:
    """Build a rectangle polygon (half-width ``a``, half-height ``b``).

    Args:
        cx: Center x (column) coordinate, px.
        cy: Center y (row) coordinate, px.
        a: Half-width before rotation, px.
        b: Half-height before rotation, px.
        angle: Rotation, radians.

    Returns:
        Closed shapely polygon (4 corners) for the rectangle.
    """
    corners = np.array([(-a, -b), (a, -b), (a, b), (-a, b)])
    rx, ry = _rotate(corners[:, 0], corners[:, 1], angle)
    return Polygon(np.column_stack([cx + rx, cy + ry]))


def _blob_polygon(
    cx: float, cy: float, a: float, b: float, angle: float, rng: np.random.Generator
) -> Polygon:
    """Build a smooth, star-shaped "blob" polygon with an elliptical envelope.

    Radius is perturbed around an ellipse (semi-axes ``a``, ``b``) by a sum
    of a few low-frequency cosine harmonics with bounded amplitude, which
    keeps the outline star-shaped about its center (radius stays positive)
    and therefore simple (non-self-intersecting).

    Args:
        cx: Center x (column) coordinate, px.
        cy: Center y (row) coordinate, px.
        a: Nominal semi-axis along local x before rotation, px.
        b: Nominal semi-axis along local y before rotation, px.
        angle: Rotation, radians.
        rng: Seeded random generator controlling the perturbation.

    Returns:
        Closed shapely polygon for the randomized blob.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, _N_OUTLINE_PTS, endpoint=False)
    n_harmonics = 3
    amplitude = 0.12
    freqs = rng.integers(2, 6, size=n_harmonics)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=n_harmonics)
    weights = rng.uniform(0.3, 1.0, size=n_harmonics)
    weights = weights / weights.sum()
    perturb = sum(
        w * np.cos(f * theta + p) for w, f, p in zip(weights, freqs, phases, strict=True)
    )
    scale = 1.0 + amplitude * perturb
    ex, ey = a * scale * np.cos(theta), b * scale * np.sin(theta)
    rx, ry = _rotate(ex, ey, angle)
    return Polygon(np.column_stack([cx + rx, cy + ry]))


def _object_polygon(spec: dict[str, Any], rng: np.random.Generator) -> Polygon:
    """Dispatch to the polygon builder for ``spec["kind"]``.

    Args:
        spec: Object dict with keys ``kind``, ``cx``, ``cy``, ``a``, ``b``,
            and optionally ``angle`` (radians, default 0.0).
        rng: Seeded random generator (used by ``"blob"``).

    Returns:
        Ground-truth polygon in full-resolution pixel coordinates.

    Raises:
        ValueError: If ``spec["kind"]`` is not one of ``"ellipse"``,
            ``"rect"``, ``"blob"``.
    """
    kind = spec["kind"]
    cx, cy = float(spec["cx"]), float(spec["cy"])
    a, b = float(spec["a"]), float(spec["b"])
    angle = float(spec.get("angle", 0.0))
    if kind == "ellipse":
        return _ellipse_polygon(cx, cy, a, b, angle)
    if kind == "rect":
        return _rect_polygon(cx, cy, a, b, angle)
    if kind == "blob":
        return _blob_polygon(cx, cy, a, b, angle, rng)
    raise ValueError(f"unknown object kind: {kind!r}")


def _rasterize_coverage(polygon: Polygon, size: tuple[int, int]) -> np.ndarray:
    """Anti-aliased per-pixel coverage fraction of ``polygon`` at ``size``.

    Rasterizes the polygon on a grid supersampled by ``_SUPERSAMPLE`` and
    downscales by area-averaging, so each output pixel holds the fraction
    of its area covered by the polygon. The 0.5 contour of this coverage
    field converges to the true polygon boundary as the supersample factor
    grows, which is what lets a half-maximum-intensity `find_contours` on
    the rendered frame recover the ground-truth boundary.

    Args:
        polygon: Ground-truth polygon in full-resolution pixel coordinates.
        size: ``(height, width)`` of the output (full-resolution) frame.

    Returns:
        Float array, shape ``size``, values in ``[0, 1]``.
    """
    height, width = size
    hi_shape = (height * _SUPERSAMPLE, width * _SUPERSAMPLE)
    xs, ys = polygon.exterior.coords.xy
    rows = np.asarray(ys) * _SUPERSAMPLE
    cols = np.asarray(xs) * _SUPERSAMPLE
    rr, cc = _sk_polygon(rows, cols, shape=hi_shape)
    hi = np.zeros(hi_shape, dtype=np.float64)
    hi[rr, cc] = 1.0
    coverage: np.ndarray = downscale_local_mean(hi, (_SUPERSAMPLE, _SUPERSAMPLE))
    return coverage


def _touches_border(polygon: Polygon, width: int, height: int) -> bool:
    """True if ``polygon``'s bounding box reaches or crosses the frame edge."""
    minx, miny, maxx, maxy = polygon.bounds
    return bool(minx <= 0 or miny <= 0 or maxx >= width or maxy >= height)


def make_frame(
    size: tuple[int, int] = (512, 512),
    objects: list[dict[str, Any]] | None = None,
    gradient: float = 0.0,
    noise_sigma: float = 0.0,
    seed: int = 0,
) -> SynthFrame:
    """Render a synthetic backlit-silhouette frame with known ground truth.

    Each object is rasterized on its own full-resolution intensity layer
    (bright background, dark anti-aliased silhouette); objects with
    ``blur_sigma > 0`` get an additional `scipy.ndimage.gaussian_filter`
    applied to that layer (simulated defocus) before compositing. Layers
    are composited by taking the per-pixel minimum (darkest wins). The
    multiplicative illumination gradient and Gaussian noise are then
    applied to the composited frame as a whole.

    Args:
        size: ``(height, width)`` of the output frame, px.
        objects: List of object dicts, each
            ``{"kind": "ellipse"|"rect"|"blob", "cx", "cy", "a", "b",
            "angle", "blur_sigma"}``. ``angle`` is radians and
            ``blur_sigma`` is px (both default to 0.0 if omitted).
            ``None``/omitted renders an empty (background-only) frame.
        gradient: Multiplicative illumination gradient strength; frame
            columns are scaled by ``1 - gradient * (x / width)``.
        noise_sigma: Standard deviation (intensity units) of additive
            Gaussian read noise; no noise added if ``0.0``.
        seed: Seed for `numpy.random.default_rng`, controlling blob shape
            randomness and noise. Same seed -> identical frame.

    Returns:
        `SynthFrame` with the rendered ``uint8`` image and per-object
        ground truth.
    """
    height, width = size
    rng = np.random.default_rng(seed)

    composite = np.full((height, width), _BRIGHT, dtype=np.float64)
    synth_objects: list[SynthObject] = []
    for spec in objects or []:
        polygon = _object_polygon(spec, rng)
        blur_sigma = float(spec.get("blur_sigma", 0.0))

        coverage = _rasterize_coverage(polygon, (height, width))
        layer = _BRIGHT - coverage * (_BRIGHT - _DARK)
        if blur_sigma > 0:
            layer = ndimage.gaussian_filter(layer, sigma=blur_sigma)
        composite = np.minimum(composite, layer)

        synth_objects.append(
            SynthObject(
                polygon=polygon,
                blurred=blur_sigma > 0,
                touches_border=_touches_border(polygon, width, height),
            )
        )

    if gradient:
        x = np.arange(width, dtype=np.float64)
        column_factor = 1.0 - gradient * (x / width)
        composite = composite * column_factor[np.newaxis, :]

    if noise_sigma > 0:
        composite = composite + rng.normal(0.0, noise_sigma, size=composite.shape)

    image = np.clip(composite, 0, 255).astype(np.uint8)
    return SynthFrame(image=image, objects=synth_objects, background_level=_BRIGHT)
