"""Per-frame QC-colored polygon-overlay visualization (`make_overlays`).

Split out of `report.py` (which the design doc's ~250-line-per-module
guideline flagged once this grew past it): unlike everything in
`report.py`, this is a separate, explicitly-frame-selected visualization
rather than part of `make_reports`'s always-run set. For each requested
frame, it draws every detected grain's subpixel polygon outline over its
full-resolution source image -- green for QC-accepted, red for rejected --
then downsamples the whole composited raster by an integer `factor`.

**Full-res-then-downsample, not the other way around** (see `make_overlays`'
docstring for the full rationale): outlines are drawn directly into the
frame's own full-resolution pixel grid *before* the combined raster is
downsampled, so registration between image and outline is exact by
construction rather than something that has to be separately verified.

Duplicates `_read_grayscale` and `_resolve_frame_paths` from `report.py`
rather than importing them, matching this codebase's existing convention of
duplicating small private cross-cutting helpers across modules (e.g.
`report._read_grayscale` itself duplicates `pipeline._read_grayscale`)
instead of exposing them as shared public API.

Uses matplotlib with the non-interactive `Agg` backend so it renders
headless (CI, a server with no display).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import imageio.v3 as iio
import matplotlib
import numpy as np
import pandas as pd
import shapely.wkt
from scipy import ndimage
from shapely.geometry import Polygon
from skimage.draw import polygon_perimeter
from skimage.transform import rescale

if not os.environ.get("MPLBACKEND"):
    # Default to the headless backend (overlays are written to disk, never shown),
    # but let an explicit MPLBACKEND win -- importing this module must not clobber
    # the interactive backend that `groundtruth` needs. Must precede the
    # `matplotlib.pyplot` import to take effect.
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

from grain_morph.config import Config

_LOGGER = logging.getLogger(__name__)

_DPI = 150  # resolution figures are saved at

# --- Overlay rendering constants ------------------------------------------

# Outline colors, `(R, G, B)` uint8 -- drawn straight into the composited
# raster (see `_compose_outlines_full_res`), so they must be pure hue (not
# grayscale) to stay high-contrast against a grayscale frame regardless of
# its own intensity: any pixel with R != G != B reads as colored even on a
# dark or bright background.
_ACCEPTED_COLOR: tuple[int, int, int] = (0, 255, 0)  # green: qc_pass True
_REJECTED_COLOR: tuple[int, int, int] = (255, 0, 0)  # red: qc_pass False

# `matplotlib` equivalents (0-1 floats) of the two colors above, for the
# annotation layer (grain_uid labels) so label color matches outline color.
_ACCEPTED_COLOR_MPL = tuple(c / 255 for c in _ACCEPTED_COLOR)
_REJECTED_COLOR_MPL = tuple(c / 255 for c in _REJECTED_COLOR)

# Centroid marker for `flag_no_polygon` grains (no polygon to trace an
# outline from) -- a third, distinct hue so it's never confused with an
# accepted/rejected outline.
_NO_POLYGON_MARKER_COLOR_MPL = "gold"
_NO_POLYGON_MARKER_COLOR_RGB: tuple[int, int, int] = (255, 215, 0)  # matplotlib "gold"

# Minimum `ndimage.binary_dilation` iterations applied to each full-res
# outline mask before downsampling, so a geometrically 1px-wide polygon
# perimeter survives anti-aliased downsampling by `factor` as a visible
# line rather than being smeared down to near-background intensity in most
# downsampled pixels. Scaled with `factor` itself (see
# `_thicken_outline_mask`) -- floored at 1 so `factor=1` (no downsampling)
# still gets a couple of dilated pixels of visible line width.
_MIN_OUTLINE_DILATION_ITER = 1

_OVERLAY_FIGSIZE = (8.0, 8.0)  # inches, annotation-layer figure

# `grain_uid`-suffix label / centroid-marker styling.
_OVERLAY_LABEL_FONTSIZE = 6
_OVERLAY_MARKER_SIZE = 10.0
_OVERLAY_TITLE_FONTSIZE = 10

# Per-grain text-label cap for the `labels=None` ("auto") tri-state: above
# this many grains on one frame, per-grain `grain_uid`-suffix text labels
# are suppressed by default -- a real-data review at 556 grains/frame found
# the unconditional per-grain labels rendered an unreadable mess. Outlines,
# `flag_no_polygon` markers, and the accepted/rejected title counts are
# always drawn regardless of this cap; only the small per-grain text labels
# are affected. `labels=True`/`labels=False` (CLI `--labels`/`--no-labels`)
# override the cap in either direction.
_LABEL_COUNT_CAP = 50


def _read_grayscale(path: Path) -> np.ndarray:
    """Read an image frame as a 2D `uint8` grayscale array.

    Mirrors `grain_morph.pipeline._read_grayscale` (and `report.
    _read_grayscale`); duplicated rather than imported since that is a
    private helper of another module (see module docstring).

    Args:
        path: Path to the image file.

    Returns:
        2D `uint8` array. If the file decodes with an extra channel axis,
        only the first channel is kept.
    """
    array = np.asarray(iio.imread(path))
    if array.ndim == 3:
        array = array[..., 0]
    return array.astype(np.uint8)


def _resolve_frame_paths(grains: pd.DataFrame, frames_root: Path) -> dict[str, Path]:
    """Map each unique `frame_path` value to a readable file on disk.

    Mirrors `report._resolve_frame_paths`; duplicated rather than imported
    (see module docstring). Tries the stored path as-is, then `frames_root
    / <filename>`, then a recursive search under `frames_root` for a
    same-named file. Values that resolve to nothing are simply absent from
    the returned mapping.

    Args:
        grains: Per-grain table. Returns `{}` immediately if it has no
            `frame_path` column at all.
        frames_root: Directory to search for frames.

    Returns:
        `{frame_path_value: resolved_path}` for every value that resolved
        to an existing file.
    """
    if "frame_path" not in grains.columns:
        return {}
    resolved: dict[str, Path] = {}
    for value in grains["frame_path"].dropna().unique():
        candidate = Path(str(value))
        if candidate.is_file():
            resolved[value] = candidate
            continue
        alt = frames_root / candidate.name
        if alt.is_file():
            resolved[value] = alt
            continue
        match = next(frames_root.rglob(candidate.name), None)
        if match is not None:
            resolved[value] = match
    return resolved


def _resolve_frame_path_for_frame(
    frame_grains: pd.DataFrame, frames_root: Path, frame_id: str
) -> Path | None:
    """Resolve one frame's image path, for the single `frame_id` it belongs to.

    Tries `_resolve_frame_paths` first (the same `frame_path`-value-based
    resolution `make_reports` uses), then falls back to `frames_root /
    f"{frame_id}.bmp"` -- covers a `grains` table with no usable
    `frame_path` value at all for this frame.

    Args:
        frame_grains: Grains already filtered to this one `frame_id`.
        frames_root: Directory to search for frames.
        frame_id: This frame's id (stem), for the `.bmp` fallback name.

    Returns:
        A resolved, existing path, or `None` if nothing resolved.
    """
    lookup = _resolve_frame_paths(frame_grains, frames_root)
    if lookup:
        return next(iter(lookup.values()))
    fallback = frames_root / f"{frame_id}.bmp"
    return fallback if fallback.is_file() else None


def _grain_uid_suffix(grain_uid: str) -> str:
    """The trailing `label` component of a `grain_uid` (`f"{frame_id}:{label}"`).

    Args:
        grain_uid: Full grain identifier (`pipeline._build_row`'s
            `f"{frame_id}:{label}"` convention).

    Returns:
        Everything after the last `:`, or the whole string if there is none.
    """
    return grain_uid.rsplit(":", 1)[-1]


def _should_draw_labels(labels: bool | None, n_grains: int) -> bool:
    """Resolve the tri-state `labels` option into a concrete on/off decision.

    Args:
        labels: `None` for "auto" (suppress once `n_grains` exceeds
            `_LABEL_COUNT_CAP`), or an explicit `True`/`False` to force
            per-grain text labels on/off regardless of `n_grains`.
        n_grains: Number of grains on this frame's overlay.

    Returns:
        Whether `_annotate` should draw per-grain text labels.
    """
    if labels is None:
        return n_grains <= _LABEL_COUNT_CAP
    return labels


def _polygon_perimeter_mask(polygon: Polygon, shape: tuple[int, int]) -> np.ndarray:
    """A full-res boolean mask of one polygon's exterior boundary pixels.

    WKT/shapely coordinates are `(x, y)` = `(col, row)`; `skimage.draw.
    polygon_perimeter` wants `(row, col)` arrays, so the axes are swapped
    here (`rows = y = coord[1]`, `cols = x = coord[0]`).

    Args:
        polygon: Grain outline, in full-resolution pixel coordinates.
        shape: `(height, width)` of the full-resolution frame -- perimeter
            pixels are clipped to this extent (`polygon_perimeter(...,
            clip=True)`).

    Returns:
        Boolean mask, `shape`, `True` along the polygon's boundary pixels.
    """
    xs, ys = polygon.exterior.coords.xy
    rows, cols = np.asarray(ys), np.asarray(xs)
    rr, cc = polygon_perimeter(rows, cols, shape=shape, clip=True)
    mask = np.zeros(shape, dtype=bool)
    mask[rr, cc] = True
    return mask


def _thicken_outline_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    """Dilate a 1px-wide outline mask so it survives downsampling by `factor`.

    A geometrically 1px-wide polygon perimeter, anti-aliased-downsampled by
    `factor`, contributes only a `1/factor` fraction of its color to most
    downsampled pixels -- often diluted down to visually indistinguishable
    from the background. Dilating the mask by `factor` px first means every
    downsampled pixel the true boundary passes through has a fully-colored
    full-res neighborhood behind it, so the line survives as a solid,
    visible color instead of fading out.

    Args:
        mask: Boolean perimeter mask (`_polygon_perimeter_mask`).
        factor: The downsample factor this outline needs to survive.

    Returns:
        A dilated copy of `mask`.
    """
    iterations = max(_MIN_OUTLINE_DILATION_ITER, factor)
    return ndimage.binary_dilation(mask, iterations=iterations)


def _compose_outlines_full_res(
    image: np.ndarray, grain_rows: pd.DataFrame, contours: pd.DataFrame, factor: int
) -> np.ndarray:
    """Draw every grain's QC-colored polygon outline into a full-res RGB frame.

    This is the step that keeps outlines exactly registered to the image
    (see `make_overlays`' docstring): everything here happens at the
    frame's native resolution, straight from the WKT polygons' own
    coordinates -- only the *result* is downsampled afterward
    (`_downsample_rgb`), never the polygon coordinates themselves.

    Args:
        image: Full-resolution grayscale frame.
        grain_rows: This frame's grains (already filtered to one
            `frame_id`); needs `grain_uid`, `qc_pass`.
        contours: `grain_uid, wkt, um_per_px` rows for (a superset of)
            `grain_rows`. A grain with no matching row here (e.g. a
            `flag_no_polygon` grain) contributes no outline -- not an
            error.
        factor: Downsample factor this composite will later be reduced by;
            outlines are dilated proportionally so they survive it (see
            `_thicken_outline_mask`).

    Returns:
        `uint8` RGB array, shape `(*image.shape, 3)`.
    """
    rgb = np.stack([image, image, image], axis=-1).astype(np.uint8)
    shape = (image.shape[0], image.shape[1])

    wkt_by_uid = (
        contours.set_index("grain_uid")["wkt"]
        if not contours.empty and "grain_uid" in contours.columns
        else pd.Series(dtype=str)
    )

    accepted_mask = np.zeros(shape, dtype=bool)
    rejected_mask = np.zeros(shape, dtype=bool)
    for _, row in grain_rows.iterrows():
        wkt_value = wkt_by_uid.get(row["grain_uid"])
        if wkt_value is None or pd.isna(wkt_value):
            continue  # no polygon for this grain (e.g. flag_no_polygon) -- nothing to trace
        try:
            polygon = shapely.wkt.loads(wkt_value)
        except Exception:
            # A malformed WKT string for one grain shouldn't take down the
            # whole overlay -- same never-crash contract as an unresolvable
            # frame or a missing contour row (see `make_overlays`'
            # docstring).
            _LOGGER.warning(
                "overlay: grain %r has malformed WKT -- skipping its outline",
                row["grain_uid"],
            )
            continue
        perimeter_mask = _polygon_perimeter_mask(polygon, shape)
        if bool(row.get("qc_pass", False)):
            accepted_mask |= perimeter_mask
        else:
            rejected_mask |= perimeter_mask

    if accepted_mask.any():
        rgb[_thicken_outline_mask(accepted_mask, factor)] = _ACCEPTED_COLOR
    if rejected_mask.any():
        rgb[_thicken_outline_mask(rejected_mask, factor)] = _REJECTED_COLOR

    return rgb


def _downsample_rgb(rgb: np.ndarray, factor: int) -> np.ndarray:
    """Anti-aliased-downsample a composited RGB raster by an integer factor.

    Args:
        rgb: `uint8` RGB array (image + outlines already composited in by
            `_compose_outlines_full_res`).
        factor: Integer downsample factor; `<= 1` returns `rgb` unchanged.

    Returns:
        `uint8` RGB array, `1/factor` the height/width of `rgb`.
    """
    if factor <= 1:
        return rgb
    scaled = rescale(rgb, 1.0 / factor, anti_aliasing=True, channel_axis=-1, preserve_range=True)
    return np.clip(np.round(scaled), 0, 255).astype(np.uint8)


def _annotate(
    rgb_small: np.ndarray,
    grain_rows: pd.DataFrame,
    factor: int,
    frame_id: str,
    out_path: Path,
    labels: bool | None,
) -> None:
    """Add a title, per-grain labels, and no-polygon markers, then save.

    Runs entirely on the already-downsampled, already-outlined `rgb_small`
    -- text/markers don't need full-resolution registration, so adding them
    last (after downsampling, not before) costs nothing and keeps this step
    cheap regardless of the source frame's resolution.

    Args:
        rgb_small: Downsampled RGB raster (`_downsample_rgb`), with polygon
            outlines already composited in.
        grain_rows: This frame's grains (already filtered to `frame_id`);
            needs `qc_pass`, `grain_uid`; `centroid_x`/`centroid_y` and
            `flag_no_polygon` are used when present.
        factor: The downsample factor `rgb_small` was produced at -- used
            to map each grain's full-res centroid into `rgb_small`'s
            coordinate system.
        frame_id: This frame's id, used in the title.
        out_path: Destination PNG path.
        labels: Tri-state per-grain text-label control, resolved via
            `_should_draw_labels`. Only the small per-grain text labels are
            affected -- polygon outlines (already baked into `rgb_small`),
            `flag_no_polygon` markers, and the accepted/rejected title
            counts are always drawn regardless.
    """
    n_accepted = int(grain_rows["qc_pass"].astype(bool).sum())
    n_rejected = len(grain_rows) - n_accepted
    draw_labels = _should_draw_labels(labels, len(grain_rows))

    fig, ax = plt.subplots(figsize=_OVERLAY_FIGSIZE)
    ax.imshow(rgb_small)
    ax.set_title(
        f"{frame_id}  (accepted={n_accepted}, rejected={n_rejected})",
        fontsize=_OVERLAY_TITLE_FONTSIZE,
    )
    ax.axis("off")

    for _, row in grain_rows.iterrows():
        cx, cy = row.get("centroid_x"), row.get("centroid_y")
        if pd.isna(cx) or pd.isna(cy):
            continue
        sx, sy = float(cx) / factor, float(cy) / factor
        accepted = bool(row.get("qc_pass", False))
        color = _ACCEPTED_COLOR_MPL if accepted else _REJECTED_COLOR_MPL

        if bool(row.get("flag_no_polygon", False)):
            # No geometry to outline -- mark the centroid instead so the
            # grain is still visible on the overlay.
            ax.plot(
                sx,
                sy,
                marker="X",
                markersize=_OVERLAY_MARKER_SIZE,
                color=_NO_POLYGON_MARKER_COLOR_MPL,
                linestyle="None",
            )

        if draw_labels:
            label = _grain_uid_suffix(str(row.get("grain_uid", "?")))
            ax.text(sx, sy, label, fontsize=_OVERLAY_LABEL_FONTSIZE, color=color)

    fig.tight_layout()
    fig.savefig(out_path, dpi=_DPI)
    plt.close(fig)


def make_overlays(
    grains: pd.DataFrame,
    contours: pd.DataFrame,
    frames_root: Path,
    out_dir: Path,
    frame_ids: list[str],
    factor: int,
    cfg: Config,
    labels: bool | None = None,
) -> list[Path]:
    """Render one QC-colored polygon-overlay PNG per explicitly requested frame.

    For each `fid` in `frame_ids`, draws every one of that frame's detected
    grains' subpixel polygon outlines over its full-resolution source
    image -- green for accepted (`qc_pass` `True`), red for rejected --
    then downsamples the whole composited raster by `factor` and adds a
    title plus (usually) per-grain `grain_uid`-suffix labels.

    **Full-res-then-downsample, not the other way around:** outlines are
    drawn directly into the frame's own full-resolution pixel grid
    (`_compose_outlines_full_res`) *before* the single combined raster is
    downsampled (`_downsample_rgb`). Separately downsampling the image and
    scaling the polygon coordinates would risk the two ending up
    sub-pixel-misregistered (rounding in the coordinate scaling doesn't
    have to match the image resampling's own rounding); drawing both into
    one raster before it is ever resampled makes exact registration a
    consequence of the construction, not something that has to be
    separately verified.

    Never raises for a per-frame or per-grain problem -- an `fid` with no
    grains, an unresolvable/unreadable frame image, a grain with no
    matching contour row, or a grain with malformed WKT all degrade to
    skipping just that frame/grain (with a logged warning) rather than
    failing the whole call. `cfg` is accepted for interface symmetry with
    `report.make_reports`, but not otherwise consumed here: accepted/
    rejected coloring uses `grains["qc_pass"]` exactly as `detect`
    persisted it (`grain_morph.qc.qc_flags`), not a coloring recomputed
    from `cfg.qc.disqualifying_flags`.

    Args:
        grains: Per-grain table; needs `frame_id`, `grain_uid`, `qc_pass`,
            `centroid_x`, `centroid_y`; `frame_path` and `flag_no_polygon`
            are used when present.
        contours: `grain_uid, wkt, um_per_px` rows (e.g. the concatenated
            `contours/*.<ext>` files from a `detect` run). May be missing
            rows for some grains (e.g. `flag_no_polygon` ones).
        frames_root: Directory to search for the frames (see
            `_resolve_frame_paths`); also the base directory for the
            `f"{fid}.bmp"` fallback used when no `frame_path` value
            resolves.
        out_dir: Destination directory for every PNG (created if missing).
        frame_ids: Frame ids (stems) to render -- explicit, never inferred;
            a caller wanting "every frame" must list them all.
        factor: Integer factor the full-res composite is downsampled by.
        cfg: Resolved pipeline configuration (see docstring above for why
            it's currently unused).
        labels: Tri-state per-grain text-label control -- `None` (default,
            "auto") suppresses labels on any frame with more than
            `_LABEL_COUNT_CAP` grains (outlines, `flag_no_polygon`
            markers, and title counts are unaffected by the cap);
            `True`/`False` force labels on/off regardless of grain count.

    Returns:
        Paths of every `{frame_id}_overlay.png` actually written, in
        `frame_ids` order, omitting any frame that was skipped.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_root = Path(frames_root)

    written: list[Path] = []
    for fid in frame_ids:
        frame_grains = grains[grains["frame_id"] == fid]
        if frame_grains.empty:
            _LOGGER.warning("overlay: frame_id %r has no grains -- skipping", fid)
            continue

        frame_path = _resolve_frame_path_for_frame(frame_grains, frames_root, fid)
        if frame_path is None:
            _LOGGER.warning(
                "overlay: frame_id %r has no resolvable frame image under %s -- skipping",
                fid,
                frames_root,
            )
            continue

        try:
            image = _read_grayscale(frame_path)
        except Exception:
            _LOGGER.warning(
                "overlay: frame_id %r's image at %s could not be read -- skipping",
                fid,
                frame_path,
            )
            continue

        frame_contours = (
            contours[contours["grain_uid"].isin(frame_grains["grain_uid"])]
            if not contours.empty
            else contours
        )

        rgb_full = _compose_outlines_full_res(image, frame_grains, frame_contours, factor)
        rgb_small = _downsample_rgb(rgb_full, factor)

        out_path = out_dir / f"{fid}_overlay.png"
        _annotate(rgb_small, frame_grains, factor, fid, out_path, labels)
        written.append(out_path)

    return written
