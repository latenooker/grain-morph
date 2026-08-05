"""QC review report artifacts: figures a human skims to sanity-check a run.

Four kinds of artifact (design doc §7 "QC review artifacts"), by trust level:

1. **Contact sheets** -- grain crops tiled with `grain_uid` + key metrics:
   one paginated set per `flag_*` column, restricted to grains that were
   actually rejected (`qc_pass` is `False`), plus one stratified-by-ECD
   sample of accepted grains. The single most trust-building output: a
   reviewer can eyeball whether the QC gate is rejecting the right things.
2. `focus_scatter.png` -- `edge_gradient_norm` vs `contrast`, colored by
   `flag_defocus`, with a threshold line (see `_focus_scatter`'s docstring
   for why only `defocus_contrast_min`, not `defocus_edge_width_px`, is
   drawn).
3. `rejection_vs_ecd.png` -- rejection rate (`~qc_pass`), binned by
   `ecd_um`, as a bar chart.
4. `illumination_field.png` -- a background-illumination estimate for one
   frame, so vignetting drift is visible at a glance.

Every helper degrades gracefully rather than raising: `make_reports` is the
last stage of a pipeline an operator may run against a partial or relocated
dataset -- e.g. a per-grain table shipped without its source frames, as in
`tests/test_report.py` -- so a missing column or an unresolvable frame skips
just that one artifact instead of failing the whole report. Only
`focus_scatter.png` and `rejection_vs_ecd.png` need nothing but the
per-grain table itself, so those two are always produced.

Uses matplotlib with the non-interactive `Agg` backend so it renders
headless (CI, a server with no display).

A fifth artifact, `{frame_id}_overlay.png` (`make_overlays`), is a separate,
explicitly-frame-selected visualization rather than part of `make_reports`'s
always-run set: it draws each detected grain's subpixel polygon outline over
its full-resolution source frame, colored by QC outcome, then downsamples
the whole composited raster by an integer `factor` (see `make_overlays`'
docstring for why outlines are drawn at full resolution before downsampling,
not the other way around).
"""

from __future__ import annotations

import logging
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

matplotlib.use("Agg")  # must precede `import matplotlib.pyplot` to take effect

import matplotlib.pyplot as plt

from grain_morph.config import Config

_LOGGER = logging.getLogger(__name__)

# --- Figure rendering constants ------------------------------------------

_DPI = 150  # resolution figures are saved at
_FIGSIZE_SCATTER = (7.0, 5.5)  # inches
_FIGSIZE_BAR = (8.0, 5.0)
_FIGSIZE_FIELD = (6.0, 5.0)
_FIGSIZE_CONTACT_SHEET = (12.0, 9.0)

_SCATTER_POINT_SIZE = 12.0
_SCATTER_ALPHA = 0.6

_ILLUM_CMAP = "viridis"

# --- Contact-sheet layout/sampling constants ------------------------------

_CONTACT_SHEET_MAX_N = 24  # grains per contact-sheet page
_CONTACT_SHEET_NCOLS = 6  # tile grid columns

# Crop half-width = ecd_px * this factor, floored at `_CROP_MIN_HALF_PX` --
# a crop a bit larger than the grain itself so its edge/background context
# is visible in the tile, not just the silhouette.
_CROP_HALF_FACTOR = 1.0
_CROP_MIN_HALF_PX = 20.0

_N_ACCEPTED_STRATA = 5  # ECD quantile strata the accepted sheet samples across
_ACCEPTED_SAMPLE_SEED = 0  # deterministic sampling across repeated runs

# Bin edges (µm) `_rejection_vs_ecd` groups `ecd_um` into. Mirrors
# `grain_morph.aggregate._ECD_BIN_EDGES_UM` (Wentworth (1922) grain-size
# class boundaries); duplicated rather than imported, since that tuple is a
# private module attribute of `aggregate` (same convention as `pipeline.
# _EXTENSIONS` mirroring `writers._EXTENSIONS`). Keeping these identical to
# `aggregate`'s own `rejection_by_ecd` table means the two "rejection vs
# size" views agree with each other.
_ECD_BIN_EDGES_UM: tuple[float, ...] = (
    0.0,
    62.5,
    125.0,
    250.0,
    500.0,
    1000.0,
    2000.0,
    4000.0,
    float("inf"),
)

# --- Overlay (`make_overlays`) rendering constants ------------------------

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


def _read_grayscale(path: Path) -> np.ndarray:
    """Read an image frame as a 2D `uint8` grayscale array.

    Mirrors `grain_morph.pipeline._read_grayscale`; duplicated rather than
    imported since that is a private helper of `pipeline`.

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

    Tries the stored path as-is, then `frames_root / <filename>`, then a
    recursive search under `frames_root` for a same-named file -- a
    per-grain table's `frame_path` records wherever the frame lived at
    detect time, which may not be this machine's `frames_root` (a report
    can legitimately be built from a per-grain table shipped without, or
    relocated from, its source frames). Values that resolve to nothing are
    simply absent from the returned mapping.

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


def _focus_scatter(grains: pd.DataFrame, out_dir: Path, cfg: Config) -> Path:
    """Scatter of `edge_gradient_norm` vs `contrast`, colored by `flag_defocus`.

    Only the `contrast` threshold (`cfg.qc.defocus_contrast_min`) is drawn
    as a line. The defocus gate's other threshold, `cfg.qc.
    defocus_edge_width_px`, is expressed in `edge_width_px` units (a
    10-90% intensity-rise distance, px) -- a different metric than the
    normalized-gradient x-axis plotted here, and one this report's
    per-grain table is not guaranteed to carry (see the module docstring)
    -- so drawing it on this axis would imply a decision boundary that
    doesn't correspond to anything QC actually applied.

    Args:
        grains: Per-grain table; needs `edge_gradient_norm`, `contrast`,
            `flag_defocus`.
        out_dir: Destination directory.
        cfg: Resolved pipeline configuration; consumes
            `cfg.qc.defocus_contrast_min`.

    Returns:
        `out_dir / "focus_scatter.png"`.
    """
    flagged = grains["flag_defocus"].astype(bool)
    fig, ax = plt.subplots(figsize=_FIGSIZE_SCATTER)
    ax.scatter(
        grains.loc[~flagged, "edge_gradient_norm"],
        grains.loc[~flagged, "contrast"],
        s=_SCATTER_POINT_SIZE,
        alpha=_SCATTER_ALPHA,
        c="tab:blue",
        label="in focus",
    )
    ax.scatter(
        grains.loc[flagged, "edge_gradient_norm"],
        grains.loc[flagged, "contrast"],
        s=_SCATTER_POINT_SIZE,
        alpha=_SCATTER_ALPHA,
        c="tab:red",
        label="flag_defocus",
    )
    ax.axhline(
        cfg.qc.defocus_contrast_min,
        color="black",
        linestyle="--",
        linewidth=1,
        label="defocus_contrast_min",
    )
    ax.set_xlabel("edge_gradient_norm")
    ax.set_ylabel("contrast")
    ax.set_title("Focus scatter")
    # flag_defocus is an OR of the contrast threshold plotted here and a
    # second, unplotted edge_width_px threshold (see this function's
    # docstring), so a red point can legitimately sit above the threshold
    # line -- called out so that doesn't read as a plotting bug.
    ax.text(
        0.01,
        0.01,
        "flag_defocus may also trigger on edge_width_px (not plotted)",
        transform=ax.transAxes,
        fontsize=7,
        color="dimgray",
        va="bottom",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = out_dir / "focus_scatter.png"
    fig.savefig(out_path, dpi=_DPI)
    plt.close(fig)
    return out_path


def _ecd_bin_labels(edges: tuple[float, ...]) -> list[str]:
    """Human-readable `"lo-hi"` (or `"lo+"` for the open top bin) labels.

    Mirrors `grain_morph.aggregate._ecd_bin_labels` (see the module-level
    duplication note on `_ECD_BIN_EDGES_UM`).

    Args:
        edges: Ascending bin edges (µm), last one typically `inf`.

    Returns:
        `len(edges) - 1` labels, one per bin, in edge order.
    """
    return [
        f"{lo:g}+" if np.isinf(hi) else f"{lo:g}-{hi:g}"
        for lo, hi in zip(edges[:-1], edges[1:], strict=True)
    ]


def _rejection_vs_ecd(grains: pd.DataFrame, out_dir: Path) -> Path:
    """Bar chart: rejection rate (`~qc_pass`) binned by `ecd_um`.

    Args:
        grains: Per-grain table; needs `ecd_um`, `qc_pass`.
        out_dir: Destination directory.

    Returns:
        `out_dir / "rejection_vs_ecd.png"`.
    """
    labels = _ecd_bin_labels(_ECD_BIN_EDGES_UM)
    bins = pd.cut(
        grains["ecd_um"], bins=_ECD_BIN_EDGES_UM, labels=labels, right=False, include_lowest=True
    )
    rejection_rate = grains.groupby(bins, observed=True)["qc_pass"].apply(
        lambda s: 1.0 - s.astype(bool).mean()
    )
    # Keep only bins that actually have grains, in bin order.
    present = [label for label in labels if label in rejection_rate.index]
    rejection_rate = rejection_rate.reindex(present)

    fig, ax = plt.subplots(figsize=_FIGSIZE_BAR)
    ax.bar(rejection_rate.index.astype(str), rejection_rate.to_numpy())
    ax.set_xlabel("ECD bin (µm)")
    ax.set_ylabel("rejection rate")
    ax.set_title("Rejection rate vs ECD")
    ax.set_ylim(0, 1)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    out_path = out_dir / "rejection_vs_ecd.png"
    fig.savefig(out_path, dpi=_DPI)
    plt.close(fig)
    return out_path


def _crop_half_px(row: pd.Series) -> float:
    """Crop half-width (px) for one grain's contact-sheet tile.

    Sized from the grain's own `ecd_um`/`um_per_px` when both are present
    (so bigger grains get bigger crops), floored at `_CROP_MIN_HALF_PX` so
    small or unsized grains still get a usable tile.

    Args:
        row: One row of the per-grain table.

    Returns:
        Half-width, px.
    """
    ecd_um = row.get("ecd_um")
    um_per_px = row.get("um_per_px")
    if pd.notna(ecd_um) and pd.notna(um_per_px) and um_per_px:
        ecd_px = float(ecd_um) / float(um_per_px)
        return max(ecd_px * _CROP_HALF_FACTOR, _CROP_MIN_HALF_PX)
    return _CROP_MIN_HALF_PX


def _crop_grain(image: np.ndarray, row: pd.Series) -> np.ndarray | None:
    """A square crop of `image` centered on one grain's centroid.

    Args:
        image: Full grayscale frame the grain was detected in.
        row: One row of the per-grain table; needs `centroid_x`,
            `centroid_y` (pixel coordinates, `(col, row)`).

    Returns:
        The cropped array, or `None` if `centroid_x`/`centroid_y` are
        missing/NaN or the resulting crop would be empty.
    """
    cx, cy = row.get("centroid_x"), row.get("centroid_y")
    if pd.isna(cx) or pd.isna(cy):
        return None
    half = _crop_half_px(row)
    height, width = image.shape[:2]
    row0 = int(max(0, cy - half))
    row1 = int(min(height, cy + half))
    col0 = int(max(0, cx - half))
    col1 = int(min(width, cx + half))
    if row1 <= row0 or col1 <= col0:
        return None
    return image[row0:row1, col0:col1]


def _grain_label(row: pd.Series) -> str:
    """A short caption for one grain's contact-sheet tile.

    Args:
        row: One row of the per-grain table.

    Returns:
        `"{grain_uid}\\necd={ecd_um:.0f}um"`, degrading to just `grain_uid`
        if `ecd_um` is absent.
    """
    uid = str(row.get("grain_uid", "?"))
    ecd_um = row.get("ecd_um")
    if pd.notna(ecd_um):
        return f"{uid}\n{float(ecd_um):.0f}µm"
    return uid


def _render_contact_sheet(
    rows: pd.DataFrame, frame_lookup: dict[str, Path], title: str, out_path: Path
) -> bool:
    """Render one contact-sheet page: a grid of grain crops with captions.

    Each distinct resolved frame is read at most once, even though several
    rows may share one `frame_path`.

    Args:
        rows: Up to `_CONTACT_SHEET_MAX_N` grains for this page.
        frame_lookup: `{frame_path_value: resolved_path}`
            (`_resolve_frame_paths`).
        title: Figure title (e.g. `"Rejected: flag_defocus"`).
        out_path: Destination PNG path.

    Returns:
        `True` if the sheet was written (at least one crop succeeded);
        `False` if every row's frame/crop was unavailable, in which case
        nothing is written.
    """
    frame_cache: dict[str, np.ndarray] = {}
    tiles: list[tuple[np.ndarray, str]] = []
    for _, row in rows.iterrows():
        path = frame_lookup.get(row.get("frame_path"))
        if path is None:
            continue
        if str(path) not in frame_cache:
            frame_cache[str(path)] = _read_grayscale(path)
        crop = _crop_grain(frame_cache[str(path)], row)
        if crop is None or crop.size == 0:
            continue
        tiles.append((crop, _grain_label(row)))

    if not tiles:
        return False

    ncols = _CONTACT_SHEET_NCOLS
    nrows = -(-len(tiles) // ncols)  # ceil division
    fig, axes = plt.subplots(nrows, ncols, figsize=_FIGSIZE_CONTACT_SHEET, squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (crop, label) in zip(axes.flat, tiles, strict=False):
        ax.imshow(crop, cmap="gray")
        ax.set_title(label, fontsize=6)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=_DPI)
    plt.close(fig)
    return True


def _paginate(rows: pd.DataFrame, page_size: int) -> list[pd.DataFrame]:
    """Split `rows` into consecutive pages of at most `page_size` rows.

    Args:
        rows: Rows to paginate, in the order they'll be tiled.
        page_size: Max rows per page.

    Returns:
        `[]` if `rows` is empty; otherwise `ceil(len(rows) / page_size)`
        pages.
    """
    return [rows.iloc[i : i + page_size] for i in range(0, len(rows), page_size)]


def _contact_sheets_rejected(
    grains: pd.DataFrame, frame_lookup: dict[str, Path], out_dir: Path
) -> list[Path]:
    """One paginated contact-sheet set per `flag_*` column, rejected grains only.

    A grain is "rejected" here iff `qc_pass` is `False` for it -- a flag
    column set `True` on an otherwise-`qc_pass` grain (e.g. a non-
    disqualifying flag under the caller's config) is excluded, so a sheet
    titled `"Rejected: {flag}"` never shows a grain that was in fact
    accepted.

    Args:
        grains: Per-grain table; needs `qc_pass`.
        frame_lookup: `{frame_path_value: resolved_path}`.
        out_dir: Destination directory.

    Returns:
        Paths of every sheet actually written -- `[]` if `qc_pass` is
        absent; a flag with zero rejected+flagged rows, or whose frames
        are all unresolvable, contributes none.
    """
    if "qc_pass" not in grains.columns:
        return []
    rejected = ~grains["qc_pass"].astype(bool)
    paths: list[Path] = []
    flag_columns = [c for c in grains.columns if c.startswith("flag_")]
    for flag in flag_columns:
        flagged = grains[grains[flag].astype(bool) & rejected]
        if flagged.empty:
            continue
        for page_num, page in enumerate(_paginate(flagged, _CONTACT_SHEET_MAX_N), start=1):
            out_path = out_dir / f"contact_sheet_rejected_{flag}_{page_num:02d}.png"
            if _render_contact_sheet(page, frame_lookup, f"Rejected: {flag}", out_path):
                paths.append(out_path)
    return paths


def _stratified_sample(accepted: pd.DataFrame, cap: int, n_strata: int, seed: int) -> pd.DataFrame:
    """Up to `cap` rows of `accepted`, spread evenly across `ecd_um` strata.

    Falls back to treating all of `accepted` as one stratum if `ecd_um` has
    too little variation to form `n_strata` quantile bins.

    Args:
        accepted: Candidate rows (already filtered to `qc_pass`); needs
            `ecd_um`.
        cap: Maximum total rows returned.
        n_strata: Number of ECD quantile strata to spread the sample
            across.
        seed: RNG seed, for a reproducible sample across repeated runs.

    Returns:
        Up to `cap` rows of `accepted`.
    """
    if len(accepted) <= cap:
        return accepted
    rng = np.random.default_rng(seed)
    try:
        strata = pd.qcut(accepted["ecd_um"], n_strata, duplicates="drop")
        groups = [g for _, g in accepted.groupby(strata, observed=True)]
    except ValueError:
        groups = []
    if not groups:
        # `ecd_um` had too little variation to form any quantile bin --
        # `pd.qcut` degrades to an all-NaN categorical (zero groups) rather
        # than raising in this case, so this can't be caught above.
        groups = [accepted]
    per_stratum = max(1, cap // len(groups))
    parts = [
        group.loc[rng.choice(group.index, size=min(len(group), per_stratum), replace=False)]
        for group in groups
    ]
    return pd.concat(parts).iloc[:cap]


def _contact_sheets_accepted(
    grains: pd.DataFrame, frame_lookup: dict[str, Path], out_dir: Path
) -> list[Path]:
    """A stratified-by-ECD sample of accepted grains, as one contact sheet.

    Args:
        grains: Per-grain table; needs `qc_pass` and `ecd_um`.
        frame_lookup: `{frame_path_value: resolved_path}`.
        out_dir: Destination directory.

    Returns:
        `[out_dir / "contact_sheet_accepted.png"]` if the sheet was
        written, else `[]` (no accepted rows, or no resolvable frames).
    """
    if "qc_pass" not in grains.columns or "ecd_um" not in grains.columns:
        return []
    accepted = grains[grains["qc_pass"].astype(bool)]
    if accepted.empty:
        return []

    sample = _stratified_sample(
        accepted, _CONTACT_SHEET_MAX_N, _N_ACCEPTED_STRATA, _ACCEPTED_SAMPLE_SEED
    )
    out_path = out_dir / "contact_sheet_accepted.png"
    if _render_contact_sheet(sample, frame_lookup, "Accepted (stratified by ECD)", out_path):
        return [out_path]
    return []


def _illumination_field(frame_lookup: dict[str, Path], out_dir: Path, cfg: Config) -> Path | None:
    """Render a background-illumination estimate for one resolvable frame.

    Always uses the morphological (grey-closing) background estimator --
    the same technique `grain_morph.flatfield._estimate_field_morphological`
    uses, duplicated here since that is a private helper of `flatfield` --
    as a vignetting-diagnostic visualization, regardless of `cfg.flatfield.
    method`: `make_reports` only has `frame_path` strings, and is not
    reliably able to re-derive and read the paired blank frame a
    `"blank"`-method run would actually divide by (e.g. `frames_root` may
    hold this frame but not its blank, the exact partial-dataset case this
    module is built to tolerate).

    Args:
        frame_lookup: `{frame_path_value: resolved_path}`
            (`_resolve_frame_paths`); the first resolved path is used.
        out_dir: Destination directory.
        cfg: Resolved pipeline configuration; consumes
            `cfg.flatfield.morph_kernel_px`.

    Returns:
        `out_dir / "illumination_field.png"`, or `None` if no frame
        resolved.
    """
    if not frame_lookup:
        return None
    path = next(iter(frame_lookup.values()))
    image = _read_grayscale(path)
    # Odd-guard the kernel size, mirroring `flatfield._odd_guarded_kernel`
    # (also duplicated rather than imported): `grey_closing` wants an odd
    # structuring-element size for a centered footprint. A no-op at the
    # default `morph_kernel_px=201` against any realistically-sized frame;
    # only matters once `min(image.shape) - 1` actually clamps it down.
    kernel = min(cfg.flatfield.morph_kernel_px, min(image.shape) - 1)
    kernel = max(kernel, 1)
    if kernel % 2 == 0:
        kernel -= 1
    kernel = max(kernel, 1)
    field = ndimage.grey_closing(image.astype(np.float32), size=kernel)

    fig, ax = plt.subplots(figsize=_FIGSIZE_FIELD)
    mesh = ax.imshow(field, cmap=_ILLUM_CMAP)
    fig.colorbar(mesh, ax=ax, label="estimated background intensity")
    ax.set_title(f"Illumination field: {path.name}")
    ax.axis("off")
    out_path = out_dir / "illumination_field.png"
    fig.savefig(out_path, dpi=_DPI)
    plt.close(fig)
    return out_path


def make_reports(grains: pd.DataFrame, frames_root: Path, out_dir: Path, cfg: Config) -> list[Path]:
    """Render every QC review artifact `report.py` can produce for `grains`.

    Always produces `focus_scatter.png` and `rejection_vs_ecd.png` -- the
    two artifacts that need only the per-grain table itself. Contact sheets
    additionally need `frame_path` (plus `centroid_x`/`centroid_y`) and
    `illumination_field.png` needs `frame_path`; both need at least one
    frame that actually resolves under `frames_root`. Each degrades to
    contributing no path -- never raises -- when its required columns or
    frames are unavailable (see module docstring).

    Args:
        grains: Per-grain table (e.g. `pd.read_parquet(out_dir / "grains")`
            from a `detect` run).
        frames_root: Directory to search for the frames `grains[
            "frame_path"]` references (see `_resolve_frame_paths`).
        out_dir: Destination directory for every figure (created if
            missing).
        cfg: Resolved pipeline configuration; consumes `cfg.qc.
            defocus_contrast_min` and `cfg.flatfield.morph_kernel_px`.

    Returns:
        Paths of every figure actually written, always including
        `focus_scatter.png` and `rejection_vs_ecd.png`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_root = Path(frames_root)

    paths: list[Path] = [
        _focus_scatter(grains, out_dir, cfg),
        _rejection_vs_ecd(grains, out_dir),
    ]

    frame_lookup = _resolve_frame_paths(grains, frames_root)
    paths.extend(_contact_sheets_rejected(grains, frame_lookup, out_dir))
    paths.extend(_contact_sheets_accepted(grains, frame_lookup, out_dir))

    illum_path = _illumination_field(frame_lookup, out_dir, cfg)
    if illum_path is not None:
        paths.append(illum_path)

    return paths


# --- Overlays (`make_overlays`) -------------------------------------------


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
        polygon = shapely.wkt.loads(wkt_value)
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
    rgb_small: np.ndarray, grain_rows: pd.DataFrame, factor: int, frame_id: str, out_path: Path
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
    """
    n_accepted = int(grain_rows["qc_pass"].astype(bool).sum())
    n_rejected = len(grain_rows) - n_accepted

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
) -> list[Path]:
    """Render one QC-colored polygon-overlay PNG per explicitly requested frame.

    For each `fid` in `frame_ids`, draws every one of that frame's detected
    grains' subpixel polygon outlines over its full-resolution source
    image -- green for accepted (`qc_pass` `True`), red for rejected --
    then downsamples the whole composited raster by `factor` and adds a
    title plus per-grain `grain_uid`-suffix labels.

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

    Never raises for a per-frame problem -- an `fid` with no grains, an
    unresolvable/unreadable frame image, or (per grain) a missing contour
    row all degrade to skipping just that frame/grain (with a logged
    warning) rather than failing the whole call. `cfg` is accepted for
    interface symmetry with `make_reports`, but not otherwise consumed
    here: accepted/rejected coloring uses `grains["qc_pass"]` exactly as
    `detect` persisted it (`grain_morph.qc.qc_flags`), not a coloring
    recomputed from `cfg.qc.disqualifying_flags`.

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
        _annotate(rgb_small, frame_grains, factor, fid, out_path)
        written.append(out_path)

    return written
