"""Lightweight grain-labeling GUI for QC groundtruthing.

Samples grains from a completed ``detect`` run with a Latin-hypercube design
over the continuous metrics the QC gate thresholds on (so hand labels inform
each cut), shows each grain zoomed with its subpixel polygon mask, and records
a keystroke-assigned class into a resumable ``groundtruth.csv``.

The pure pieces -- sampling (:func:`_sample_grains`), the label store, the
:class:`_LabelSession` state machine, and crop geometry -- are unit-tested; the
matplotlib layer (:func:`run_groundtruth`) only binds keystrokes to the session
and draws.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import pandas as pd
from scipy.stats import qmc
from shapely import wkt as _wkt

from grain_morph.config import Config, load_config
from grain_morph.io import _IMAGE_EXTENSIONS, read_contours, read_grains

_LOGGER = logging.getLogger(__name__)

_FLAG_COLUMNS = (
    "flag_defocus",
    "flag_border",
    "flag_too_small",
    "flag_sliver",
    "flag_possible_agglomerate",
    "flag_no_polygon",
)
_NON_INTERACTIVE_BACKENDS = {"agg", "pdf", "ps", "svg", "cairo", "template"}

_LABEL_COLUMNS = (
    "grain_uid",
    "label",
    "label_name",
    "qc_status",
    "flags",
    "ecd_px",
    "sample_id",
    "camera",
    "frame_id",
    "labeled_at",
)


def _resolve_count(n: int | None, frac: float | None, total: int) -> int:
    """Resolve an absolute count / fraction / unset into a count in ``[0, total]``.

    Args:
        n: Absolute count, or ``None``.
        frac: Fraction of ``total``, or ``None``. Ignored when ``n`` is given.
        total: Population size the count is capped to.

    Returns:
        ``n`` (capped), ``round(frac*total)`` (capped), or ``total`` when both
        are ``None``.
    """
    if n is not None:
        return max(0, min(int(n), total))
    if frac is not None:
        return max(0, min(int(round(frac * total)), total))
    return total


def _quantile_ranks(values: np.ndarray) -> np.ndarray:
    """Empirical quantile rank of each value, in ``[0, 1)``.

    Args:
        values: 1-D array of finite values.

    Returns:
        Array of the same length; the ``k``-th smallest value maps to
        ``(k + 0.5) / len(values)`` (stable ties), so the axis is spread
        uniformly regardless of its raw scale or outliers.
    """
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = (np.arange(len(values)) + 0.5) / len(values)
    return ranks


def _frame_pool(
    grains: pd.DataFrame,
    n_frames: int | None,
    frac_frames: float | None,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Restrict candidates to a seed-random pool of frames (or all frames)."""
    frame_ids = grains["frame_id"].unique()
    k = _resolve_count(n_frames, frac_frames, len(frame_ids))
    if k >= len(frame_ids):
        return grains
    chosen = rng.choice(frame_ids, size=k, replace=False)
    return grains[grains["frame_id"].isin(chosen)]


def _greedy_match(design: np.ndarray, ranks: np.ndarray) -> list[int]:
    """Match each design point to its nearest not-yet-used candidate.

    Args:
        design: ``(m, d)`` Latin-hypercube points in the unit cube.
        ranks: ``(c, d)`` candidate coordinates (per-axis quantile ranks).

    Returns:
        ``m`` distinct candidate row indices, one per design point, each the
        Euclidean-nearest candidate still unused when that point is matched.
    """
    used = np.zeros(len(ranks), dtype=bool)
    chosen: list[int] = []
    for point in design:
        dist = np.sum((ranks - point) ** 2, axis=1)
        dist[used] = np.inf
        j = int(np.argmin(dist))
        used[j] = True
        chosen.append(j)
    return chosen


def _reserve(
    mask: np.ndarray,
    k: int,
    rng: np.random.Generator,
    taken: set[int],
) -> list[int]:
    """Seed-randomly pick up to ``k`` unused rows where ``mask`` is true."""
    if k <= 0:
        return []
    idxs = [int(i) for i in np.where(mask)[0] if int(i) not in taken]
    if not idxs:
        return []
    chosen = rng.choice(idxs, size=min(k, len(idxs)), replace=False)
    return [int(i) for i in chosen]


def _sample_grains(
    grains: pd.DataFrame,
    cfg: Config,
    *,
    n_frames: int | None = None,
    frac_frames: float | None = None,
    n_grains: int | None = None,
    frac_grains: float | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Select grains to label via a Latin-hypercube over the QC-driving metrics.

    Picks a frame pool (``n_frames``/``frac_frames``, else all frames), reserves
    a minimum number of the boolean-flag QC cases that have no continuous axis
    (``cfg.groundtruth.reserved`` -- border-touching and no-polygon grains), then
    fills the remainder of the target count with a space-filling Latin-hypercube
    sample over ``cfg.groundtruth.lhs_axes`` (each axis empirical-quantile-ranked
    so its raw scale is irrelevant). Deterministic for a given ``seed``.

    Args:
        grains: Per-grain table (must carry ``frame_id``, the ``lhs_axes``
            columns, ``touches_border``, and ``has_polygon``).
        cfg: Resolved configuration providing ``groundtruth`` settings.
        n_frames: Number of frames to draw the pool from (``None`` = all).
        frac_frames: Fraction of frames for the pool (ignored if ``n_frames``).
        n_grains: Target number of grains to select (``None`` = whole pool).
        frac_grains: Fraction of the pool to select (ignored if ``n_grains``).
        seed: Seed for both the frame/reservation RNG and the LHS design.

    Returns:
        The selected subset of ``grains`` (a copy, index reset). Reserved
        boolean-flag grains come first, then the hypercube selection.
    """
    rng = np.random.default_rng(seed)
    pool = _frame_pool(grains, n_frames, frac_frames, rng).reset_index(drop=True)
    target = _resolve_count(n_grains, frac_grains, len(pool))
    if target >= len(pool):
        return pool.copy()

    selected: list[int] = []
    taken: set[int] = set()

    reserved = cfg.groundtruth.reserved
    for mask, key in (
        (pool["touches_border"].to_numpy(dtype=bool), "border"),
        ((~pool["has_polygon"].to_numpy(dtype=bool)), "no_polygon"),
    ):
        room = target - len(selected)
        for i in _reserve(mask, min(reserved.get(key, 0), room), rng, taken):
            selected.append(i)
            taken.add(i)

    axes = list(cfg.groundtruth.lhs_axes)
    axis_vals = pool[axes].to_numpy(dtype=float)
    finite = np.all(np.isfinite(axis_vals), axis=1)
    cand_idx = np.array(
        [i for i in range(len(pool)) if finite[i] and i not in taken], dtype=int
    )
    remaining = min(target - len(selected), len(cand_idx))
    if remaining > 0:
        ranks = np.column_stack(
            [_quantile_ranks(axis_vals[cand_idx, j]) for j in range(len(axes))]
        )
        design = qmc.LatinHypercube(d=len(axes), seed=seed).random(n=remaining)
        selected.extend(int(cand_idx[local]) for local in _greedy_match(design, ranks))

    return pool.iloc[selected].reset_index(drop=True)


def _crop_bbox(
    bounds: tuple[float, float, float, float],
    pad: int,
    frame_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Padded, frame-clamped crop bounds for a grain's polygon.

    Args:
        bounds: Polygon ``(minx, miny, maxx, maxy)`` in image coordinates,
            where ``x`` is column and ``y`` is row (``shapely``'s ``.bounds``).
        pad: Pixels of padding to add on every side.
        frame_shape: ``(height, width)`` of the source frame.

    Returns:
        ``(row0, row1, col0, col1)`` integer bounds, clamped to ``[0, height]``
        and ``[0, width]``, suitable for ``image[row0:row1, col0:col1]``.
    """
    minx, miny, maxx, maxy = bounds
    height, width = frame_shape
    row0 = max(0, int(np.floor(miny)) - pad)
    row1 = min(height, int(np.ceil(maxy)) + pad)
    col0 = max(0, int(np.floor(minx)) - pad)
    col1 = min(width, int(np.ceil(maxx)) + pad)
    return row0, row1, col0, col1


class _LabelStore:
    """Resumable, autosaving store of ground-truth labels keyed by ``grain_uid``.

    Every mutating call rewrites ``groundtruth.csv`` atomically (temp file +
    replace), so a crash mid-session never loses or corrupts prior labels and a
    relaunch resumes exactly where it left off.
    """

    def __init__(self, path: str | Path, rows: dict[str, dict] | None = None) -> None:
        """Initialize a store at ``path`` with optional preloaded ``rows``."""
        self.path = Path(path)
        self._rows: dict[str, dict] = rows or {}

    @classmethod
    def load(cls, path: str | Path) -> _LabelStore:
        """Load an existing ``groundtruth.csv``, or an empty store if absent."""
        path = Path(path)
        rows: dict[str, dict] = {}
        if path.exists():
            for record in pd.read_csv(path).to_dict("records"):
                rows[str(record["grain_uid"])] = record
        return cls(path, rows)

    def labeled_uids(self) -> set[str]:
        """Return the set of currently-labeled ``grain_uid``s."""
        return set(self._rows)

    def get(self, grain_uid: str) -> int | None:
        """Return the integer label for ``grain_uid``, or ``None`` if unlabeled."""
        record = self._rows.get(grain_uid)
        return None if record is None else int(record["label"])

    def set(self, grain_uid: str, record: dict) -> None:
        """Record (or overwrite) ``grain_uid``'s label and autosave."""
        self._rows[grain_uid] = {"grain_uid": grain_uid, **record}
        self.save()

    def unset(self, grain_uid: str) -> None:
        """Remove ``grain_uid``'s label (if any) and autosave."""
        if self._rows.pop(grain_uid, None) is not None:
            self.save()

    def to_frame(self) -> pd.DataFrame:
        """Materialize all labels as a DataFrame with the canonical columns."""
        return pd.DataFrame(list(self._rows.values()), columns=list(_LABEL_COLUMNS))

    def save(self) -> None:
        """Write the store to ``self.path`` atomically (temp file + replace)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        self.to_frame().to_csv(tmp, index=False)
        tmp.replace(self.path)


class _LabelSession:
    """Navigation + label bookkeeping for the GUI, independent of matplotlib.

    Holds the review order, the current index, and the mask/status visibility
    toggles; delegates persistence to a :class:`_LabelStore`. Record content is
    built by an injected ``record_fn`` so this class stays free of grain-table
    and config specifics (and trivially testable).
    """

    def __init__(
        self,
        order: list[str],
        store: _LabelStore,
        record_fn: Callable[[str, int, str], dict],
        *,
        show_status: bool = False,
    ) -> None:
        """Initialize at the first unlabeled grain in ``order``.

        Args:
            order: Grain uids in review order.
            store: Backing label store (already loaded; may hold prior labels).
            record_fn: ``(grain_uid, value, name) -> record dict`` used to build
                the row persisted on :meth:`assign`.
            show_status: Initial predicted-QC-status visibility.
        """
        self.order = order
        self.store = store
        self.record_fn = record_fn
        self.mask_visible = True
        self.status_visible = show_status
        self.index = self._first_unlabeled()

    def _first_unlabeled(self) -> int:
        """Index of the first uid in ``order`` with no label (else 0)."""
        labeled = self.store.labeled_uids()
        for i, uid in enumerate(self.order):
            if uid not in labeled:
                return i
        return 0

    def current(self) -> str:
        """The uid currently under review."""
        return self.order[self.index]

    def assign(self, value: int, name: str) -> None:
        """Label the current grain (via ``record_fn``) and advance."""
        uid = self.current()
        self.store.set(uid, self.record_fn(uid, value, name))
        self.next()

    def next(self) -> None:
        """Advance to the next grain (clamped at the end)."""
        self.index = min(self.index + 1, len(self.order) - 1)

    def prev(self) -> None:
        """Step back to the previous grain (clamped at the start)."""
        self.index = max(self.index - 1, 0)

    def unset(self) -> None:
        """Clear the current grain's label."""
        self.store.unset(self.current())

    def toggle_mask(self) -> None:
        """Toggle the polygon-mask overlay."""
        self.mask_visible = not self.mask_visible

    def toggle_status(self) -> None:
        """Toggle whether the predicted QC status is shown."""
        self.status_visible = not self.status_visible

    def labeled_count(self) -> int:
        """Number of grains in ``order`` that currently have a label."""
        labeled = self.store.labeled_uids()
        return sum(uid in labeled for uid in self.order)


def _active_flags(row: dict) -> list[str]:
    """QC flag columns that are true for ``row`` (order = ``_FLAG_COLUMNS``)."""
    return [flag for flag in _FLAG_COLUMNS if bool(row.get(flag, False))]


def _predicted_status(row: dict, cfg: Config) -> tuple[str, str]:
    """The gate's predicted outcome for a grain, and its active flags.

    Args:
        row: A grain record (mapping of column -> value).
        cfg: Resolved configuration; ``qc.disqualifying_flags`` defines the gate.

    Returns:
        ``("reject", flags)`` if any active flag is disqualifying, else
        ``("accept", flags)``; ``flags`` is a comma-joined list of active flags.
    """
    disqualifying = set(cfg.qc.disqualifying_flags)
    active = _active_flags(row)
    status = "reject" if any(flag in disqualifying for flag in active) else "accept"
    return status, ",".join(active)


def _make_record_fn(rows: dict[str, dict], cfg: Config) -> Callable[[str, int, str], dict]:
    """Build the ``record_fn`` that turns a keystroke into a stored label row."""

    def record_fn(grain_uid: str, value: int, name: str) -> dict:
        row = rows[grain_uid]
        status, flags = _predicted_status(row, cfg)
        return {
            "label": value,
            "label_name": name,
            "qc_status": status,
            "flags": flags,
            "ecd_px": row.get("ecd_px"),
            "sample_id": row.get("sample_id"),
            "camera": row.get("camera"),
            "frame_id": row.get("frame_id"),
            "labeled_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }

    return record_fn


def _require_interactive_backend(backend: str) -> None:
    """Raise a clear error if ``backend`` can't open an interactive window.

    Args:
        backend: The active matplotlib backend name (``matplotlib.get_backend()``).

    Raises:
        RuntimeError: If ``backend`` is a non-interactive (file-only) backend.
    """
    if backend.lower() in _NON_INTERACTIVE_BACKENDS:
        raise RuntimeError(
            f"groundtruth needs an interactive matplotlib backend, but the current "
            f"backend is {backend!r}. Run in a desktop session, e.g. "
            f"`MPLBACKEND=QtAgg grain-morph groundtruth ...` (or TkAgg / macosx)."
        )


def _interactive_backend_candidates() -> tuple[str, ...]:
    """Interactive matplotlib backends to try, best-first for this platform.

    Returns:
        Backend names in preference order. ``macosx`` ships with matplotlib on
        Darwin and needs no toolkit install, so it leads there; elsewhere it is
        omitted entirely.
    """
    if sys.platform == "darwin":
        return ("macosx", "QtAgg", "TkAgg")
    return ("QtAgg", "TkAgg")


def _ensure_interactive_backend() -> str:
    """Switch to an interactive matplotlib backend if the active one is file-only.

    Importing ``grain_morph.cli`` pulls in ``overlay`` and ``report``, which
    select ``Agg`` for their headless figure writing. That would otherwise leave
    ``groundtruth`` stuck on a backend that can't open a window, so pick a real
    one here instead of making every caller prefix ``MPLBACKEND=``.

    An explicit ``MPLBACKEND`` in the environment is always respected -- if the
    operator named a file-only backend they get the usual error rather than a
    silent override, which keeps headless invocations failing fast.

    Returns:
        The name of the active interactive backend.

    Raises:
        RuntimeError: If ``MPLBACKEND`` names a non-interactive backend, or if no
            candidate backend could be imported (no GUI toolkit installed).
    """
    import matplotlib
    import matplotlib.pyplot as plt

    current = matplotlib.get_backend()
    if current.lower() not in _NON_INTERACTIVE_BACKENDS:
        return current

    if os.environ.get("MPLBACKEND"):
        _require_interactive_backend(current)  # explicit choice -- don't second-guess
        return current

    failures: list[str] = []
    for candidate in _interactive_backend_candidates():
        try:
            plt.switch_backend(candidate)
        except Exception as exc:  # ImportError, or a toolkit with no display
            failures.append(f"{candidate} ({type(exc).__name__}: {exc})")
            continue
        resolved = matplotlib.get_backend()
        _LOGGER.info("switched matplotlib backend %s -> %s", current, resolved)
        return resolved

    raise RuntimeError(
        "groundtruth needs an interactive matplotlib backend and none could be "
        "loaded. Tried: " + "; ".join(failures) + ". Install a GUI toolkit "
        "(`conda install -c conda-forge pyqt`) and/or run in a desktop session. "
        "To force a specific backend, set MPLBACKEND."
    )


def _resolve_frame_path(row: dict, frames_dir: str | Path | None) -> Path:
    """Locate a grain's source frame, preferring ``frames_dir`` if given.

    Args:
        row: Grain record carrying ``frame_id`` and the stored ``frame_path``.
        frames_dir: Optional directory to relocate frames into (used when the
            stored absolute ``frame_path`` is stale, e.g. a remounted drive).

    Returns:
        The path to load the frame image from: ``frames_dir/<frame_id><ext>``
        for the first matching image extension if present, else the stored
        ``frame_path``.
    """
    if frames_dir is not None:
        base = Path(frames_dir)
        fid = str(row["frame_id"])
        for ext in _IMAGE_EXTENSIONS:
            candidate = base / f"{fid}{ext}"
            if candidate.exists():
                return candidate
    return Path(str(row["frame_path"]))


def run_groundtruth(
    run_dir: str | Path,
    *,
    n_frames: int | None = None,
    frac_frames: float | None = None,
    n_grains: int | None = None,
    frac_grains: float | None = None,
    seed: int = 0,
    config: str | Path | None = None,
    out: str | Path | None = None,
    frames_dir: str | Path | None = None,
    show: bool = True,
) -> pd.DataFrame:
    """Open the labeling GUI over a sampled subset of a ``detect`` run.

    Loads the run's grains + contours, draws a Latin-hypercube sample over the
    QC-driving metrics (:func:`_sample_grains`), and opens a matplotlib window
    that shows one grain at a time (zoomed crop + toggleable polygon mask) with
    keystroke class assignment. Labels autosave to ``groundtruth.csv`` and the
    session resumes from the first unlabeled grain on relaunch.

    Args:
        run_dir: Completed ``detect`` output directory (``grains/`` + ``contours/``).
        n_frames: Frames to draw the pool from (``None`` = all).
        frac_frames: Fraction of frames for the pool (ignored if ``n_frames``).
        n_grains: Target grains to label (``None`` = whole pool).
        frac_grains: Fraction of the pool to label (ignored if ``n_grains``).
        seed: Seed for reproducible sampling.
        config: Optional user config YAML overriding packaged defaults.
        out: Labels CSV path (default ``run_dir/groundtruth.csv``).
        frames_dir: Optional directory to relocate source frames from.
        show: If ``False``, build the session but don't open the window (returns
            immediately with the current labels -- used by tests).

    Returns:
        The labels table (``groundtruth.csv`` contents) as a DataFrame.

    Raises:
        FileNotFoundError: If the run has no grains table.
    """
    run_dir = Path(run_dir)
    cfg = load_config(config)
    grains = read_grains(run_dir / "grains", cfg)
    if grains is None:
        raise FileNotFoundError(
            f"no grains table under {run_dir / 'grains'} -- run `detect` first"
        )

    sample = _sample_grains(
        grains,
        cfg,
        n_frames=n_frames,
        frac_frames=frac_frames,
        n_grains=n_grains,
        frac_grains=frac_grains,
        seed=seed,
    )
    order = [str(uid) for uid in sample["grain_uid"]]
    frame_ids = sorted({str(fid) for fid in sample["frame_id"]})
    contours = read_contours(run_dir, frame_ids, cfg)
    polygons = {
        str(rec["grain_uid"]): _wkt.loads(rec["wkt"]) for rec in contours.to_dict("records")
    }
    rows = {str(uid): rec for uid, rec in sample.set_index("grain_uid").to_dict("index").items()}

    out_path = Path(out) if out is not None else run_dir / "groundtruth.csv"
    store = _LabelStore.load(out_path)
    session = _LabelSession(
        order,
        store,
        _make_record_fn(rows, cfg),
        show_status=cfg.groundtruth.show_predicted_status,
    )

    if show:
        _launch_gui(session, rows, polygons, cfg, frames_dir)
    return store.to_frame()


def _launch_gui(
    session: _LabelSession,
    rows: dict[str, dict],
    polygons: dict[str, Any],
    cfg: Config,
    frames_dir: str | Path | None,
) -> None:
    """Bind keystrokes to ``session`` and drive the one-grain matplotlib view."""
    import matplotlib.pyplot as plt

    _ensure_interactive_backend()

    key_to_class = {c.key: (c.value, c.name) for c in cfg.groundtruth.classes}
    key_help = ", ".join(f"{c.key}={c.name}" for c in cfg.groundtruth.classes)
    fig, ax = plt.subplots(figsize=(6, 6))

    def draw() -> None:
        ax.clear()
        uid = session.current()
        row = rows[uid]
        image = iio.imread(_resolve_frame_path(row, frames_dir))
        if image.ndim == 3:
            image = image[..., 0]
        poly = polygons.get(uid)
        pad = cfg.groundtruth.crop_pad_px
        if poly is not None:
            r0, r1, c0, c1 = _crop_bbox(poly.bounds, pad, image.shape[:2])
        else:
            cx = float(row.get("centroid_x", image.shape[1] / 2))
            cy = float(row.get("centroid_y", image.shape[0] / 2))
            r0, r1, c0, c1 = _crop_bbox((cx - 1, cy - 1, cx + 1, cy + 1), pad * 4, image.shape[:2])
        ax.imshow(image[r0:r1, c0:c1], cmap="gray", vmin=0, vmax=255)
        if poly is not None and session.mask_visible:
            xs, ys = poly.exterior.xy
            px, py = np.asarray(xs) - c0, np.asarray(ys) - r0
            ax.plot(px, py, "-", color="#ff3b3b", lw=1.2)
            ax.fill(px, py, color="#ff3b3b", alpha=cfg.groundtruth.mask_alpha)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(_status_line(session, row, cfg), fontsize=9, loc="left")
        fig.suptitle(f"keys: {key_help}, m=mask n/p=nav u=unset i=info q=quit", fontsize=8)
        fig.canvas.draw_idle()

    def on_key(event: object) -> None:
        key = getattr(event, "key", None)
        if key in key_to_class:
            session.assign(*key_to_class[key])
        elif key == "m":
            session.toggle_mask()
        elif key in ("n", "right"):
            session.next()
        elif key in ("p", "left"):
            session.prev()
        elif key == "u":
            session.unset()
        elif key == "i":
            session.toggle_status()
        elif key in ("q", "escape"):
            plt.close(fig)
            return
        else:
            return
        draw()

    fig.canvas.mpl_connect("key_press_event", on_key)
    draw()
    plt.show()


def _status_line(session: _LabelSession, row: dict, cfg: Config) -> str:
    """Compose the per-grain status/title text for the GUI."""
    uid = session.current()
    parts = [
        f"[{session.index + 1}/{len(session.order)}]  {uid}",
        f"{row.get('sample_id')} / {row.get('camera')}",
    ]
    ecd = row.get("ecd_px")
    if ecd is not None and np.isfinite(ecd):
        parts.append(f"ecd {float(ecd):.1f} px")
    if session.status_visible:
        status, flags = _predicted_status(row, cfg)
        parts.append(f"predicted: {status}" + (f" ({flags})" if flags else ""))
    label = session.store.get(uid)
    parts.append(f"label: {label if label is not None else '-'}")
    return "   |   ".join(parts)
