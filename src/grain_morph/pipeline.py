"""Detect pipeline orchestration: per-frame processing and parallel `run_detect`.

Ties together every module from Tasks 4-11 (`io`, `flatfield`, `detect`,
`measure`, `qc`, `writers`) into the two entry points Stage 1 (`grain-morph
detect`) needs:

- :func:`process_frame` — a pure, picklable, single-frame worker: reads its
  own frame (+ paired blank) from disk via a `FrameSpec`'s paths, flat-fields,
  detects, measures, and QC-flags every object, and returns only small
  dicts/WKT strings (never an image array) as a :class:`FrameResult`. This is
  what makes it safe to run in a `loky` worker process (design doc §11.2).
- :func:`run_detect` — discovers frames, skips ones already recorded in
  `manifest` (resume, unless `force=True`), fans `process_frame` out across
  frames with `joblib.Parallel(backend="loky", return_as="generator")`, and
  streams each frame's rows to disk **as that frame completes** (one
  deterministically-named output file per `frame_id`, never accumulated
  across frames) rather than accumulating a whole run's rows in memory.
  Every frame (including ones that raise) gets exactly one manifest entry;
  nothing ever aborts the whole run *except* an unrecoverable whole-run
  precondition -- an uncalibrated camera among the discovered frames is
  checked and raised up front, before any frame is dispatched, rather than
  being caught by `process_frame`'s per-frame isolation and silently turned
  into a run full of `status="error"` frames.

**Resume/force correctness note:** every frame's grain rows (and, if
`cfg.output.save_contours`, its contour rows) live at a path determined
*only* by that frame's identity (`frame_id`, plus `sample_id`/`camera` for
partitioned parquet) -- never by *when* or *how many times* `run_detect` has
been called. Reprocessing a frame (via `force=True`, or via resume after
that frame's file fingerprint changed) overwrites exactly that frame's own
output, and a frame that now yields zero objects has its stale output file
deleted. So `pd.read_parquet(out_dir / "grains")` (or the csv/feather
per-frame files) always reflects exactly the current on-disk frames' current
detections -- no stale rows left behind by an earlier run, no duplicates
from re-running. An earlier version of this module wrote each run's chunks
under a random per-run token that was never cleaned up, which silently
accumulated duplicate/stale rows across repeated `run_detect` calls; see the
task-12 fix report for how that was found and fixed.

**Determinism / resume note:** `Manifest.add`'s `seconds` field is always
persisted as `0.0` here, never the frame's real wall-clock processing time
(which *is* returned in `FrameResult.seconds`, and rolled into
`summary.json`'s aggregate wall time). Wall-clock timing is inherently
non-reproducible between runs, and criterion 7 requires a resumed run's
`manifest` bytes to be byte-identical to the original — so per-frame timing
cannot live in the persisted manifest. This is a deliberate scope decision,
not an oversight (see task-12 report).
"""

from __future__ import annotations

import json
import shutil
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import joblib
import numpy as np
import pandas as pd
import yaml

from grain_morph import __version__
from grain_morph.config import Config, config_hash
from grain_morph.detect import Detection, detect_objects
from grain_morph.flatfield import apply_flatfield
from grain_morph.io import FrameSpec, Manifest, discover_frames, frame_fingerprint
from grain_morph.measure import (
    measure_curvature_entropy,
    measure_efd,
    measure_polygon,
    measure_wadell,
    perimeter_crofton_px,
)
from grain_morph.qc import qc_flags, qc_metrics
from grain_morph.writers import read_table, write_table

# Mirrors `grain_morph.writers._EXTENSIONS`. Duplicated (rather than imported)
# because that mapping is a private module attribute of `writers` -- this is
# the one place pipeline.py needs to predict a table's on-disk extension
# *before* writing it (to check for a prior manifest/errors table to resume
# from), and the three formats are already pinned by `OutputConfig.format`'s
# `Literal`, so the duplication can't silently drift out of sync unnoticed.
_EXTENSIONS = {"parquet": ".parquet", "csv": ".csv", "feather": ".feather"}

# Identity columns that lead every persisted grain row (design doc §7).
_IDENTITY_COLUMNS = (
    "grain_uid",
    "frame_id",
    "frame_path",
    "sample_id",
    "camera",
    "run",
    "label",
    "centroid_x",
    "centroid_y",
    "um_per_px",
    "wadell_smoothing",
    "pipeline_version",
    "config_hash",
)

# `measure_polygon`'s exact key set (measure.py), in its own return order.
_POLYGON_KEYS = (
    "area_px",
    "area_um2",
    "ecd_um",
    "feret_max_um",
    "feret_min_um",
    "major_axis_um",
    "minor_axis_um",
    "perimeter_um",
    "aspect_ratio",
    "solidity",
    "convexity",
    "circularity",
    "extent",
    "eccentricity",
    "orientation",
)

_WADELL_KEYS = ("wadell_roundness", "wadell_sphericity")
_CURVATURE_KEYS = ("curvature_entropy", "curvature_smoothing")

# `qc_metrics`' own `aspect_ratio`/`solidity` are a *second*, independently
# computed (raster, locally-cropped) estimate of the same physical
# quantities `measure_polygon` already reports under those exact names --
# see `_build_row`'s docstring for why they are persisted under the
# `qc_`-prefixed names below instead of silently overwriting
# `measure_polygon`'s columns on dict merge.
_QC_METRIC_COLUMNS = (
    "edge_width_px",
    "edge_gradient",
    "edge_gradient_norm",
    "contrast",
    "ecd_px",
    "qc_aspect_ratio",
    "qc_solidity",
    "touches_border",
    "has_polygon",
)

_QC_FLAG_COLUMNS = (
    "flag_defocus",
    "flag_border",
    "flag_too_small",
    "flag_sliver",
    "flag_possible_agglomerate",
    "flag_no_polygon",
    "qc_pass",
)

_CONTOUR_COLUMNS = ("grain_uid", "wkt", "um_per_px")


@dataclass
class FrameResult:
    """The result of processing one frame, safe to pass across a process boundary.

    Attributes:
        rows: One dict per detected object (see `_build_row` for the full
            column set), or `[]` if the frame had no objects or failed.
        contours: One `{"grain_uid", "wkt", "um_per_px"}` dict per detected
            object that has a polygon, or `[]`.
        status: `"ok"` (>= 1 object), `"empty"` (0 objects), or `"error"`.
        n_objects: Number of objects `detect_objects` found (0 on error).
        seconds: Wall-clock time spent processing this frame.
        flatfield_method: `"blank"` or `"morphological"` (`""` on error,
            since flat-fielding may not have completed).
        error: The full traceback string if this frame raised, else `None`.
    """

    rows: list[dict[str, Any]] = field(default_factory=list)
    contours: list[dict[str, Any]] = field(default_factory=list)
    status: str = "ok"
    n_objects: int = 0
    seconds: float = 0.0
    flatfield_method: str = ""
    error: str | None = None


def _read_grayscale(path: Path) -> np.ndarray:
    """Read an image frame as a 2D `uint8` grayscale array.

    Args:
        path: Path to the image file.

    Returns:
        2D `uint8` array. If the file decodes with an extra channel axis
        (e.g. an accidental RGB save of a grayscale source), only the first
        channel is kept.
    """
    array = np.asarray(iio.imread(path))
    if array.ndim == 3:
        array = array[..., 0]
    return array.astype(np.uint8)


def _efd_keys(order: int) -> list[str]:
    """Column names `measure_efd` produces for a given harmonic `order`.

    Args:
        order: Number of Fourier harmonics (`cfg.measure.efd_order`).

    Returns:
        `4 * order` column names, `efd_{h}_a/b/c/d` for `h` in `1..order`.
    """
    keys: list[str] = []
    for h in range(1, order + 1):
        keys.extend([f"efd_{h}_a", f"efd_{h}_b", f"efd_{h}_c", f"efd_{h}_d"])
    return keys


def _grain_columns(efd_order: int) -> list[str]:
    """The full, stable per-grain column order for a given config.

    Computed directly from `cfg.measure.efd_order` (not from any actual
    row) so it is identical for every frame's output file across a run --
    including a frame whose rows are entirely empty/`NaN`-filled -- which
    is what lets every frame's rows be reindexed to one consistent schema
    before they are written.

    Args:
        efd_order: `cfg.measure.efd_order`, which determines how many
            `efd_*` columns there are.

    Returns:
        Ordered column names: identity, `measure_polygon`, `measure_efd`
        (+ `fourier_power_cum_90`), `measure_wadell`,
        `measure_curvature_entropy`, `perimeter_crofton_px`, `qc_metrics`,
        `qc_flags`.
    """
    return (
        list(_IDENTITY_COLUMNS)
        + list(_POLYGON_KEYS)
        + _efd_keys(efd_order)
        + ["fourier_power_cum_90"]
        + list(_WADELL_KEYS)
        + list(_CURVATURE_KEYS)
        + ["perimeter_crofton_px"]
        + list(_QC_METRIC_COLUMNS)
        + list(_QC_FLAG_COLUMNS)
    )


def _grain_leaf_path(
    grains_dir: Path, fmt: str, partition: bool, sample_id: str, camera: str, frame_id: str
) -> Path:
    """Deterministic per-frame output path for one frame's grain rows.

    One frame maps to exactly one path, named by `frame_id` (never by a
    per-run token or an incrementing counter) -- reprocessing a frame,
    whether via `force=True` or via resume after that frame's fingerprint
    changed, always targets and overwrites this exact path, so repeated
    `run_detect` calls can never leave stale or duplicate rows for that
    frame lying around under a different, orphaned filename. A prior
    version of this module named each run's output chunks with a random
    `uuid4()` token that was never cleaned up, which meant every call's
    chunks piled up on disk forever and `pd.read_parquet(grains_dir)`
    silently summed rows across every run that had ever touched
    `out_dir` -- this is the fix.

    Args:
        grains_dir: `out_dir / "grains"`.
        fmt: `cfg.output.format`.
        partition: `cfg.output.partition`.
        sample_id: The frame's sample id.
        camera: The frame's camera.
        frame_id: The frame's stem.

    Returns:
        The path (extension included) that this frame's grain rows are
        written to, or deleted from if the frame now has none.
    """
    ext = _EXTENSIONS[fmt]
    if not partition:
        return grains_dir / f"{frame_id}{ext}"
    if fmt == "parquet":
        # Hive convention (matches `writers.write_partitioned`): partition
        # columns live only in the directory path, not the leaf filename.
        return grains_dir / f"sample_id={sample_id}" / f"camera={camera}" / f"{frame_id}{ext}"
    # csv/feather have no directory-encoded partition layer (writers.py's
    # own documented convention for those formats): partition columns stay
    # embedded as columns instead, and the partition key is folded into the
    # filename alongside `frame_id` so distinct (sample_id, camera) groups
    # don't collide.
    return grains_dir / f"{sample_id}__{camera}__{frame_id}{ext}"


def _write_or_clear_grain_frame(
    rows: list[dict[str, Any]],
    grains_dir: Path,
    fmt: str,
    partition: bool,
    sample_id: str,
    camera: str,
    frame_id: str,
    grain_columns: list[str],
) -> None:
    """Write one frame's grain rows to its deterministic path, or clear it.

    Called once per processed frame (never accumulated across frames),
    which is what keeps a run memory-bounded: at most one frame's rows are
    ever resident before being flushed to disk. If `rows` is empty (the
    frame now has zero objects -- e.g. re-run after an edit that removed
    its only object), any stale file left over from an earlier run is
    deleted instead of written, so a re-run never leaves orphaned rows.

    Args:
        rows: This frame's grain row dicts; may be empty.
        grains_dir: `out_dir / "grains"`.
        fmt: `cfg.output.format`.
        partition: `cfg.output.partition`.
        sample_id: The frame's sample id.
        camera: The frame's camera.
        frame_id: The frame's stem.
        grain_columns: Canonical column order (`_grain_columns`).
    """
    path = _grain_leaf_path(grains_dir, fmt, partition, sample_id, camera, frame_id)
    if not rows:
        path.unlink(missing_ok=True)
        return
    df = pd.DataFrame(rows).reindex(columns=grain_columns)
    df = df.sort_values("grain_uid").reset_index(drop=True)
    if partition and fmt == "parquet":
        df = df.drop(columns=["sample_id", "camera"])
    write_table(df, path, fmt)


def _write_or_clear_contour_frame(
    contours: list[dict[str, Any]], contours_dir: Path, fmt: str, frame_id: str
) -> None:
    """Write one frame's contour rows to its deterministic path, or clear it.

    Mirrors `_write_or_clear_grain_frame`, but contours are never
    partitioned (design doc §7: a single flat table keyed on `grain_uid`),
    so the path is just `contours_dir / f"{frame_id}{ext}"`.

    Args:
        contours: This frame's `{"grain_uid", "wkt", "um_per_px"}` dicts;
            may be empty.
        contours_dir: `out_dir / "contours"`.
        fmt: `cfg.output.format`.
        frame_id: The frame's stem.
    """
    path = contours_dir / f"{frame_id}{_EXTENSIONS[fmt]}"
    if not contours:
        path.unlink(missing_ok=True)
        return
    df = pd.DataFrame(contours).reindex(columns=list(_CONTOUR_COLUMNS))
    df = df.sort_values("grain_uid").reset_index(drop=True)
    write_table(df, path, fmt)


def _validate_calibration(specs: list[FrameSpec], cfg: Config) -> None:
    """Fail loudly, up front, if any discovered frame's camera is uncalibrated.

    Calibration is a whole-run configuration precondition, not per-frame
    data corruption. Left unchecked here, an uncalibrated camera would
    only surface inside `process_frame`'s blanket per-frame exception
    handling -- `cfg.um_per_px(camera)` raising `KeyError` mid-frame -- so
    the run would exit 0 with a `manifest`/`errors` full of `status=
    "error"` frames instead of aborting loudly before any work is done.
    Checked once for every distinct camera among *all* discovered frames
    (not just ones that still need processing), before `out_dir` is even
    created.

    Args:
        specs: Discovered frames (`io.discover_frames`).
        cfg: Resolved pipeline configuration.

    Raises:
        KeyError: If any distinct camera among `specs` has no (or a still
            `null`) `calibration.um_per_px` entry.
    """
    for camera in sorted({spec.parsed.camera for spec in specs}):
        cfg.um_per_px(camera)


def _build_row(
    det: Detection,
    mask: np.ndarray,
    corrected: np.ndarray,
    raw_image: np.ndarray,
    frame_shape: tuple[int, int],
    cfg: Config,
    um_per_px: float,
    spec: FrameSpec,
    grain_uid: str,
    pipeline_version: str,
    cfg_hash_value: str,
) -> dict[str, Any]:
    """Assemble one detected object's full persisted row.

    Merges `measure_polygon` + `measure_efd` + `measure_wadell` +
    `measure_curvature_entropy` + `perimeter_crofton_px` + `qc_metrics` +
    `qc_flags` with the identity columns (design doc §7). When
    `det.polygon` is `None` (`det.contour_ok is False`) the
    polygon-dependent measures (everything from `measure_polygon`,
    `measure_efd`, `measure_wadell`, `measure_curvature_entropy`) are
    `NaN`-filled rather than raising -- `qc_metrics`/`qc_flags` already
    support `poly=None` via their raster-mask fallback (and set
    `flag_no_polygon`), and `perimeter_crofton_px` only ever needs the
    mask -- so a single bad contour degrades that one row's geometry
    columns instead of failing the whole frame.

    Ambiguity resolved here (documented, not tested by any acceptance
    criterion): `qc_metrics` independently computes its own `aspect_ratio`
    and `solidity` (from a locally-cropped raster, for `qc_flags`'
    thresholding) under the *same* key names `measure_polygon` already
    uses for its own, differently-computed versions of the same physical
    quantities. Persisting both under one shared name would silently drop
    one on dict-merge; they are kept under separate names instead --
    `aspect_ratio`/`solidity` from `measure_polygon` (design §5's
    documented per-grain shape descriptors), `qc_aspect_ratio`/
    `qc_solidity` from `qc_metrics` (the exact values `qc_flags` actually
    thresholded on for this row) -- so nothing is lost and both are
    independently inspectable.

    Args:
        det: The detected object.
        mask: Boolean mask for `det.label` over the full frame
            (`label_image == det.label`).
        corrected: Flat-fielded frame.
        raw_image: Raw (un-flat-fielded) frame, `float64`.
        frame_shape: `(height, width)` of the frame.
        cfg: Resolved pipeline configuration.
        um_per_px: Calibrated microns-per-pixel for this frame's camera.
        spec: The frame's `FrameSpec` (for identity columns).
        grain_uid: `f"{frame_id}:{label}"`.
        pipeline_version: `grain_morph.__version__`.
        cfg_hash_value: `grain_morph.config.config_hash(cfg)`.

    Returns:
        One fully-assembled row dict, columns per `_grain_columns`.
    """
    nan = float("nan")
    if det.polygon is not None:
        poly_metrics: dict[str, float] = measure_polygon(det.polygon, um_per_px)
        efd_metrics, cum90 = measure_efd(
            det.polygon, cfg.measure.efd_order, cfg.measure.efd_resample_n
        )
        wadell_metrics = measure_wadell(det.polygon, cfg.measure.wadell_smoothing)
        curvature_metrics = measure_curvature_entropy(
            det.polygon,
            cfg.measure.curvature_resample_n,
            cfg.measure.curvature_smoothing,
            cfg.measure.curvature_bins,
        )
    else:
        poly_metrics = dict.fromkeys(_POLYGON_KEYS, nan)
        efd_metrics = dict.fromkeys(_efd_keys(cfg.measure.efd_order), nan)
        cum90 = nan
        wadell_metrics = dict.fromkeys(_WADELL_KEYS, nan)
        curvature_metrics = {
            "curvature_entropy": nan,
            "curvature_smoothing": float(cfg.measure.curvature_smoothing),
        }

    crofton = perimeter_crofton_px(mask)
    metrics = qc_metrics(raw_image, corrected, det.polygon, mask, det.mask_bbox, frame_shape, cfg)
    flags = qc_flags(metrics, cfg)

    centroid_x, centroid_y = det.centroid_xy
    row: dict[str, Any] = {
        "grain_uid": grain_uid,
        "frame_id": spec.parsed.stem,
        "frame_path": str(spec.path),
        "sample_id": spec.parsed.sample_id,
        "camera": spec.parsed.camera,
        # `run` is populated from `sample_regex`'s optional `run` group when
        # present, else `None` (the packaged default has no `run` group and
        # folds any run token into `sample_id` instead).
        "run": spec.parsed.run,
        "label": det.label,
        "centroid_x": float(centroid_x),
        "centroid_y": float(centroid_y),
        "um_per_px": um_per_px,
        "wadell_smoothing": float(cfg.measure.wadell_smoothing),
        "pipeline_version": pipeline_version,
        "config_hash": cfg_hash_value,
    }
    row.update(poly_metrics)
    row.update(efd_metrics)
    row["fourier_power_cum_90"] = cum90
    row.update(wadell_metrics)
    row.update(curvature_metrics)
    row["perimeter_crofton_px"] = crofton
    row["edge_width_px"] = metrics["edge_width_px"]
    row["edge_gradient"] = metrics["edge_gradient"]
    row["edge_gradient_norm"] = metrics["edge_gradient_norm"]
    row["contrast"] = metrics["contrast"]
    row["ecd_px"] = metrics["ecd_px"]
    row["qc_aspect_ratio"] = metrics["aspect_ratio"]
    row["qc_solidity"] = metrics["solidity"]
    row["touches_border"] = metrics["touches_border"]
    row["has_polygon"] = metrics["has_polygon"]
    row.update(flags)
    return row


def process_frame(
    spec: FrameSpec, cfg: Config, pipeline_version: str, cfg_hash_value: str
) -> FrameResult:
    """Process one frame end-to-end: flat-field, detect, measure, QC.

    Pure and picklable, per design doc §11.2: reads only its own frame
    (and paired blank, if any) from disk via `spec`'s paths, does all
    compute locally, and returns only small dicts/WKT strings -- never a
    large array -- so it is safe to run in a `loky` worker process. All
    exceptions are caught; this function never raises.

    Args:
        spec: The frame to process (paths + parsed identity).
        cfg: Resolved pipeline configuration.
        pipeline_version: `grain_morph.__version__`, stamped onto every row.
        cfg_hash_value: `grain_morph.config.config_hash(cfg)`, stamped onto
            every row.

    Returns:
        A `FrameResult`. On any exception, `status="error"`,
        `error=<traceback text>`, and `rows`/`contours` are `[]`.
    """
    start = time.perf_counter()
    try:
        image = _read_grayscale(spec.path)
        blank = _read_grayscale(spec.blank_path) if spec.blank_path is not None else None
        corrected, flatfield_method = apply_flatfield(image, blank, cfg)
        label_image, detections = detect_objects(corrected, cfg)
        um_per_px = cfg.um_per_px(spec.parsed.camera)
        raw_image = image.astype(np.float64)
        frame_shape = (image.shape[0], image.shape[1])

        rows: list[dict[str, Any]] = []
        contours: list[dict[str, Any]] = []
        for det in detections:
            mask = label_image == det.label
            grain_uid = f"{spec.parsed.stem}:{det.label}"
            rows.append(
                _build_row(
                    det,
                    mask,
                    corrected,
                    raw_image,
                    frame_shape,
                    cfg,
                    um_per_px,
                    spec,
                    grain_uid,
                    pipeline_version,
                    cfg_hash_value,
                )
            )
            if det.polygon is not None:
                contours.append(
                    {"grain_uid": grain_uid, "wkt": det.polygon.wkt, "um_per_px": um_per_px}
                )

        return FrameResult(
            rows=rows,
            contours=contours,
            status="ok" if detections else "empty",
            n_objects=len(detections),
            seconds=time.perf_counter() - start,
            flatfield_method=flatfield_method,
            error=None,
        )
    except Exception:
        return FrameResult(
            rows=[],
            contours=[],
            status="error",
            n_objects=0,
            seconds=time.perf_counter() - start,
            flatfield_method="",
            error=traceback.format_exc(),
        )


def _load_records(path: Path, fmt: str) -> dict[str, dict[str, Any]]:
    """Load a small keyed table (manifest/errors) as `{frame_id: row}`.

    Args:
        path: Path to a table previously written by `write_table`.
        fmt: One of `"parquet"`, `"csv"`, `"feather"`.

    Returns:
        `{}` if `path` doesn't exist; otherwise its rows keyed by
        `frame_id`.
    """
    if not path.exists():
        return {}
    frame = read_table(path, fmt)
    return {str(row["frame_id"]): row for row in frame.to_dict("records")}


def run_detect(
    root: str | Path,
    out_dir: str | Path,
    cfg: Config,
    n_jobs: int | None = None,
    force: bool = False,
    *,
    sample_id: str | None = None,
    camera: str | None = None,
) -> None:
    """Run Stage 1 (detect) over every frame under `root`.

    Discovers frames (`io.discover_frames`), validates up front that every
    distinct camera among them is calibrated (`_validate_calibration` --
    raises before any work starts, rather than letting an uncalibrated
    camera masquerade as N per-frame errors), skips frames already
    recorded in `out_dir`'s manifest with a matching fingerprint (unless
    `force=True`), and fans `process_frame` out across the frames that
    still need work with `joblib.Parallel(backend="loky",
    return_as="generator")` -- consuming `FrameResult`s as they complete
    and immediately writing each completed frame's rows to its own
    deterministically-named path (`_write_or_clear_grain_frame`/
    `_write_or_clear_contour_frame`) rather than accumulating a whole
    run's rows in memory (design doc §11.2; this is what bounds peak RSS
    on long runs, and what makes reprocessing a frame overwrite exactly
    that frame's own output instead of appending a duplicate). The
    parallel section runs under `joblib.parallel_config(
    inner_max_num_threads=1)` so N worker processes don't each also spawn
    their own BLAS/OpenMP threads.

    `force=True` additionally deletes `grains/`, `contours/`, `manifest.
    <ext>`, and `errors.<ext>` up front, so a forced run is a true clean
    slate (including correctly dropping output for any frame that existed
    in an earlier run but no longer does).

    Writes, under `out_dir`:
    - `grains/` -- per-grain table (partitioned by `sample_id`, `camera`
      if `cfg.output.partition`), one file per frame. Never created if
      the whole run produced zero rows.
    - `contours/` -- geometry table (WKT + `um_per_px`, keyed on
      `grain_uid`), one file per frame, only when
      `cfg.output.save_contours`.
    - `manifest.<ext>` -- one row per frame (status, object count,
      flatfield method); `seconds` is always persisted as `0.0` so a
      resumed run's manifest bytes are reproducible (see module
      docstring).
    - `errors.<ext>` -- one row per frame that raised (`frame_id`,
      `frame_path`, `traceback`), carried forward across resumes for
      frames that are skipped rather than reprocessed. Deleted (not left
      stale) if this run ends with zero errors; not written at all if
      there were never any.
    - `run_config.yaml` -- the fully resolved `cfg`.
    - `summary.json` -- frame/object counts (cumulative, from the
      manifest) and QC-rejection breakdown (counted only from frames
      processed *in this call* -- a skipped/carried-forward frame's
      rejection breakdown isn't re-derived from its already-written grain
      rows, since the manifest doesn't store per-flag detail; see
      task-12 report) plus wall time.

    Args:
        root: Directory to search recursively for frames.
        out_dir: Destination directory for all run artifacts (created if
            missing).
        cfg: Resolved pipeline configuration.
        n_jobs: Overrides `cfg.runtime.n_jobs` for this call (e.g. CLI
            `--jobs`); `None` uses the config value. `1` runs everything
            in the calling process (no worker pool), which is what makes
            output deterministic for tests.
        force: If `True`, reprocess every frame regardless of the
            existing manifest, after clearing all prior run artifacts.
        sample_id: If given, forces this sample id on *every* discovered
            frame, ignoring any filename-parsed sample (see
            `io.parse_name`). Enables the "one directory = one sample"
            layout where the filenames carry no sample token.
        camera: If given, forces this camera (e.g. `"basic"`/`"zoom"`) on
            every discovered frame. Must be one of `cfg.filename.camera_map`'s
            values.

    Raises:
        ValueError: If `camera` is not one of the configured cameras
            (`cfg.filename.camera_map` values). Raised before `out_dir` is
            created or any frame is processed.
        KeyError: If any distinct camera among the discovered frames has
            no (or a still-`null`) calibration entry -- see
            `_validate_calibration`. Raised before `out_dir` is created or
            any frame is processed.
    """
    run_start = time.perf_counter()
    root = Path(root)
    out_dir = Path(out_dir)

    valid_cameras = set(cfg.filename.camera_map.values())
    if camera is not None and camera not in valid_cameras:
        raise ValueError(
            f"camera override {camera!r} is not one of the configured cameras "
            f"{sorted(valid_cameras)}"
        )

    specs = discover_frames(root, cfg, sample_id=sample_id, camera=camera)
    _validate_calibration(specs, cfg)  # whole-run precondition; raises loudly, nothing written yet

    out_dir.mkdir(parents=True, exist_ok=True)

    fmt = cfg.output.format
    ext = _EXTENSIONS[fmt]
    manifest_path = out_dir / f"manifest{ext}"
    errors_path = out_dir / f"errors{ext}"
    grains_dir = out_dir / "grains"
    contours_dir = out_dir / "contours"

    if force:
        # A forced run is a true clean slate: without this, a frame that
        # existed (and was written) in an earlier run but has since been
        # deleted from `root` would leave its old grain/contour/error rows
        # behind forever, since nothing would ever reprocess or clear them.
        shutil.rmtree(grains_dir, ignore_errors=True)
        shutil.rmtree(contours_dir, ignore_errors=True)
        manifest_path.unlink(missing_ok=True)
        errors_path.unlink(missing_ok=True)

    frame_infos = [(spec, spec.parsed.stem, frame_fingerprint(spec.path)) for spec in specs]

    existing_manifest: Manifest | None = None
    existing_error_rows: dict[str, dict[str, Any]] = {}
    if not force:
        if manifest_path.exists():
            existing_manifest = Manifest.load(manifest_path, fmt)
        existing_error_rows = _load_records(errors_path, fmt)

    def needs_processing(frame_id: str, fingerprint: str) -> bool:
        return existing_manifest is None or not existing_manifest.is_done(frame_id, fingerprint)

    todo = [
        (spec, frame_id, fp)
        for spec, frame_id, fp in frame_infos
        if needs_processing(frame_id, fp)
    ]
    existing_rows_by_id = (
        {r["frame_id"]: r for r in existing_manifest.to_frame().to_dict("records")}
        if existing_manifest is not None
        else {}
    )

    resolved_n_jobs = cfg.runtime.n_jobs if n_jobs is None else n_jobs
    pipeline_version = __version__
    cfg_hash_value = config_hash(cfg)
    grain_columns = _grain_columns(cfg.measure.efd_order)

    new_manifest = Manifest()
    error_rows: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    n_objects_total = 0
    rejection_counts: dict[str, int] = {}

    with joblib.parallel_config(backend="loky", inner_max_num_threads=1):
        if todo:
            parallel = joblib.Parallel(
                n_jobs=resolved_n_jobs, backend="loky", return_as="generator"
            )
            results_iter = iter(
                parallel(
                    joblib.delayed(process_frame)(spec, cfg, pipeline_version, cfg_hash_value)
                    for spec, _frame_id, _fp in todo
                )
            )
        else:
            results_iter = iter(())

        for spec, frame_id, fp in frame_infos:
            if needs_processing(frame_id, fp):
                result = next(results_iter)
                new_manifest.add(
                    frame_id,
                    str(spec.path),
                    result.status,
                    result.n_objects,
                    0.0,  # never persist wall-clock timing -- see module docstring
                    result.flatfield_method,
                    fingerprint=fp,
                )
                status_counts[result.status] = status_counts.get(result.status, 0) + 1
                n_objects_total += result.n_objects
                if result.status == "error" and result.error is not None:
                    error_rows.append(
                        {
                            "frame_id": frame_id,
                            "frame_path": str(spec.path),
                            "traceback": result.error,
                        }
                    )
                # Write (or clear) this frame's own output immediately -- at
                # most one frame's rows are ever resident in memory, and
                # reprocessing this frame_id always overwrites exactly its
                # own prior output (see `_write_or_clear_grain_frame`).
                _write_or_clear_grain_frame(
                    result.rows,
                    grains_dir,
                    fmt,
                    cfg.output.partition,
                    spec.parsed.sample_id,
                    spec.parsed.camera,
                    frame_id,
                    grain_columns,
                )
                if cfg.output.save_contours:
                    _write_or_clear_contour_frame(result.contours, contours_dir, fmt, frame_id)
                for row in result.rows:
                    for flag_name in cfg.qc.disqualifying_flags:
                        if row.get(flag_name):
                            rejection_counts[flag_name] = rejection_counts.get(flag_name, 0) + 1
            else:
                prior = existing_rows_by_id[frame_id]
                prior_status = str(prior["status"])
                new_manifest.add(
                    frame_id,
                    str(prior["frame_path"]),
                    prior_status,
                    int(prior["n_objects"]),
                    0.0,
                    str(prior["flatfield_method"]),
                    fingerprint=str(prior["fingerprint"]),
                )
                status_counts[prior_status] = status_counts.get(prior_status, 0) + 1
                n_objects_total += int(prior["n_objects"])
                if prior_status == "error" and frame_id in existing_error_rows:
                    error_rows.append(existing_error_rows[frame_id])
                # Not reprocessed -> its existing grain/contour file (if
                # any) is untouched, exactly as an earlier run left it.

    new_manifest.save(manifest_path, fmt)
    if error_rows:
        errors_df = pd.DataFrame(error_rows).reindex(
            columns=["frame_id", "frame_path", "traceback"]
        )
        write_table(errors_df, errors_path, fmt)
    else:
        # No errors this run (e.g. every previously-erroring frame was
        # fixed and reprocessed) -- don't leave a stale errors table with
        # rows for frames that no longer error.
        errors_path.unlink(missing_ok=True)

    (out_dir / "run_config.yaml").write_text(yaml.safe_dump(cfg.model_dump(), sort_keys=False))
    summary = {
        "n_frames": len(frame_infos),
        "n_frames_by_status": status_counts,
        "n_objects": n_objects_total,
        # NOTE: rejection_counts is accumulated only from frames processed
        # in *this* call (see the todo-branch loop above), not re-derived
        # from already-written grain rows for skipped/carried-forward
        # frames -- on a resume that skips most frames, this understates
        # the true whole-dataset rejection breakdown. summary.json is a
        # diagnostic artifact only; nothing depends on it being exact
        # across resumes (unlike `manifest`, which is byte-identical by
        # construction -- see module docstring).
        "rejection_counts": rejection_counts,
        "wall_seconds": time.perf_counter() - run_start,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str)
    )
