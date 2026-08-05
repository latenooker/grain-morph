"""Tests for `grain_morph.overlay.make_overlays` and the `overlay` CLI command.

Built without the full `detect` pipeline for the `overlay.py`-level tests
(a tiny synthetic frame + hand-built `grains`/`contours` DataFrames is
enough to exercise the rendering logic); the CLI test runs a real (tiny)
`detect` first, since it needs an on-disk run directory in the exact shape
`overlay`'s own `_read_grains`/contour-file reading expects.
"""

from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import matplotlib.axes
import numpy as np
import pandas as pd
import yaml
from typer.testing import CliRunner

from grain_morph.cli import app
from grain_morph.config import load_config
from grain_morph.overlay import (
    _ACCEPTED_COLOR,
    _LABEL_COUNT_CAP,
    _NO_POLYGON_MARKER_COLOR_RGB,
    _REJECTED_COLOR,
    _compose_outlines_full_res,
    _should_draw_labels,
    make_overlays,
)
from tests.synth import make_frame

runner = CliRunner()

_FRAME_SIZE = 120
_BACKGROUND_LEVEL = 200

# Ground-truth polygons for the two synthetic grains, in full-res pixel
# (x y) = (col row) WKT coordinates, matching what `detect`'s real pipeline
# would persist in `contours`.
_SQUARE_WKT = "POLYGON ((20 20, 40 20, 40 40, 20 40, 20 20))"
_TRIANGLE_WKT = "POLYGON ((60 60, 85 60, 70 90, 60 60))"


def _write_frame(path: Path, size: int = _FRAME_SIZE, level: int = _BACKGROUND_LEVEL) -> None:
    frame = np.full((size, size), level, np.uint8)
    iio.imwrite(path, frame)


def _base_grains(frame_id: str, frame_path: Path | str) -> pd.DataFrame:
    """Two grains for `frame_id`: one accepted, one rejected (flag_defocus)."""
    return pd.DataFrame({
        "frame_id": [frame_id, frame_id],
        "grain_uid": [f"{frame_id}:1", f"{frame_id}:2"],
        "frame_path": [str(frame_path)] * 2,
        "qc_pass": [True, False],
        "centroid_x": [30.0, 71.0],
        "centroid_y": [30.0, 71.0],
        "flag_no_polygon": [False, False],
        "flag_defocus": [False, True],
    })


def _base_contours(frame_id: str) -> pd.DataFrame:
    return pd.DataFrame({
        "grain_uid": [f"{frame_id}:1", f"{frame_id}:2"],
        "wkt": [_SQUARE_WKT, _TRIANGLE_WKT],
        "um_per_px": [5.0, 5.0],
    })


# --- `_compose_outlines_full_res` (unit-level, exact color check) --------


def test_compose_outlines_uses_distinct_colors_by_qc_outcome():
    frame_id = "f1"
    image = np.full((_FRAME_SIZE, _FRAME_SIZE), _BACKGROUND_LEVEL, np.uint8)
    grains = _base_grains(frame_id, "unused.bmp")
    contours = _base_contours(frame_id)

    rgb = _compose_outlines_full_res(image, grains, contours, factor=4)

    assert rgb.shape == (_FRAME_SIZE, _FRAME_SIZE, 3)
    assert rgb.dtype == np.uint8
    accepted_present = bool(np.any(np.all(rgb == np.array(_ACCEPTED_COLOR), axis=-1)))
    rejected_present = bool(np.any(np.all(rgb == np.array(_REJECTED_COLOR), axis=-1)))
    assert accepted_present, "accepted (qc_pass=True) grain outline color missing"
    assert rejected_present, "rejected (qc_pass=False) grain outline color missing"


# --- `make_overlays` end-to-end -------------------------------------------


def test_make_overlays_writes_nontrivial_png(tmp_path):
    frame_id = "f1"
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    frame_path = frames_dir / f"{frame_id}.bmp"
    _write_frame(frame_path)

    grains = _base_grains(frame_id, frame_path)
    contours = _base_contours(frame_id)

    out_dir = tmp_path / "overlays"
    paths = make_overlays(
        grains,
        contours,
        frames_root=frames_dir,
        out_dir=out_dir,
        frame_ids=[frame_id],
        factor=4,
        cfg=load_config(None),
    )

    assert len(paths) == 1
    out_path = paths[0]
    assert out_path.name == f"{frame_id}_overlay.png"
    assert out_path.exists()
    assert out_path.stat().st_size > 3_000  # non-trivial, not a blank/degenerate PNG

    arr = np.asarray(iio.imread(out_path))
    assert arr.ndim == 3  # RGB(A) figure render
    # Non-uniform: outlines + title/labels means more than a flat fill.
    assert np.unique(arr.reshape(-1, arr.shape[-1]), axis=0).shape[0] > 10


def test_make_overlays_saved_png_contains_both_qc_colors(tmp_path):
    """Sample the saved PNG itself (not just the pre-annotation array) for
    both accepted and rejected outline colors, with a small tolerance for
    matplotlib's `imshow` interpolation/resampling of the underlying RGB."""
    frame_id = "f1"
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    frame_path = frames_dir / f"{frame_id}.bmp"
    _write_frame(frame_path)

    grains = _base_grains(frame_id, frame_path)
    contours = _base_contours(frame_id)

    out_dir = tmp_path / "overlays"
    paths = make_overlays(
        grains,
        contours,
        frames_root=frames_dir,
        out_dir=out_dir,
        frame_ids=[frame_id],
        factor=4,
        cfg=load_config(None),
    )
    arr = np.asarray(iio.imread(paths[0]))[..., :3].astype(np.int16)

    def _close_to(color: tuple[int, int, int], tol: int = 40) -> bool:
        dist = np.abs(arr - np.array(color)).sum(axis=-1)
        return bool(np.any(dist < tol))

    assert _close_to(_ACCEPTED_COLOR), "no accepted-color pixels found in saved PNG"
    assert _close_to(_REJECTED_COLOR), "no rejected-color pixels found in saved PNG"


def test_frame_with_no_grains_is_skipped_without_crash(tmp_path):
    frame_id = "f1"
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    frame_path = frames_dir / f"{frame_id}.bmp"
    _write_frame(frame_path)

    grains = _base_grains(frame_id, frame_path)
    contours = _base_contours(frame_id)

    out_dir = tmp_path / "overlays"
    paths = make_overlays(
        grains,
        contours,
        frames_root=frames_dir,
        out_dir=out_dir,
        frame_ids=[frame_id, "does_not_exist"],
        factor=4,
        cfg=load_config(None),
    )

    assert len(paths) == 1  # only the real frame produced a PNG
    assert paths[0].exists()


def test_frame_with_unresolvable_image_is_skipped_without_crash(tmp_path):
    frame_id = "ghost"
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    grains = _base_grains(frame_id, "/nonexistent/path/ghost.bmp")
    contours = _base_contours(frame_id)

    out_dir = tmp_path / "overlays"
    paths = make_overlays(
        grains,
        contours,
        frames_root=frames_dir,
        out_dir=out_dir,
        frame_ids=[frame_id],
        factor=4,
        cfg=load_config(None),
    )

    assert paths == []
    assert not out_dir.exists() or not any(out_dir.iterdir())


def test_flag_no_polygon_grain_does_not_crash_and_gets_marker(tmp_path):
    frame_id = "f1"
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    frame_path = frames_dir / f"{frame_id}.bmp"
    _write_frame(frame_path)

    grains = _base_grains(frame_id, frame_path)
    extra = pd.DataFrame({
        "frame_id": [frame_id],
        "grain_uid": [f"{frame_id}:3"],
        "frame_path": [str(frame_path)],
        "qc_pass": [False],
        "centroid_x": [100.0],
        "centroid_y": [15.0],
        "flag_no_polygon": [True],
        "flag_defocus": [False],
    })
    grains = pd.concat([grains, extra], ignore_index=True)
    contours = _base_contours(frame_id)  # deliberately no contour row for grain 3

    out_dir = tmp_path / "overlays"
    paths = make_overlays(
        grains,
        contours,
        frames_root=frames_dir,
        out_dir=out_dir,
        frame_ids=[frame_id],
        factor=4,
        cfg=load_config(None),
    )

    assert len(paths) == 1
    assert paths[0].exists()

    arr = np.asarray(iio.imread(paths[0]))[..., :3].astype(np.int16)
    dist = np.abs(arr - np.array(_NO_POLYGON_MARKER_COLOR_RGB)).sum(axis=-1)
    assert bool(np.any(dist < 40)), "no flag_no_polygon marker color found in saved PNG"


def test_malformed_wkt_is_skipped_without_crash(tmp_path):
    """One grain's WKT is garbage -- its outline is skipped, not a crash,
    and the other (well-formed) grain's outline still renders."""
    frame_id = "f1"
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    frame_path = frames_dir / f"{frame_id}.bmp"
    _write_frame(frame_path)

    grains = _base_grains(frame_id, frame_path)
    contours = pd.DataFrame({
        "grain_uid": [f"{frame_id}:1", f"{frame_id}:2"],
        "wkt": ["NOT A VALID POLYGON WKT", _TRIANGLE_WKT],
        "um_per_px": [5.0, 5.0],
    })

    out_dir = tmp_path / "overlays"
    paths = make_overlays(
        grains,
        contours,
        frames_root=frames_dir,
        out_dir=out_dir,
        frame_ids=[frame_id],
        factor=4,
        cfg=load_config(None),
    )

    assert len(paths) == 1
    assert paths[0].exists()
    # grain 2 (rejected, well-formed WKT) still gets its outline even though
    # grain 1 (accepted, malformed WKT) was skipped.
    arr = np.asarray(iio.imread(paths[0]))[..., :3].astype(np.int16)
    dist = np.abs(arr - np.array(_REJECTED_COLOR)).sum(axis=-1)
    assert bool(np.any(dist < 40)), "well-formed grain's outline missing after malformed-WKT skip"


# --- Label-count cap (`--labels`/`--no-labels`, `labels: bool | None`) ---


def _many_grains(frame_id: str, frame_path: Path | str, n: int) -> pd.DataFrame:
    """`n` grains for `frame_id`, scattered centroids -- no contours needed
    since these tests exercise labels, not outlines."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "frame_id": [frame_id] * n,
        "grain_uid": [f"{frame_id}:{i}" for i in range(n)],
        "frame_path": [str(frame_path)] * n,
        "qc_pass": [i % 2 == 0 for i in range(n)],
        "centroid_x": rng.uniform(10, _FRAME_SIZE - 10, n),
        "centroid_y": rng.uniform(10, _FRAME_SIZE - 10, n),
        "flag_no_polygon": [False] * n,
        "flag_defocus": [i % 2 != 0 for i in range(n)],
    })


def _spy_on_axes_text(monkeypatch) -> list[tuple]:
    """Patch `matplotlib.axes.Axes.text` to record every call it receives.

    Lets a test assert exactly how many per-grain text labels `_annotate`
    drew, end-to-end through the public `make_overlays` API, without
    parsing rendered pixels.
    """
    calls: list[tuple] = []
    original_text = matplotlib.axes.Axes.text

    def spy_text(self, *args, **kwargs):
        calls.append(args)
        return original_text(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "text", spy_text)
    return calls


def test_should_draw_labels_tri_state():
    assert _should_draw_labels(None, _LABEL_COUNT_CAP) is True
    assert _should_draw_labels(None, _LABEL_COUNT_CAP + 1) is False
    assert _should_draw_labels(True, _LABEL_COUNT_CAP + 1) is True
    assert _should_draw_labels(False, 1) is False


def test_labels_suppressed_above_cap_by_default(tmp_path, monkeypatch):
    frame_id = "f1"
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    frame_path = frames_dir / f"{frame_id}.bmp"
    _write_frame(frame_path)

    n = _LABEL_COUNT_CAP + 10
    grains = _many_grains(frame_id, frame_path, n)
    contours = pd.DataFrame(columns=["grain_uid", "wkt", "um_per_px"])
    calls = _spy_on_axes_text(monkeypatch)

    out_dir = tmp_path / "overlays"
    paths = make_overlays(
        grains,
        contours,
        frames_root=frames_dir,
        out_dir=out_dir,
        frame_ids=[frame_id],
        factor=4,
        cfg=load_config(None),
        # labels omitted -> default None -> auto, suppressed above the cap
    )

    assert len(paths) == 1
    assert calls == [], "per-grain labels should be suppressed by default above the cap"


def test_labels_true_forces_labels_on_above_cap(tmp_path, monkeypatch):
    frame_id = "f1"
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    frame_path = frames_dir / f"{frame_id}.bmp"
    _write_frame(frame_path)

    n = _LABEL_COUNT_CAP + 10
    grains = _many_grains(frame_id, frame_path, n)
    contours = pd.DataFrame(columns=["grain_uid", "wkt", "um_per_px"])
    calls = _spy_on_axes_text(monkeypatch)

    out_dir = tmp_path / "overlays"
    paths = make_overlays(
        grains,
        contours,
        frames_root=frames_dir,
        out_dir=out_dir,
        frame_ids=[frame_id],
        factor=4,
        cfg=load_config(None),
        labels=True,
    )

    assert len(paths) == 1
    assert len(calls) == n, "labels=True should force a label for every grain regardless of cap"


# --- CLI --------------------------------------------------------------


def _detect_small_run(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Run a tiny real `detect` (one ellipse grain) via the CLI.

    Returns:
        `(run_dir, out_dir, config_path)` -- `out_dir` is the completed
        `detect` output (`grains/`, `contours/`), `run_dir` the source
        frames directory.
    """
    run = tmp_path / "run"
    run.mkdir()
    iio.imwrite(run / "S1_b_back.bmp", make_frame(size=(128, 128), objects=[]).image)
    objects = [
        {"kind": "ellipse", "cx": 64, "cy": 64, "a": 18, "b": 18, "angle": 0.0, "blur_sigma": 0.0}
    ]
    iio.imwrite(run / "S1_b_0000001.bmp", make_frame(size=(128, 128), objects=objects).image)
    cfg = tmp_path / "c.yaml"
    calibration = {"calibration": {"um_per_px": {"basic": 5.0, "zoom": 1.0}}}
    cfg.write_text(yaml.safe_dump(calibration))
    out = tmp_path / "out"
    res = runner.invoke(app, ["detect", str(run), str(out), "--config", str(cfg), "--jobs", "1"])
    assert res.exit_code == 0, res.output
    return run, out, cfg


def test_overlay_cli_end_to_end(tmp_path):
    run, out, cfg = _detect_small_run(tmp_path)

    overlay_out = tmp_path / "overlay"
    args = [
        "overlay",
        str(out),
        str(run),
        str(overlay_out),
        "--frames",
        "S1_b_0000001",
        "--config",
        str(cfg),
    ]
    res = runner.invoke(app, args)
    assert res.exit_code == 0, res.output
    assert (overlay_out / "S1_b_0000001_overlay.png").exists()


def test_overlay_cli_missing_frames_option_gives_clear_message(tmp_path):
    run, out, cfg = _detect_small_run(tmp_path)

    overlay_out = tmp_path / "overlay"
    args = ["overlay", str(out), str(run), str(overlay_out), "--config", str(cfg)]
    res = runner.invoke(app, args)
    assert res.exit_code != 0
    assert "--frames" in res.output
    assert not overlay_out.exists()


def test_overlay_cli_help_lists_tristate_labels_flag():
    res = runner.invoke(app, ["overlay", "--help"])
    assert res.exit_code == 0
    assert "--labels" in res.output
    assert "--no-labels" in res.output
