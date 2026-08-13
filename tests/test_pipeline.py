from __future__ import annotations

import os
from pathlib import Path

import imageio.v3 as iio
import pandas as pd
import pytest

from grain_morph.config import load_config
from grain_morph.pipeline import run_detect
from grain_morph.writers import read_table
from tests.synth import make_frame


def _cfg_with_calib():
    cfg = load_config(None)
    cfg.calibration.um_per_px = {"basic": 5.0, "zoom": 1.0}
    # These tests assert on parquet output specifics (hive-partitioned grains
    # dir, `manifest.parquet` byte-reproducibility, force-clean of a partitioned
    # dataset), so pin the format here rather than ride the packaged default
    # (now csv). The csv default path is covered by the CLI/groundtruth tests.
    cfg.output.format = "parquet"
    return cfg


def _write_run(dirpath: Path, gradient=0.0):
    dirpath.mkdir(parents=True, exist_ok=True)
    # blank
    blank = make_frame(size=(256, 256), objects=[], gradient=gradient).image
    iio.imwrite(dirpath / "S1_b_back.bmp", blank)
    # one frame, object in the BRIGHT corner and one in the DARK corner (same size)
    f = make_frame(
        size=(256, 256),
        gradient=gradient,
        objects=[
            {
                "kind": "ellipse", "cx": 40, "cy": 128, "a": 25, "b": 25,
                "angle": 0.0, "blur_sigma": 0.0,
            },
            {
                "kind": "ellipse", "cx": 216, "cy": 128, "a": 25, "b": 25,
                "angle": 0.0, "blur_sigma": 0.0,
            },
        ],
    )
    iio.imwrite(dirpath / "S1_b_0000001.bmp", f.image)
    return f


def test_flatfield_area_agreement_bright_vs_dark(tmp_path):
    # ACCEPTANCE CRITERION 2
    run = tmp_path / "run"
    _write_run(run, gradient=0.4)
    out = tmp_path / "out"
    run_detect(run, out, _cfg_with_calib(), n_jobs=1)
    if (out / "grains.parquet").exists():
        df = read_table(out / "grains.parquet", "parquet")
    else:
        df = pd.read_parquet(out / "grains")
    df = df.sort_values("centroid_x")
    areas = df["area_um2"].tolist()
    assert len(areas) == 2
    assert abs(areas[0] - areas[1]) / max(areas) < 0.02


def test_zero_particle_frame(tmp_path):
    # ACCEPTANCE CRITERION 6
    run = tmp_path / "run"
    run.mkdir()
    iio.imwrite(run / "S1_b_back.bmp", make_frame(size=(128, 128), objects=[]).image)
    iio.imwrite(run / "S1_b_0000001.bmp", make_frame(size=(128, 128), objects=[]).image)
    out = tmp_path / "out"
    run_detect(run, out, _cfg_with_calib(), n_jobs=1)
    from grain_morph.io import Manifest
    man = Manifest.load(out / "manifest.parquet", "parquet")
    assert len(man.to_frame()) == 1
    # no grains file OR an empty one
    gdir = out / "grains"
    n_grain_rows = sum(len(read_table(p, "parquet")) for p in gdir.rglob("*.parquet"))
    assert (not gdir.exists()) or n_grain_rows == 0


def test_resume_does_no_work(tmp_path):
    # ACCEPTANCE CRITERION 7
    run = tmp_path / "run"
    _write_run(run)
    out = tmp_path / "out"
    run_detect(run, out, _cfg_with_calib(), n_jobs=1)
    man1 = (out / "manifest.parquet").read_bytes()
    run_detect(run, out, _cfg_with_calib(), n_jobs=1)  # resume
    man2 = (out / "manifest.parquet").read_bytes()
    assert man1 == man2  # byte-identical; no reprocessing


def test_force_rerun_does_not_duplicate_rows(tmp_path):
    # Review-fix regression test (CRITICAL bug 1a): a random-token chunk
    # name per run_detect call, never cleaned up, meant a `force=True`
    # re-run against the same out_dir left the prior run's chunk files on
    # disk *and* wrote a fresh set, doubling `pd.read_parquet(grains_root)`
    # row counts (confirmed 2 -> 4 rows before the fix). Deterministic
    # per-frame output paths mean reprocessing a frame overwrites exactly
    # its own file, so a force re-run must leave the row count unchanged.
    run = tmp_path / "run"
    _write_run(run)  # one frame, 2 objects
    out = tmp_path / "out"

    run_detect(run, out, _cfg_with_calib(), n_jobs=1)
    df1 = pd.read_parquet(out / "grains")
    assert len(df1) == 2

    run_detect(run, out, _cfg_with_calib(), n_jobs=1, force=True)
    df2 = pd.read_parquet(out / "grains")
    assert len(df2) == 2  # unchanged, not doubled to 4
    assert sorted(df2["grain_uid"]) == sorted(df1["grain_uid"])


def test_resume_after_frame_edit_replaces_not_duplicates(tmp_path):
    # Review-fix regression test (CRITICAL bug 1b): resuming after editing
    # one frame's pixels (so its fingerprint changes) must REPLACE that
    # frame's grain rows, not add a second copy alongside the stale ones
    # (confirmed 1 -> 2 rows accumulating to 3 before the fix).
    run = tmp_path / "run"
    run.mkdir()
    iio.imwrite(run / "S1_b_back.bmp", make_frame(size=(256, 256), objects=[]).image)
    frame_path = run / "S1_b_0000001.bmp"
    f1 = make_frame(
        size=(256, 256),
        objects=[{"kind": "ellipse", "cx": 128, "cy": 128, "a": 25, "b": 25, "angle": 0.0}],
    )
    iio.imwrite(frame_path, f1.image)

    out = tmp_path / "out"
    run_detect(run, out, _cfg_with_calib(), n_jobs=1)
    df1 = pd.read_parquet(out / "grains")
    assert len(df1) == 1

    # Overwrite the SAME filename with a different frame (now 2 objects).
    # Explicitly bump mtime so `frame_fingerprint` (size + int(mtime))
    # changes regardless of same-second filesystem timestamp resolution.
    f2 = make_frame(
        size=(256, 256),
        objects=[
            {"kind": "ellipse", "cx": 60, "cy": 128, "a": 20, "b": 20, "angle": 0.0},
            {"kind": "ellipse", "cx": 196, "cy": 128, "a": 20, "b": 20, "angle": 0.0},
        ],
    )
    iio.imwrite(frame_path, f2.image)
    new_mtime = frame_path.stat().st_mtime + 5.0
    os.utime(frame_path, (new_mtime, new_mtime))

    run_detect(run, out, _cfg_with_calib(), n_jobs=1)  # resume: only this frame changed
    df2 = pd.read_parquet(out / "grains")
    assert len(df2) == 2  # replaced, not accumulated to 1 + 2 = 3
    assert sorted(df2["grain_uid"]) == ["S1_b_0000001:1", "S1_b_0000001:2"]


def test_overrides_stamp_identity_on_structureless_frames(tmp_path):
    # All frames in one dir, named only by index (+ a `back` blank): identity
    # comes from the run_detect overrides, and the grain rows carry it.
    run = tmp_path / "run"
    run.mkdir()
    iio.imwrite(run / "back.bmp", make_frame(size=(256, 256), objects=[]).image)
    obj = [{"kind": "ellipse", "cx": 128, "cy": 128, "a": 25, "b": 25,
            "angle": 0.0, "blur_sigma": 0.0}]
    iio.imwrite(run / "0000001.bmp", make_frame(size=(256, 256), objects=obj).image)
    out = tmp_path / "out"

    run_detect(run, out, _cfg_with_calib(), n_jobs=1, sample_id="P1", camera="basic")

    if (out / "grains.parquet").exists():
        df = read_table(out / "grains.parquet", "parquet")
    else:
        df = pd.read_parquet(out / "grains")
    assert len(df) >= 1
    assert set(df["sample_id"].astype(str)) == {"P1"}
    assert set(df["camera"].astype(str)) == {"basic"}


def test_invalid_camera_override_raises_before_processing(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    iio.imwrite(run / "0000001.bmp", make_frame(size=(64, 64), objects=[]).image)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="camera"):
        run_detect(run, out, _cfg_with_calib(), n_jobs=1, sample_id="P1", camera="telescope")
    assert not out.exists()


def test_uncalibrated_camera_raises_before_processing(tmp_path):
    # Review-fix regression test (Important bug 2): an uncalibrated camera
    # is a whole-run config precondition, not per-frame data corruption --
    # run_detect must raise loudly (and write nothing) rather than letting
    # `process_frame`'s per-frame isolation turn it into a silent, exit-0
    # run full of `status="error"` frames.
    run = tmp_path / "run"
    run.mkdir()
    iio.imwrite(run / "S1_b_back.bmp", make_frame(size=(64, 64), objects=[]).image)
    iio.imwrite(run / "S1_b_0000001.bmp", make_frame(size=(64, 64), objects=[]).image)
    out = tmp_path / "out"
    cfg = load_config(None)  # default calibration is null for every camera

    with pytest.raises(KeyError, match="calibration"):
        run_detect(run, out, cfg, n_jobs=1)

    assert not out.exists()  # aborted before any output was written
