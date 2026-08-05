from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import pandas as pd

from grain_morph.config import load_config
from grain_morph.pipeline import run_detect
from grain_morph.writers import read_table
from tests.synth import make_frame


def _cfg_with_calib():
    cfg = load_config(None)
    cfg.calibration.um_per_px = {"basic": 5.0, "zoom": 1.0}
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
