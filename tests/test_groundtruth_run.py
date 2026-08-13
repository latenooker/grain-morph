from __future__ import annotations

import imageio.v3 as iio
import pytest
import yaml

from grain_morph.config import load_config
from grain_morph.groundtruth import _LABEL_COLUMNS, run_groundtruth
from grain_morph.pipeline import run_detect
from tests.synth import make_frame


def _detect_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    iio.imwrite(run / "S1_b_back.bmp", make_frame(size=(256, 256), objects=[]).image)
    obj = [{"kind": "ellipse", "cx": 128, "cy": 128, "a": 25, "b": 25,
            "angle": 0.0, "blur_sigma": 0.0}]
    iio.imwrite(run / "S1_b_0000001.bmp", make_frame(size=(256, 256), objects=obj).image)
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump({"calibration": {"um_per_px": {"basic": 5.0, "zoom": 1.0}}}))
    out = tmp_path / "out"
    run_detect(run, out, load_config(cfg_path), n_jobs=1)
    return out, cfg_path


def test_run_groundtruth_no_show_builds_labels_frame(tmp_path):
    # Exercises the whole non-GUI path (load grains+contours, sample, build the
    # session) on a real detect run, without opening a window.
    out, cfg_path = _detect_run(tmp_path)
    df = run_groundtruth(out, n_grains=1, seed=0, config=cfg_path, show=False)
    assert list(df.columns) == list(_LABEL_COLUMNS)
    assert len(df) == 0  # nothing labeled yet


def test_run_groundtruth_missing_grains_raises(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        run_groundtruth(empty, n_grains=1, show=False)
