from __future__ import annotations

import imageio.v3 as iio
import yaml
from typer.testing import CliRunner

from grain_morph.cli import app
from tests.synth import make_frame

runner = CliRunner()


def test_detect_cli_end_to_end(tmp_path):
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
    args = ["detect", str(run), str(out), "--config", str(cfg), "--jobs", "1"]
    res = runner.invoke(app, args)
    assert res.exit_code == 0, res.output
    assert (out / "summary.json").exists()
    assert (out / "run_config.yaml").exists()
