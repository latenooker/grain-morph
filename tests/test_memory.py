from __future__ import annotations

import imageio.v3 as iio
import psutil

from grain_morph.config import load_config
from grain_morph.pipeline import run_detect
from tests.synth import make_frame


def test_500_frames_bounded_rss(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    iio.imwrite(run / "S1_b_back.bmp", make_frame(size=(256, 256), objects=[]).image)
    for i in range(1, 501):
        f = make_frame(
            size=(256, 256),
            seed=i,
            objects=[
                {
                    "kind": "ellipse", "cx": 128, "cy": 128, "a": 18, "b": 15,
                    "angle": 0.0, "blur_sigma": 0.0,
                }
            ],
        )
        iio.imwrite(run / f"S1_b_{i:07d}.bmp", f.image)
    cfg = load_config(None)
    cfg.calibration.um_per_px = {"basic": 5.0, "zoom": 1.0}
    proc = psutil.Process()
    before = proc.memory_info().rss
    run_detect(run, tmp_path / "out", cfg, n_jobs=1)
    after = proc.memory_info().rss
    assert (after - before) < 300 * 1024 * 1024  # < 300 MB growth
