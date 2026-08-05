from __future__ import annotations

from grain_morph.config import load_config
from grain_morph.io import discover_frames


def test_discovers_data_frames_only(frame_dir):
    specs = discover_frames(frame_dir, load_config(None))
    assert len(specs) == 2  # blank excluded from the data set
    assert all(s.parsed.is_blank is False for s in specs)


def test_blank_paired_to_data_frames(frame_dir):
    specs = discover_frames(frame_dir, load_config(None))
    for s in specs:
        assert s.blank_path is not None
        assert s.blank_path.name == "S1_b_back.bmp"


def test_missing_blank_yields_none(tmp_path):
    import imageio.v3 as iio
    import numpy as np

    iio.imwrite(tmp_path / "S2_z_0000005.bmp", np.full((16, 16), 200, np.uint8))
    specs = discover_frames(tmp_path, load_config(None))
    assert len(specs) == 1
    assert specs[0].blank_path is None
