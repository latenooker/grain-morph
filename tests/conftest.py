from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest


def _write_bmp(path: Path, arr: np.ndarray) -> None:
    iio.imwrite(path, arr.astype(np.uint8))


@pytest.fixture
def frame_dir(tmp_path: Path) -> Path:
    """A directory with two data frames + one blank for one (sample, camera)."""
    d = tmp_path / "frames"
    d.mkdir()
    bg = np.full((64, 64), 200, np.uint8)
    _write_bmp(d / "S1_b_0000001.bmp", bg)
    _write_bmp(d / "S1_b_0000002.bmp", bg)
    _write_bmp(d / "S1_b_back.bmp", bg)
    return d
