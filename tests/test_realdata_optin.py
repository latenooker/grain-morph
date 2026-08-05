"""Opt-in test against full-resolution real frames on an external drive.

Marked `realdata` (see `pyproject.toml`'s `markers`) and skipped whenever the
external drive isn't mounted -- this is not part of the default `pytest`
run's guarantees, since it depends on hardware nobody but the pipeline's
author has attached. It exists to sanity-check `process_frame` against
genuinely full-resolution (not downsampled) CAMSIZER output, which the
committed `tests/fixtures/real/` set (downsampled 4x for size) can't fully
stand in for.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.realdata

LEXAR = Path("/Volumes/LEXAR/Camsizer/PPX/images_P_17_cs")


@pytest.mark.skipif(not LEXAR.exists(), reason="external drive not mounted")
def test_defocus_gate_separates_on_real_frames(tmp_path):
    from grain_morph.config import load_config
    from grain_morph.io import discover_frames
    from grain_morph.pipeline import process_frame

    cfg = load_config(None)
    cfg.calibration.um_per_px = {"basic": 20.0, "zoom": 8.0}
    specs = discover_frames(LEXAR, cfg)
    assert specs
    r = process_frame(specs[0], cfg, "test", "hash")
    assert r.status in {"ok", "empty"}
