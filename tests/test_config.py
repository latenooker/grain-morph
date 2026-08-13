from __future__ import annotations

import pytest

from grain_morph.config import Config, config_hash, load_config


def test_load_default_returns_config():
    cfg = load_config(None)
    assert isinstance(cfg, Config)
    assert cfg.threshold.method == "half_max"
    assert cfg.output.format == "csv"


def test_uncalibrated_camera_fails_loudly():
    cfg = load_config(None)  # default calibration is null
    with pytest.raises(KeyError, match="calibration"):
        cfg.um_per_px("basic")


def test_calibration_lookup(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("calibration:\n  um_per_px:\n    basic: 5.0\n    zoom: 1.0\n")
    cfg = load_config(p)
    assert cfg.um_per_px("basic") == 5.0
    assert cfg.um_per_px("zoom") == 1.0


def test_config_hash_is_deterministic_and_sensitive():
    a = load_config(None)
    b = load_config(None)
    assert config_hash(a) == config_hash(b)
    a.threshold.half_max_fraction = 0.6
    assert config_hash(a) != config_hash(b)
