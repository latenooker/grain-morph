from __future__ import annotations

from grain_morph.config import load_config
from grain_morph.io import parse_name


def _cfg():
    return load_config(None)


def test_parse_basic_data_frame():
    p = parse_name("P_01_cs_003_b_0004258.bmp", _cfg())
    assert p is not None
    assert p.sample_id == "P_01_cs_003"
    assert p.camera == "basic"
    assert p.frame_index == 4258
    assert p.is_blank is False
    assert p.stem == "P_01_cs_003_b_0004258"


def test_parse_zoom_blank():
    p = parse_name("P_17_cs_002_z_back.bmp", _cfg())
    assert p is not None
    assert p.sample_id == "P_17_cs_002"
    assert p.camera == "zoom"
    assert p.is_blank is True
    assert p.frame_index is None


def test_parse_generic_sample_name():
    # sample naming is user-defined; only camera/frame/back tokens are fixed
    p = parse_name("OK_sand_2_b_0000042.bmp", _cfg())
    assert p is not None
    assert p.sample_id == "OK_sand_2"
    assert p.camera == "basic"
    assert p.frame_index == 42


def test_non_matching_returns_none():
    assert parse_name("notes.txt", _cfg()) is None
