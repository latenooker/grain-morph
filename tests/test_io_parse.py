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


def test_run_group_populates_run_field(tmp_path):
    """An optional `run` regex group is parsed into `ParsedName.run`.

    Default schema has no `run` group -> run is None and the run token stays
    folded into sample_id (backward-compatible). A run-separating regex splits
    them, so `run` is populated and `sample_id` excludes it.
    """
    import yaml

    # default: no run group
    p = parse_name("P_01_cs_003_b_0001189.bmp", _cfg())
    assert p.run is None
    assert p.sample_id == "P_01_cs_003"

    # run-separating regex
    regex = r"(?P<sample>.+)_(?P<run>\d+)_(?P<cam>[bz])_(?:(?P<frame>\d+)|back)$"
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump({"filename": {"sample_regex": regex}}))
    p2 = parse_name("P_01_cs_003_b_0001189.bmp", load_config(cfg_path))
    assert p2.run == "003"
    assert p2.sample_id == "P_01_cs"
