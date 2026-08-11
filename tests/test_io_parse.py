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


def test_override_sample_id_on_matching_name():
    # sample_id override replaces the filename's captured sample; camera and
    # frame still come from the filename.
    p = parse_name("S1_b_0000001.bmp", _cfg(), sample_id="P_9")
    assert p is not None
    assert p.sample_id == "P_9"
    assert p.camera == "basic"
    assert p.frame_index == 1
    assert p.is_blank is False


def test_override_camera_on_matching_name():
    # camera override replaces the filename's captured camera; sample stays.
    p = parse_name("S1_b_0000001.bmp", _cfg(), camera="zoom")
    assert p is not None
    assert p.sample_id == "S1"
    assert p.camera == "zoom"
    assert p.frame_index == 1


def test_override_both_on_structureless_name():
    # A stem that does NOT match sample_regex is still processable when both
    # identity fields are supplied; frame index comes from trailing digits.
    p = parse_name("0000042.png", _cfg(), sample_id="P1", camera="basic")
    assert p is not None
    assert p.sample_id == "P1"
    assert p.camera == "basic"
    assert p.frame_index == 42
    assert p.is_blank is False
    assert p.run is None


def test_override_fallback_blank_via_blank_regex():
    # In the override/structureless path, a blank is recognized by blank_regex.
    p = parse_name("back.png", _cfg(), sample_id="P1", camera="basic")
    assert p is not None
    assert p.is_blank is True
    assert p.frame_index is None


def test_single_override_on_structureless_returns_none():
    # One override is not enough to resolve a structureless name: the other
    # field is unknown, so the file is skipped (as with no overrides).
    assert parse_name("0000042.png", _cfg(), sample_id="P1") is None
    assert parse_name("0000042.png", _cfg(), camera="basic") is None


def test_sampleless_regex_with_sample_override(tmp_path):
    # A custom regex may omit the `sample` group entirely and rely on the
    # override to supply it, while camera/frame still parse from the filename.
    import yaml

    regex = r"(?P<cam>[bz])_(?P<frame>\d+)$"
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump({"filename": {"sample_regex": regex}}))
    p = parse_name("b_0000007.bmp", load_config(cfg_path), sample_id="P1")
    assert p is not None
    assert p.sample_id == "P1"
    assert p.camera == "basic"
    assert p.frame_index == 7


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
