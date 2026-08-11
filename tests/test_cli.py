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


def _detect_with_no_objects(tmp_path):
    """Run `detect` on a frame set with zero detected objects.

    Both the blank and the sole data frame are plain, uniform-background
    renders (no ``objects``) -- a realistic "nothing here" capture (e.g. a
    blank-only sample, or a calibration batch with no particles) -- so
    `run_detect` writes a manifest/summary but never creates `out_dir /
    "grains"` (see `pipeline.run_detect`'s docstring).

    Args:
        tmp_path: Pytest `tmp_path` fixture of the calling test.

    Returns:
        `(out_dir, config_path)`.
    """
    run = tmp_path / "run"
    run.mkdir()
    iio.imwrite(run / "S1_b_back.bmp", make_frame(size=(128, 128), objects=[]).image)
    iio.imwrite(run / "S1_b_0000001.bmp", make_frame(size=(128, 128), objects=[]).image)
    cfg = tmp_path / "c.yaml"
    calibration = {"calibration": {"um_per_px": {"basic": 5.0, "zoom": 1.0}}}
    cfg.write_text(yaml.safe_dump(calibration))
    out = tmp_path / "out"
    args = ["detect", str(run), str(out), "--config", str(cfg), "--jobs", "1"]
    res = runner.invoke(app, args)
    assert res.exit_code == 0, res.output
    assert not (out / "grains").exists()
    return out, cfg


def test_aggregate_cli_on_zero_grains_exits_cleanly(tmp_path):
    out, cfg = _detect_with_no_objects(tmp_path)
    agg_out = tmp_path / "agg"
    args = ["aggregate", str(out / "grains"), str(agg_out), "--config", str(cfg)]
    res = runner.invoke(app, args)
    assert res.exit_code == 0, res.output
    assert "No grains found" in res.output
    assert not agg_out.exists()


def test_report_cli_on_zero_grains_exits_cleanly(tmp_path):
    out, cfg = _detect_with_no_objects(tmp_path)
    rep_out = tmp_path / "rep"
    frames_dir = tmp_path / "run"
    args = ["report", str(out / "grains"), str(frames_dir), str(rep_out), "--config", str(cfg)]
    res = runner.invoke(app, args)
    assert res.exit_code == 0, res.output
    assert "No grains found" in res.output
    assert not rep_out.exists()


def test_aggregate_cli_on_csv_grains_directory(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    iio.imwrite(run / "S1_b_back.bmp", make_frame(size=(128, 128), objects=[]).image)
    objects = [
        {"kind": "ellipse", "cx": 64, "cy": 64, "a": 18, "b": 18, "angle": 0.0, "blur_sigma": 0.0}
    ]
    iio.imwrite(run / "S1_b_0000001.bmp", make_frame(size=(128, 128), objects=objects).image)
    cfg = tmp_path / "c.yaml"
    cfg_dict = {
        "calibration": {"um_per_px": {"basic": 5.0, "zoom": 1.0}},
        "output": {"format": "csv"},
    }
    cfg.write_text(yaml.safe_dump(cfg_dict))
    out = tmp_path / "out"
    detect_args = ["detect", str(run), str(out), "--config", str(cfg), "--jobs", "1"]
    res = runner.invoke(app, detect_args)
    assert res.exit_code == 0, res.output
    assert (out / "grains").is_dir()

    agg_out = tmp_path / "agg"
    agg_args = ["aggregate", str(out / "grains"), str(agg_out), "--config", str(cfg)]
    res2 = runner.invoke(app, agg_args)
    assert res2.exit_code == 0, res2.output
    assert (agg_out / "summary.csv").exists()
    assert (agg_out / "accepted.csv").exists()


def test_aggregate_cli_writes_per_frame_table(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    iio.imwrite(run / "S1_b_back.bmp", make_frame(size=(128, 128), objects=[]).image)
    obj = [{"kind": "ellipse", "cx": 64, "cy": 64, "a": 18, "b": 18,
            "angle": 0.0, "blur_sigma": 0.0}]
    iio.imwrite(run / "S1_b_0000001.bmp", make_frame(size=(128, 128), objects=obj).image)
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump({
        "calibration": {"um_per_px": {"basic": 5.0, "zoom": 1.0}},
        "output": {"format": "csv"},
    }))
    out = tmp_path / "out"
    res = runner.invoke(app, ["detect", str(run), str(out), "--config", str(cfg), "--jobs", "1"])
    assert res.exit_code == 0, res.output

    agg_out = tmp_path / "agg"
    res = runner.invoke(app, ["aggregate", str(out / "grains"), str(agg_out), "--config", str(cfg)])
    assert res.exit_code == 0, res.output
    assert (agg_out / "per_frame.csv").exists()


def test_write_overviews_renders_all_detected_frames(tmp_path):
    from grain_morph.cli import _write_overviews
    from grain_morph.config import load_config

    run = tmp_path / "run"
    run.mkdir()
    iio.imwrite(run / "S1_b_back.bmp", make_frame(size=(128, 128), objects=[]).image)
    obj = [{"kind": "ellipse", "cx": 64, "cy": 64, "a": 18, "b": 18,
            "angle": 0.0, "blur_sigma": 0.0}]
    iio.imwrite(run / "S1_b_0000001.bmp", make_frame(size=(128, 128), objects=obj).image)
    iio.imwrite(run / "S1_b_0000002.bmp", make_frame(size=(128, 128), objects=obj).image)
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump({"calibration": {"um_per_px": {"basic": 5.0, "zoom": 1.0}}}))
    out = tmp_path / "out"
    args = ["detect", str(run), str(out), "--config", str(cfg_path), "--jobs", "1"]
    res = runner.invoke(app, args)
    assert res.exit_code == 0, res.output

    _write_overviews(out, run, load_config(cfg_path), factor=4)
    pngs = sorted((out / "overviews").glob("*.png"))
    assert len(pngs) == 2


def test_write_overviews_zero_grains_no_crash(tmp_path):
    from grain_morph.cli import _write_overviews
    from grain_morph.config import load_config

    out, cfg_path = _detect_with_no_objects(tmp_path)
    _write_overviews(out, tmp_path / "run", load_config(cfg_path), factor=4)
    assert not (out / "overviews").exists()


def _detect_one_object_args(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    iio.imwrite(run / "S1_b_back.bmp", make_frame(size=(128, 128), objects=[]).image)
    obj = [{"kind": "ellipse", "cx": 64, "cy": 64, "a": 18, "b": 18,
            "angle": 0.0, "blur_sigma": 0.0}]
    iio.imwrite(run / "S1_b_0000001.bmp", make_frame(size=(128, 128), objects=obj).image)
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump({"calibration": {"um_per_px": {"basic": 5.0, "zoom": 1.0}}}))
    out = tmp_path / "out"
    return run, out, cfg


def test_detect_overview_writes_pngs(tmp_path):
    run, out, cfg = _detect_one_object_args(tmp_path)
    res = runner.invoke(
        app, ["detect", str(run), str(out), "--config", str(cfg), "--jobs", "1", "--overview"]
    )
    assert res.exit_code == 0, res.output
    assert list((out / "overviews").glob("*.png"))


def test_detect_without_overview_writes_no_overviews_dir(tmp_path):
    run, out, cfg = _detect_one_object_args(tmp_path)
    res = runner.invoke(
        app, ["detect", str(run), str(out), "--config", str(cfg), "--jobs", "1"]
    )
    assert res.exit_code == 0, res.output
    assert not (out / "overviews").exists()


def _structureless_run(tmp_path):
    """One dir of index-named frames + a `back` blank, no identity tokens."""
    run = tmp_path / "run"
    run.mkdir()
    iio.imwrite(run / "back.bmp", make_frame(size=(128, 128), objects=[]).image)
    obj = [{"kind": "ellipse", "cx": 64, "cy": 64, "a": 18, "b": 18,
            "angle": 0.0, "blur_sigma": 0.0}]
    iio.imwrite(run / "0000001.bmp", make_frame(size=(128, 128), objects=obj).image)
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump({"calibration": {"um_per_px": {"basic": 5.0, "zoom": 1.0}}}))
    return run, tmp_path / "out", cfg


def test_detect_cli_sample_and_camera_overrides(tmp_path):
    run, out, cfg = _structureless_run(tmp_path)
    args = ["detect", str(run), str(out), "--config", str(cfg), "--jobs", "1",
            "--sample-id", "P1", "--camera", "basic"]
    res = runner.invoke(app, args)
    assert res.exit_code == 0, res.output
    import pandas as pd
    df = pd.read_parquet(out / "grains")
    assert set(df["sample_id"].astype(str)) == {"P1"}
    assert set(df["camera"].astype(str)) == {"basic"}


def test_detect_cli_invalid_camera_rejected(tmp_path):
    run, out, cfg = _structureless_run(tmp_path)
    args = ["detect", str(run), str(out), "--config", str(cfg), "--jobs", "1",
            "--sample-id", "P1", "--camera", "telescope"]
    res = runner.invoke(app, args)
    assert res.exit_code != 0
    assert not out.exists()
