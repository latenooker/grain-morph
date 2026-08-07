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


def test_appledouble_sidecar_ignored(frame_dir):
    # Regression: macOS writes `._<name>` AppleDouble sidecar files (not
    # real image data) when copying onto non-HFS+ volumes -- e.g. the
    # exFAT external drives this pipeline reads CAMSIZER frames from.
    # `sample_regex`'s leading `.+?` group would happily swallow a `._`
    # prefix as part of the sample id, matching the sidecar as if it were
    # a genuine frame and letting it reach (and fail) image decoding
    # downstream. It must never be discovered as a frame.
    (frame_dir / "._S1_b_0000001.bmp").write_bytes(b"not a real image")
    specs = discover_frames(frame_dir, load_config(None))
    assert len(specs) == 2  # unchanged from test_discovers_data_frames_only
    assert all(not s.path.name.startswith(".") for s in specs)


def test_blank_pairing_is_run_scoped(tmp_path):
    """Two runs of one sample+camera each pair to their OWN run's blank.

    Regression: when `sample_regex` separates a `run` group, blanks must key on
    (sample, camera, run). Otherwise the two runs' blanks collide and a frame is
    flat-fielded against the wrong run's background.
    """
    import imageio.v3 as iio
    import numpy as np
    import yaml

    d = tmp_path / "frames"
    d.mkdir()
    bg = np.full((16, 16), 200, np.uint8)
    for run, frame in [("001", "0000001"), ("002", "0000002")]:
        iio.imwrite(d / f"SAMP_{run}_b_{frame}.bmp", bg)
        iio.imwrite(d / f"SAMP_{run}_b_back.bmp", bg)
    regex = r"(?P<sample>.+)_(?P<run>\d+)_(?P<cam>[bz])_(?:(?P<frame>\d+)|back)$"
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump({"filename": {"sample_regex": regex}}))
    specs = discover_frames(d, load_config(cfg_path))
    assert len(specs) == 2
    for s in specs:
        assert s.parsed.run is not None
        assert s.blank_path is not None
        assert f"_{s.parsed.run}_b_back" in s.blank_path.name  # same run's blank
