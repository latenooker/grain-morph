from __future__ import annotations

import pytest

from grain_morph.io import Manifest, frame_fingerprint


def test_fingerprint_changes_with_content(tmp_path):
    p = tmp_path / "f.bmp"
    p.write_bytes(b"aaaa")
    fp1 = frame_fingerprint(p)
    p.write_bytes(b"aaaaaaaa")
    assert frame_fingerprint(p) != fp1


def test_is_done_requires_matching_fingerprint():
    m = Manifest()
    m.add("S1_b_0000001", "S1_b_0000001.bmp", "ok", 3, 0.1, "blank", fingerprint="10:5")
    assert m.is_done("S1_b_0000001", "10:5") is True
    assert m.is_done("S1_b_0000001", "99:5") is False   # changed file -> reprocess
    assert m.is_done("other", "10:5") is False


@pytest.mark.parametrize("fmt", ["parquet", "csv", "feather"])
def test_save_load_roundtrip(tmp_path, fmt):
    m = Manifest()
    m.add("S1_b_0000001", "S1_b_0000001.bmp", "ok", 3, 0.125, "blank", fingerprint="10:5")
    m.add("S1_1_0000002", "S1_1_0000002.bmp", "error", 0, 1.5, "morphological", fingerprint="20:9")

    saved_path = m.save(tmp_path / "manifest", fmt)

    loaded = Manifest.load(saved_path, fmt)
    assert loaded.is_done("S1_b_0000001", "10:5") is True
    assert loaded.is_done("S1_1_0000002", "20:9") is True
    assert loaded.is_done("S1_1_0000002", "stale") is False

    left = m.to_frame().sort_values("frame_id").reset_index(drop=True)
    right = loaded.to_frame().sort_values("frame_id").reset_index(drop=True)
    assert left.equals(right)
