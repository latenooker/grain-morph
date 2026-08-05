from __future__ import annotations

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
