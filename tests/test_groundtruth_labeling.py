from __future__ import annotations

import pandas as pd
import pytest

from grain_morph.config import load_config
from grain_morph.groundtruth import (
    _crop_bbox,
    _LabelSession,
    _LabelStore,
    _predicted_status,
    _require_interactive_backend,
)

# ---- label store ---------------------------------------------------------

def _rec(value, name):
    return {"label": value, "label_name": name}


def test_label_store_roundtrip_and_resume(tmp_path):
    p = tmp_path / "groundtruth.csv"
    store = _LabelStore.load(p)
    store.set("F0:1", _rec(1, "include"))
    store.set("F0:2", _rec(0, "exclude"))

    reloaded = _LabelStore.load(p)
    assert reloaded.labeled_uids() == {"F0:1", "F0:2"}
    assert reloaded.get("F0:1") == 1
    assert reloaded.get("F0:2") == 0


def test_label_store_autosaves_after_each_set(tmp_path):
    p = tmp_path / "groundtruth.csv"
    store = _LabelStore.load(p)
    store.set("F0:1", _rec(2, "special"))  # no explicit save()
    assert p.exists()
    assert str(pd.read_csv(p)["grain_uid"].iloc[0]) == "F0:1"


def test_label_store_unset(tmp_path):
    p = tmp_path / "groundtruth.csv"
    store = _LabelStore.load(p)
    store.set("F0:1", _rec(1, "include"))
    store.unset("F0:1")
    assert store.labeled_uids() == set()
    assert _LabelStore.load(p).labeled_uids() == set()


def test_label_store_missing_file_is_empty(tmp_path):
    store = _LabelStore.load(tmp_path / "nope.csv")
    assert store.labeled_uids() == set()


# ---- session state machine ----------------------------------------------

def _session(tmp_path, order, prelabel=None):
    store = _LabelStore.load(tmp_path / "g.csv")
    for uid in prelabel or []:
        store.set(uid, _rec(1, "include"))
    return _LabelSession(order, store, lambda uid, v, n: _rec(v, n))


def test_session_starts_at_first_unlabeled(tmp_path):
    s = _session(tmp_path, ["a", "b", "c"], prelabel=["a"])
    assert s.current() == "b"


def test_session_assign_sets_and_advances(tmp_path):
    s = _session(tmp_path, ["a", "b", "c"])
    s.assign(1, "include")
    assert s.store.get("a") == 1
    assert s.current() == "b"


def test_session_next_prev_clamp(tmp_path):
    s = _session(tmp_path, ["a", "b"])
    for _ in range(3):
        s.next()
    assert s.current() == "b"  # clamped at end
    for _ in range(3):
        s.prev()
    assert s.current() == "a"  # clamped at start


def test_session_toggle_mask(tmp_path):
    s = _session(tmp_path, ["a"])
    before = s.mask_visible
    s.toggle_mask()
    assert s.mask_visible is (not before)


def test_session_unset_removes_current_label(tmp_path):
    s = _session(tmp_path, ["a", "b"])
    s.assign(1, "include")   # labels "a", advances to "b"
    s.prev()                 # back to "a"
    s.unset()
    assert s.store.get("a") is None


# ---- crop geometry -------------------------------------------------------

def test_crop_bbox_pads_and_returns_row_col_bounds():
    # polygon.bounds = (minx, miny, maxx, maxy) == (col0, row0, col1, row1)
    row0, row1, col0, col1 = _crop_bbox((10.0, 20.0, 30.0, 50.0), pad=5, frame_shape=(100, 200))
    assert (row0, row1) == (15, 55)
    assert (col0, col1) == (5, 35)


def test_crop_bbox_clamps_to_frame():
    row0, row1, col0, col1 = _crop_bbox((2.0, 1.0, 30.0, 40.0), pad=10, frame_shape=(45, 25))
    assert row0 == 0 and col0 == 0        # clamped low
    assert row1 == 45 and col1 == 25      # clamped to frame height/width


# ---- predicted QC status + backend guard --------------------------------

def test_predicted_status_reject_when_disqualifying_flag_active():
    cfg = load_config(None)  # default disqualifying_flags includes flag_defocus
    row = {f: False for f in (
        "flag_defocus", "flag_border", "flag_too_small",
        "flag_sliver", "flag_possible_agglomerate", "flag_no_polygon",
    )}
    row["flag_defocus"] = True
    status, flags = _predicted_status(row, cfg)
    assert status == "reject"
    assert flags == "flag_defocus"


def test_predicted_status_accept_when_no_flags():
    cfg = load_config(None)
    row = {f: False for f in (
        "flag_defocus", "flag_border", "flag_too_small",
        "flag_sliver", "flag_possible_agglomerate", "flag_no_polygon",
    )}
    status, flags = _predicted_status(row, cfg)
    assert status == "accept"
    assert flags == ""


def test_require_interactive_backend_rejects_agg():
    with pytest.raises(RuntimeError, match="interactive"):
        _require_interactive_backend("agg")


def test_require_interactive_backend_allows_qtagg():
    _require_interactive_backend("QtAgg")  # no raise
