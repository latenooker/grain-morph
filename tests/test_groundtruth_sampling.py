from __future__ import annotations

import numpy as np
import pandas as pd

from grain_morph.config import load_config
from grain_morph.groundtruth import _sample_grains


def _grains(n=120, n_frames=12, seed=0, *, n_border=0, n_no_poly=0, n_nan_axis=0):
    """Synthetic per-grain table with the columns the sampler reads.

    The continuous QC-driving metrics span a wide range so space-filling
    coverage is checkable; optional border/no-polygon/NaN-axis rows exercise
    the reservation and exclusion paths.
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "grain_uid": [f"F{ i % n_frames :02d}:{i}" for i in range(n)],
            "frame_id": [f"F{ i % n_frames :02d}" for i in range(n)],
            "ecd_px": rng.uniform(5, 100, n),
            "edge_width_px": rng.uniform(2, 12, n),
            "contrast": rng.uniform(0.3, 0.95, n),
            "qc_aspect_ratio": rng.uniform(1.0, 6.0, n),
            "qc_solidity": rng.uniform(0.5, 1.0, n),
            "touches_border": np.zeros(n, bool),
            "has_polygon": np.ones(n, bool),
        }
    )
    for i in range(n_border):
        df.loc[i, "touches_border"] = True
    for i in range(n_no_poly):
        df.loc[n - 1 - i, "has_polygon"] = False
        df.loc[n - 1 - i, ["edge_width_px", "contrast", "qc_solidity"]] = np.nan
    for i in range(n_nan_axis):
        df.loc[n // 2 + i, "edge_width_px"] = np.nan  # partial-NaN, still has_polygon
    return df


def _cfg():
    cfg = load_config(None)
    cfg.groundtruth.reserved = {"border": 0, "no_polygon": 0}
    return cfg


def test_sample_size_and_same_seed_determinism():
    g = _grains()
    a = _sample_grains(g, _cfg(), n_grains=20, seed=1)
    b = _sample_grains(g, _cfg(), n_grains=20, seed=1)
    assert len(a) == 20
    assert list(a["grain_uid"]) == list(b["grain_uid"])  # deterministic


def test_different_seed_changes_selection():
    g = _grains()
    a = _sample_grains(g, _cfg(), n_grains=20, seed=1)
    c = _sample_grains(g, _cfg(), n_grains=20, seed=2)
    assert set(a["grain_uid"]) != set(c["grain_uid"])


def test_frame_pool_respected():
    g = _grains()
    sel = _sample_grains(g, _cfg(), n_frames=3, n_grains=15, seed=0)
    assert sel["frame_id"].nunique() <= 3


def test_lhs_spans_metric_range():
    # Space-filling: a reasonably sized sample reaches both ends of an axis.
    g = _grains(n=200)
    sel = _sample_grains(g, _cfg(), n_grains=40, seed=0)
    lo, hi = g["ecd_px"].min(), g["ecd_px"].max()
    span = hi - lo
    assert sel["ecd_px"].min() < lo + 0.2 * span
    assert sel["ecd_px"].max() > hi - 0.2 * span


def test_frac_grains():
    g = _grains(n=40)
    sel = _sample_grains(g, _cfg(), frac_grains=0.5, seed=0)
    assert len(sel) == 20


def test_n_grains_capped_at_pool():
    g = _grains(n=5, n_frames=1)
    sel = _sample_grains(g, _cfg(), n_grains=100, seed=0)
    assert len(sel) == 5


def test_reserved_border_and_no_polygon_included():
    g = _grains(n=120, n_border=4, n_no_poly=3)
    cfg = load_config(None)
    cfg.groundtruth.reserved = {"border": 2, "no_polygon": 1}
    sel = _sample_grains(g, cfg, n_grains=20, seed=0)
    assert int(sel["touches_border"].sum()) >= 2
    assert int((~sel["has_polygon"]).sum()) >= 1


def test_nan_axis_grain_excluded_from_lhs():
    # A grain with a NaN continuous axis (but has_polygon) must never be chosen
    # by the LHS when no reservation would rescue it.
    g = _grains(n=60, n_nan_axis=1)
    nan_uid = g.loc[g["edge_width_px"].isna(), "grain_uid"].iloc[0]
    for seed in range(4):
        sel = _sample_grains(g, _cfg(), n_grains=30, seed=seed)
        assert nan_uid not in set(sel["grain_uid"])
