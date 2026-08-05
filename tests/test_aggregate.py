from __future__ import annotations

import numpy as np
import pandas as pd

from grain_morph.aggregate import aggregate_run
from grain_morph.config import load_config


def _grains():
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame({
        "sample_id": ["S1"] * n,
        "camera": ["basic"] * n,
        "ecd_um": rng.uniform(50, 500, n),
        "feret_min_um": rng.uniform(40, 450, n),
        "solidity": rng.uniform(0.9, 1.0, n),
        "flag_defocus": rng.random(n) < 0.1,
        "flag_border": rng.random(n) < 0.05,
        "flag_too_small": [False] * n,
        "flag_sliver": [False] * n,
        "flag_no_polygon": [False] * n,
    })


def test_summary_counts_and_percentiles():
    out = aggregate_run(_grains(), load_config(None))
    s = out["summary"].iloc[0]
    assert s["n_total"] == 200
    assert s["n_accepted"] + s["n_rejected"] == 200
    assert s["ecd_um_D10"] < s["ecd_um_D50"] < s["ecd_um_D90"]


def test_rejection_binned_by_ecd_present():
    out = aggregate_run(_grains(), load_config(None))
    rej = out["rejection_by_ecd"]
    assert {"ecd_bin", "n", "rejection_rate"}.issubset(rej.columns)
    assert (rej["rejection_rate"].between(0, 1)).all()
