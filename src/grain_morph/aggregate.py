"""Stage 2: per-grain table -> per (sample_id, camera) summaries.

Per the design doc (§"Stage 2 — `grain-morph aggregate`") this is where the
QC flags recorded (never enforced) by :mod:`grain_morph.qc` finally become a
pass/fail filter — and that filter is applied *here*, at aggregation time,
not baked into the per-grain table. `qc_pass` is therefore always
recomputed from `cfg.qc.disqualifying_flags` (see :func:`_recompute_qc_pass`)
rather than trusted from a stored `qc_pass` column, so a stricter or looser
gate can be explored just by re-running `aggregate_run` with a different
config, without re-running detection.

One function, :func:`aggregate_run`, produces three tables:

- `"summary"` — one row per `(sample_id, camera)`: total/accepted/rejected
  counts, a rejection count per QC flag, D10/D50/D90 of `ecd_um` and
  `feret_min_um` (the sieve-analogous axis), and mean/SD of every shape
  descriptor present in `grains` — all computed over *accepted* grains only,
  since the whole point of the QC gate is to keep defocused/border/sliver
  artifacts out of the reported particle-size distribution.
- `"rejection_by_ecd"` — rejection rate binned by ECD (see
  :data:`_ECD_BIN_EDGES_UM`), per `(sample_id, camera)`, so a size-dependent
  focus gate is visible rather than silent (design doc requirement).
- `"accepted"` — the row subset of `grains` that passed the recomputed
  `qc_pass`, for downstream consumers (e.g. `report.py`) that want the
  clean grain table directly rather than re-deriving it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from grain_morph.config import Config

# Bin edges (µm) `rejection_by_ecd` groups `ecd_um` into. Fixed for v1
# (`Config` has no `aggregate` section to source these from) rather than
# derived from the data, so bin boundaries are stable and comparable across
# runs/samples instead of shifting with whatever grains happen to be in a
# given batch. Follows the Wentworth (1922) grain-size class boundaries
# (very-fine/fine/medium/coarse/very-coarse sand, granule, and a >4mm
# catch-all), which comfortably spans CAMSIZER X2's silhouette-imaging
# range and gives every bin a physically meaningful name.
_ECD_BIN_EDGES_UM: tuple[float, ...] = (
    0.0,
    62.5,
    125.0,
    250.0,
    500.0,
    1000.0,
    2000.0,
    4000.0,
    float("inf"),
)

# Shape-descriptor columns `summary` reports mean/SD of, when present in
# `grains` — the brief's synthetic test df only has `solidity`, but the real
# per-grain table (`pipeline._POLYGON_KEYS` + `_WADELL_KEYS` +
# `_CURVATURE_KEYS`) has the full set below, so only the ones actually
# present are reported rather than requiring all of them.
_SHAPE_DESCRIPTOR_COLUMNS: tuple[str, ...] = (
    "aspect_ratio",
    "solidity",
    "convexity",
    "circularity",
    "extent",
    "eccentricity",
    "wadell_roundness",
    "wadell_sphericity",
    "curvature_entropy",
)

# Percentiles (percent-finer convention) reported for each size column.
_PERCENTILES: tuple[tuple[int, str], ...] = ((10, "D10"), (50, "D50"), (90, "D90"))

# Size columns `summary` reports D10/D50/D90 of.
_SIZE_COLUMNS: tuple[str, ...] = ("ecd_um", "feret_min_um")


def _ecd_bin_labels(edges: tuple[float, ...]) -> list[str]:
    """Human-readable `"lo-hi"` (or `"lo+"` for the open-ended top bin) labels.

    Args:
        edges: Ascending bin edges (µm), last one typically `inf`.

    Returns:
        `len(edges) - 1` labels, one per bin, in edge order.
    """
    labels = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        labels.append(f"{lo:g}+" if np.isinf(hi) else f"{lo:g}-{hi:g}")
    return labels


def _bin_ecd(ecd_um: pd.Series) -> pd.Series:
    """Bin an `ecd_um` column into `_ECD_BIN_EDGES_UM` classes.

    Args:
        ecd_um: Equivalent circular diameter, microns.

    Returns:
        Ordered `Categorical` series of bin labels (see
        :func:`_ecd_bin_labels`), half-open `[lo, hi)` per bin.
    """
    labels = _ecd_bin_labels(_ECD_BIN_EDGES_UM)
    return pd.cut(ecd_um, bins=_ECD_BIN_EDGES_UM, labels=labels, right=False, include_lowest=True)


def _recompute_qc_pass(grains: pd.DataFrame, cfg: Config) -> pd.Series:
    """Recompute `qc_pass` from `cfg.qc.disqualifying_flags`.

    Deliberately ignores any stored `qc_pass` column: aggregation-time flag
    choices (a different, e.g. looser or stricter, `cfg` than the one
    detection ran with) must be honored, not overridden by whatever gate
    was baked in at detection time.

    Args:
        grains: Per-grain table; must have a boolean column for every name
            in `cfg.qc.disqualifying_flags`.
        cfg: Resolved pipeline configuration.

    Returns:
        Boolean series, `True` iff none of `cfg.qc.disqualifying_flags` are
        set for that row.

    Raises:
        KeyError: If any name in `cfg.qc.disqualifying_flags` is not a
            column of `grains` — a config/data mismatch that should fail
            loudly rather than be silently treated as "not disqualifying".
    """
    disqualifying = list(cfg.qc.disqualifying_flags)
    if not disqualifying:
        return pd.Series(True, index=grains.index)
    flags = grains[disqualifying].astype(bool)
    return ~flags.any(axis=1)


def _flag_columns(grains: pd.DataFrame) -> list[str]:
    """Every `flag_*` column present in `grains`, in column order.

    Args:
        grains: Per-grain table.

    Returns:
        Column names starting with `"flag_"`.
    """
    return [c for c in grains.columns if c.startswith("flag_")]


def _percentile_stats(values: pd.Series, prefix: str) -> dict[str, float]:
    """D10/D50/D90 of `values` (percent-finer convention), `nan` if empty.

    Args:
        values: Size measurements (e.g. `ecd_um` of accepted grains in one
            group).
        prefix: Column-name prefix, e.g. `"ecd_um"` -> `"ecd_um_D10"` etc.

    Returns:
        Dict with keys `f"{prefix}_D10"`, `f"{prefix}_D50"`, `f"{prefix}_D90"`.
    """
    if values.empty:
        return {f"{prefix}_{name}": float("nan") for _, name in _PERCENTILES}
    percentiles = np.percentile(values, [p for p, _ in _PERCENTILES])
    return {
        f"{prefix}_{name}": float(value)
        for (_, name), value in zip(_PERCENTILES, percentiles, strict=True)
    }


def _summarize_group(group: pd.DataFrame) -> pd.Series:
    """Build one `summary` row for a single `(sample_id, camera)` group.

    Args:
        group: This group's rows of the per-grain table, with `qc_pass`
            already recomputed.

    Returns:
        Series of summary fields (counts, per-flag rejection counts,
        size percentiles, shape descriptor mean/SD) — all size/shape
        statistics computed over `qc_pass` accepted rows only.
    """
    accepted = group[group["qc_pass"]]
    n_total = len(group)
    n_accepted = len(accepted)

    result: dict[str, float | int] = {
        "n_total": n_total,
        "n_accepted": n_accepted,
        "n_rejected": n_total - n_accepted,
    }
    for flag_col in _flag_columns(group):
        result[f"n_{flag_col}"] = int(group[flag_col].sum())
    for size_col in _SIZE_COLUMNS:
        result.update(_percentile_stats(accepted[size_col], size_col))
    for shape_col in _SHAPE_DESCRIPTOR_COLUMNS:
        if shape_col not in group.columns:
            continue
        mean = float(accepted[shape_col].mean()) if n_accepted else float("nan")
        sd = float(accepted[shape_col].std()) if n_accepted else float("nan")
        result[f"{shape_col}_mean"] = mean
        result[f"{shape_col}_sd"] = sd
    return pd.Series(result)


def _build_summary(grains: pd.DataFrame) -> pd.DataFrame:
    """Per `(sample_id, camera)` summary table (see module docstring).

    Args:
        grains: Per-grain table with `qc_pass` already recomputed.

    Returns:
        One row per `(sample_id, camera)`, columns per :func:`_summarize_group`.
    """
    summary = (
        grains.groupby(["sample_id", "camera"], sort=True)
        .apply(_summarize_group, include_groups=False)
        .reset_index()
    )
    # Count columns are always well-defined integers (never `nan`) for any
    # group, even one with zero accepted grains — unlike the percentile/
    # mean columns. `_summarize_group` mixes them into one `pd.Series` per
    # group alongside those float/`nan`-valued columns, which upcasts the
    # whole per-group Series (and hence these columns, once assembled into
    # `summary`) to `float64`; cast back to `int64` here for a tidy result.
    fixed_count_cols = {"n_total", "n_accepted", "n_rejected"}
    count_cols = [c for c in summary.columns if c in fixed_count_cols or c.startswith("n_flag_")]
    summary[count_cols] = summary[count_cols].astype("int64")
    return summary


def _build_rejection_by_ecd(grains: pd.DataFrame) -> pd.DataFrame:
    """Rejection rate binned by ECD, per `(sample_id, camera)`.

    Args:
        grains: Per-grain table with `qc_pass` already recomputed.

    Returns:
        Columns `sample_id, camera, ecd_bin, n, n_rejected, rejection_rate`
        (rate in `[0, 1]`); only bins with at least one grain are present.
    """
    working = grains.copy()
    working["ecd_bin"] = _bin_ecd(working["ecd_um"])
    grouped = working.groupby(["sample_id", "camera", "ecd_bin"], observed=True, sort=True)
    out = grouped["qc_pass"].agg(n="size", n_rejected=lambda s: int((~s).sum())).reset_index()
    out["rejection_rate"] = out["n_rejected"] / out["n"]
    return out


def aggregate_run(grains: pd.DataFrame, cfg: Config) -> dict[str, pd.DataFrame]:
    """Aggregate a per-grain table into per-(sample, camera) summaries.

    Args:
        grains: Per-grain table (e.g. `pd.read_parquet(out_dir / "grains")`),
            with at least `sample_id, camera, ecd_um, feret_min_um` and a
            boolean column for every name in `cfg.qc.disqualifying_flags`.
        cfg: Resolved pipeline configuration; consumes `cfg.qc.disqualifying_flags`.

    Returns:
        Dict with keys:

        - `"summary"`: one row per `(sample_id, camera)` (see module
          docstring for the full column set).
        - `"rejection_by_ecd"`: rejection rate binned by ECD, per
          `(sample_id, camera)`.
        - `"accepted"`: the row subset of `grains` (with `qc_pass`
          recomputed) that passed QC.

    Raises:
        KeyError: If any name in `cfg.qc.disqualifying_flags` is not a
            column of `grains` (see :func:`_recompute_qc_pass`).
    """
    working = grains.copy()
    working["qc_pass"] = _recompute_qc_pass(working, cfg)
    accepted = working[working["qc_pass"]].reset_index(drop=True)
    return {
        "summary": _build_summary(working),
        "rejection_by_ecd": _build_rejection_by_ecd(working),
        "accepted": accepted,
    }
