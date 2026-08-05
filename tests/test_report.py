from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from grain_morph.config import load_config
from grain_morph.report import make_reports


def test_report_files_created(tmp_path):
    rng = np.random.default_rng(0)
    n = 50
    grains = pd.DataFrame({
        "sample_id": ["S1"] * n, "camera": ["basic"] * n,
        "grain_uid": [f"f:{i}" for i in range(n)],
        "ecd_um": rng.uniform(50, 500, n),
        "edge_gradient_norm": rng.uniform(0.02, 0.2, n),
        "contrast": rng.uniform(0.8, 0.95, n),
        "flag_defocus": rng.random(n) < 0.2,
        "qc_pass": rng.random(n) < 0.8,
    })
    paths = make_reports(
        grains, frames_root=tmp_path, out_dir=tmp_path / "rep", cfg=load_config(None)
    )
    assert any(p.name == "focus_scatter.png" for p in paths)
    assert any(p.name == "rejection_vs_ecd.png" for p in paths)
    assert all(p.exists() for p in paths)


def _write_frame(path: Path, size: int = 200, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    frame = np.full((size, size), 200, np.uint8)
    for _ in range(6):
        cy, cx = rng.integers(20, size - 20, 2)
        frame[cy - 8 : cy + 8, cx - 8 : cx + 8] = 30
    iio.imwrite(path, frame)


def _grains_with_frames(frame_paths: list[Path], n: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    frame_choice = [str(frame_paths[i % len(frame_paths)]) for i in range(n)]
    flag_defocus = np.array([i < 5 for i in range(n)])  # first 5 rejected for defocus
    flag_border = np.array([5 <= i < 8 for i in range(n)])  # next 3 rejected for border
    qc_pass = ~(flag_defocus | flag_border)
    return pd.DataFrame({
        "sample_id": ["S1"] * n,
        "camera": ["basic"] * n,
        "grain_uid": [f"f:{i}" for i in range(n)],
        "frame_path": frame_choice,
        "centroid_x": rng.uniform(30, 170, n),
        "centroid_y": rng.uniform(30, 170, n),
        "um_per_px": [20.0] * n,
        "ecd_um": rng.uniform(50, 500, n),
        "edge_gradient_norm": rng.uniform(0.02, 0.2, n),
        "contrast": rng.uniform(0.8, 0.95, n),
        "flag_defocus": flag_defocus,
        "flag_border": flag_border,
        "qc_pass": qc_pass,
    })


def test_report_with_real_frames_renders_contact_sheets_and_field(tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    frame_paths = [frames_dir / "S1_b_0000001.bmp", frames_dir / "S1_b_0000002.bmp"]
    for i, p in enumerate(frame_paths):
        _write_frame(p, seed=i)

    grains = _grains_with_frames(frame_paths)
    paths = make_reports(
        grains, frames_root=frames_dir, out_dir=tmp_path / "rep", cfg=load_config(None)
    )
    names = {p.name for p in paths}

    assert "focus_scatter.png" in names
    assert "rejection_vs_ecd.png" in names
    assert "illumination_field.png" in names
    assert "contact_sheet_accepted.png" in names
    assert "contact_sheet_rejected_flag_defocus_01.png" in names
    assert "contact_sheet_rejected_flag_border_01.png" in names
    assert all(p.exists() for p in paths)


def test_report_resolves_frames_by_basename_under_frames_root(tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    real_path = frames_dir / "S1_b_0000001.bmp"
    _write_frame(real_path)

    # Record a stale/relocated absolute path that doesn't exist on this
    # machine -- only the basename matches what's actually in frames_dir.
    stale_path = Path("/nonexistent/original/location/S1_b_0000001.bmp")
    grains = _grains_with_frames([stale_path], n=10)

    paths = make_reports(
        grains, frames_root=frames_dir, out_dir=tmp_path / "rep", cfg=load_config(None)
    )
    names = {p.name for p in paths}
    assert "illumination_field.png" in names
    assert all(p.exists() for p in paths)


def _base_grain_columns(frame_path: Path, n: int) -> dict:
    """Columns every `make_reports` call needs, regardless of test focus.

    `flag_defocus` is always required (`_focus_scatter` is one of the two
    always-produced figures) even in tests that don't care about defocus
    rejection specifically -- defaulted to all-`False` here so callers only
    override it when the test is actually about that flag.
    """
    rng = np.random.default_rng(2)
    return {
        "sample_id": ["S1"] * n,
        "camera": ["basic"] * n,
        "grain_uid": [f"f:{i}" for i in range(n)],
        "frame_path": [str(frame_path)] * n,
        "centroid_x": rng.uniform(30, 170, n),
        "centroid_y": rng.uniform(30, 170, n),
        "um_per_px": [20.0] * n,
        "edge_gradient_norm": rng.uniform(0.02, 0.2, n),
        "contrast": rng.uniform(0.8, 0.95, n),
        "flag_defocus": [False] * n,
    }


def test_rejected_contact_sheet_excludes_grains_that_are_actually_accepted(tmp_path):
    """A flag_* column that's True on an otherwise-accepted grain must not
    produce a "Rejected" sheet containing it -- regression test for the
    contact-sheet mislabeling bug caught in code review."""
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    frame_path = frames_dir / "S1_b_0000001.bmp"
    _write_frame(frame_path)

    n = 10
    grains = pd.DataFrame({
        **_base_grain_columns(frame_path, n),
        "ecd_um": np.full(n, 150.0),
        # Every grain has this flag set, but qc_pass is True for all of
        # them (e.g. a non-disqualifying flag under the caller's config) --
        # none of these grains were actually rejected.
        "flag_possible_agglomerate": [True] * n,
        "qc_pass": [True] * n,
    })

    paths = make_reports(
        grains, frames_root=frames_dir, out_dir=tmp_path / "rep", cfg=load_config(None)
    )
    names = {p.name for p in paths}
    assert "contact_sheet_rejected_flag_possible_agglomerate_01.png" not in names
    assert "contact_sheet_accepted.png" in names
    assert all(p.exists() for p in paths)


def test_rejected_contact_sheet_paginates_past_cap(tmp_path):
    """More than one contact-sheet page's worth of rejected grains for the
    same flag must produce page 01 *and* 02."""
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    frame_path = frames_dir / "S1_b_0000001.bmp"
    _write_frame(frame_path)

    n = 30  # > _CONTACT_SHEET_MAX_N (24)
    grains = pd.DataFrame({
        **_base_grain_columns(frame_path, n),
        "ecd_um": np.linspace(50, 500, n),
        "flag_defocus": [True] * n,
        "qc_pass": [False] * n,
    })

    paths = make_reports(
        grains, frames_root=frames_dir, out_dir=tmp_path / "rep", cfg=load_config(None)
    )
    names = {p.name for p in paths}
    assert "contact_sheet_rejected_flag_defocus_01.png" in names
    assert "contact_sheet_rejected_flag_defocus_02.png" in names
    assert all(p.exists() for p in paths)


def test_accepted_contact_sheet_stratifies_past_cap(tmp_path):
    """More accepted grains than the per-sheet cap must still produce a
    single stratified-by-ECD sheet (exercises `_stratified_sample`'s
    `len(accepted) > cap` sampling branch)."""
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    frame_path = frames_dir / "S1_b_0000001.bmp"
    _write_frame(frame_path)

    n = 40  # > _CONTACT_SHEET_MAX_N (24)
    grains = pd.DataFrame({
        **_base_grain_columns(frame_path, n),
        "ecd_um": np.linspace(50, 500, n),
        "qc_pass": [True] * n,
    })

    paths = make_reports(
        grains, frames_root=frames_dir, out_dir=tmp_path / "rep", cfg=load_config(None)
    )
    names = {p.name for p in paths}
    assert "contact_sheet_accepted.png" in names
    assert all(p.exists() for p in paths)


def test_accepted_contact_sheet_handles_degenerate_ecd_without_crashing(tmp_path):
    """A constant `ecd_um` column leaves `pd.qcut` unable to form any
    quantile bin -- regression test for the `ZeroDivisionError` caught in
    code review (the fallback previously only caught `ValueError`, but a
    degenerate column makes `qcut` return zero groups without raising)."""
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    frame_path = frames_dir / "S1_b_0000001.bmp"
    _write_frame(frame_path)

    n = 30  # > _CONTACT_SHEET_MAX_N (24), so the stratified-sample path runs
    grains = pd.DataFrame({
        **_base_grain_columns(frame_path, n),
        "ecd_um": np.full(n, 150.0),  # zero variation -> qcut forms 0 bins
        "qc_pass": [True] * n,
    })

    paths = make_reports(
        grains, frames_root=frames_dir, out_dir=tmp_path / "rep", cfg=load_config(None)
    )
    names = {p.name for p in paths}
    assert "contact_sheet_accepted.png" in names
    assert all(p.exists() for p in paths)
