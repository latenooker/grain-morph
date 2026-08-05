"""Integration test: `detect` over the committed downsampled real fixtures.

Runs the full Stage 1 pipeline (`grain-morph detect`) against
`tests/fixtures/real/` -- real CAMSIZER frames, anti-aliased-downsampled 4x
by `make-fixtures` (see `tests/fixtures/real/GENERATION.md`) so they're small
enough to commit. This is not a scientific-accuracy check (the calibration
values below are positive placeholders, not the fixtures' true um_per_px) --
it exercises the pipeline end-to-end on real image content (real noise,
real illumination gradients, real grain silhouettes) rather than the
synthetic frames the rest of the suite uses, catching anything that only
shows up on real data.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml
from typer.testing import CliRunner

from grain_morph.cli import app

runner = CliRunner()
FIX = Path("tests/fixtures/real")


def test_detect_on_downsampled_real(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump({"calibration": {"um_per_px": {"basic": 20.0, "zoom": 8.0}}}))
    out = tmp_path / "out"
    res = runner.invoke(app, ["detect", str(FIX), str(out), "--config", str(cfg), "--jobs", "1"])
    assert res.exit_code == 0, res.output

    grains_dir = out / "grains"
    assert grains_dir.exists(), "expected some detected grains from real fixtures"
    # Default config partitions parquet output by (sample_id, camera) as a
    # hive-style dataset (writers.write_partitioned) -- the partition
    # columns live only in the directory path, not in each leaf part file's
    # own schema (see writers.py's module docstring), so the dataset must be
    # read as a whole directory (pd.read_parquet on the directory) to
    # reconstruct `sample_id`/`camera` as columns. Concatenating individual
    # leaf-file reads (as cli._read_grains's csv/feather branch does) would
    # silently drop those two columns for parquet.
    df = pd.read_parquet(grains_dir)
    assert not df.empty, "expected some detected grains from real fixtures"
    # both cameras represented; at least some grains detected
    assert set(df["camera"]).issubset({"basic", "zoom"})
    assert (df["um_per_px"] > 0).all()
