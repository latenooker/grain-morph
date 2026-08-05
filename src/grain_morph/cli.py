"""Typer CLI: `detect`, `aggregate`, `report`, and `make-fixtures` commands.

Thin argument-wiring layer only — every command loads a resolved `Config`
via `grain_morph.config.load_config` and delegates straight to the
corresponding library entry point (`pipeline.run_detect`,
`aggregate.aggregate_run`, `report.make_reports`); no business logic lives
here. `make-fixtures` is the one exception with real (if small) logic of its
own — an anti-aliased image downsampler with no library home yet, used by
Task 16 to build committed integration-test fixtures from full-resolution
dev data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import imageio.v3 as iio
import numpy as np
import pandas as pd
import typer
from skimage.transform import downscale_local_mean

from grain_morph.aggregate import aggregate_run
from grain_morph.config import Config, load_config
from grain_morph.pipeline import run_detect
from grain_morph.report import make_reports
from grain_morph.writers import read_table, write_table

app = typer.Typer(
    help="Classical-CV morphometry + QC pipeline for backlit silhouette grain images."
)

# `make-fixtures`'s default downsample factor -- a CLI option default, not a
# pipeline tunable, so it lives here rather than in `configs/default.yaml`.
_DEFAULT_DOWNSAMPLE_FACTOR = 4


def _read_grains(path: Path, cfg: Config) -> pd.DataFrame:
    """Load a per-grain table from either a grains root or a single file.

    Args:
        path: A grains root directory (e.g. `out_dir / "grains"` from a
            `detect` run), read whole via `pandas.read_parquet` -- which
            transparently unions every part file, hive-partitioned or not;
            or a single table file, read via `grain_morph.writers.
            read_table` using `cfg.output.format`.
        cfg: Resolved pipeline configuration; `cfg.output.format` selects
            the format `read_table` assumes when `path` is a single file.

    Returns:
        The loaded per-grain table.
    """
    if path.is_dir():
        return pd.read_parquet(path)
    return read_table(path, cfg.output.format)


_ConfigOpt = Annotated[
    Path | None,
    typer.Option("--config", help="User config YAML, deep-merged over the packaged defaults."),
]
_GrainsPathHelp = "Grains root directory (from `detect`) or a single grains table file."


@app.command()
def detect(
    frames_dir: Annotated[
        Path, typer.Argument(help="Directory to search recursively for frames.")
    ],
    out_dir: Annotated[
        Path, typer.Argument(help="Destination directory for run artifacts.")
    ],
    config: _ConfigOpt = None,
    jobs: Annotated[
        int | None, typer.Option("--jobs", help="Override cfg.runtime.n_jobs for this run.")
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Reprocess every frame, clearing prior run artifacts first."),
    ] = False,
) -> None:
    """Run Stage 1 detection over every frame under FRAMES_DIR.

    Args:
        frames_dir: Directory to search recursively for frames.
        out_dir: Destination directory for all run artifacts (created if
            missing).
        config: Optional user config YAML overriding the packaged defaults.
        jobs: Overrides `cfg.runtime.n_jobs` for this call (`None` keeps the
            config value).
        force: Reprocess every frame regardless of the existing manifest.
    """
    cfg = load_config(config)
    run_detect(frames_dir, out_dir, cfg, n_jobs=jobs, force=force)


@app.command()
def aggregate(
    grains_path: Annotated[Path, typer.Argument(help=_GrainsPathHelp)],
    out_dir: Annotated[
        Path, typer.Argument(help="Destination directory for aggregate tables.")
    ],
    config: _ConfigOpt = None,
) -> None:
    """Aggregate a per-grain table into per-(sample, camera) summaries.

    Args:
        grains_path: Grains root directory or single grains table file (see
            `_read_grains`).
        out_dir: Destination directory `summary`/`rejection_by_ecd`/
            `accepted` are written into, as `cfg.output.format`.
        config: Optional user config YAML overriding the packaged defaults.
    """
    cfg = load_config(config)
    grains = _read_grains(grains_path, cfg)
    tables = aggregate_run(grains, cfg)
    for name, table in tables.items():
        write_table(table, out_dir / name, cfg.output.format)


@app.command()
def report(
    grains_path: Annotated[Path, typer.Argument(help=_GrainsPathHelp)],
    frames_dir: Annotated[
        Path, typer.Argument(help="Directory to search for the source frames.")
    ],
    out_dir: Annotated[
        Path, typer.Argument(help="Destination directory for report figures.")
    ],
    config: _ConfigOpt = None,
) -> None:
    """Render QC review artifacts for a per-grain table.

    Args:
        grains_path: Grains root directory or single grains table file (see
            `_read_grains`).
        frames_dir: Directory to search for the frames `grains["frame_path"]`
            references.
        out_dir: Destination directory for every figure (created if
            missing).
        config: Optional user config YAML overriding the packaged defaults.
    """
    cfg = load_config(config)
    grains = _read_grains(grains_path, cfg)
    make_reports(grains, frames_dir, out_dir, cfg)


def _downsample_image(src: Path, dest: Path, factor: int) -> None:
    """Anti-aliased-downsample one image by an integer factor.

    Uses `skimage.transform.downscale_local_mean` (block-averaging, hence
    anti-aliased by construction) rather than `resize`/`rescale`, since the
    output size only ever needs to be an exact `1/factor` of the input --
    no arbitrary target shape. A trailing channel axis (e.g. an RGB-saved
    BMP), if present, is left undownsampled.

    Args:
        src: Source image path.
        dest: Destination path (parent created if missing).
        factor: Integer factor each of the image's height/width is divided
            by.
    """
    image = np.asarray(iio.imread(src))
    factors = (factor, factor, 1) if image.ndim == 3 else (factor, factor)
    downsampled = downscale_local_mean(image, factors)
    downsampled = np.clip(np.round(downsampled), 0, 255).astype(np.uint8)
    dest.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(dest, downsampled)


@app.command(name="make-fixtures")
def make_fixtures(
    src_dir: Annotated[
        Path, typer.Argument(help="Directory of source images to downsample.")
    ],
    dest_dir: Annotated[
        Path,
        typer.Argument(help="Destination directory for downsampled images (created if missing)."),
    ],
    factor: Annotated[
        int, typer.Option("--factor", help="Integer factor to downsample each image by.")
    ] = _DEFAULT_DOWNSAMPLE_FACTOR,
) -> None:
    """Anti-aliased-downsample every image in SRC_DIR into DEST_DIR.

    Preserves filenames (so downstream filename parsing/blank-pairing still
    works on the downsampled set) -- used by Task 16 to build committed
    integration-test fixtures from full-resolution dev data.

    Args:
        src_dir: Directory of source images (e.g. BMP frames).
        dest_dir: Destination directory; created if missing.
        factor: Integer factor each image's height/width is divided by.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(src_dir.iterdir()):
        if path.is_file():
            _downsample_image(path, dest_dir / path.name, factor)
