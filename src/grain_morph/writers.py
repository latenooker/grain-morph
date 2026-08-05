"""Tabular output writers/readers for grain-morph pipeline results.

Supports the three formats exposed by ``Config.output.format``
(``"parquet"``, ``"csv"``, ``"feather"``), plus an optional
``pyarrow.dataset``-partitioned layout for parquet. CSV round-trips lose
dtype information (everything comes back from disk as text), so
:func:`read_table` restores the written frame's dtypes for that format —
without this, a CSV round-trip of a boolean column like ``qc_pass`` would
come back as the strings ``"True"``/``"False"`` instead of ``bool``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

_EXTENSIONS = {"parquet": ".parquet", "csv": ".csv", "feather": ".feather"}


def _with_extension(path: Path, fmt: str) -> Path:
    """Append the canonical extension for ``fmt`` to ``path``, if missing.

    Args:
        path: Target path, with or without the format's extension.
        fmt: One of ``"parquet"``, ``"csv"``, ``"feather"``.

    Returns:
        ``path`` unchanged if it already ends with the right extension,
        otherwise ``path`` with that extension appended.

    Raises:
        ValueError: If ``fmt`` is not a supported format.
    """
    try:
        ext = _EXTENSIONS[fmt]
    except KeyError as exc:
        raise ValueError(f"Unsupported table format: {fmt!r}") from exc
    return path if path.suffix == ext else path.with_name(path.name + ext)


def write_table(df: pd.DataFrame, path: Path, fmt: str) -> Path:
    """Write a DataFrame to disk in one of the supported table formats.

    Args:
        df: Table to write.
        path: Destination path. The correct extension for ``fmt`` is
            appended if ``path`` doesn't already end with it.
        fmt: One of ``"parquet"``, ``"csv"``, ``"feather"``.

    Returns:
        The actual path written (``path`` plus extension).

    Raises:
        ValueError: If ``fmt`` is not a supported format.
    """
    out_path = _with_extension(Path(path), fmt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        df.to_parquet(out_path, index=False)
    elif fmt == "csv":
        df.to_csv(out_path, index=False)
    elif fmt == "feather":
        df.to_feather(out_path)
    else:
        raise ValueError(f"Unsupported table format: {fmt!r}")
    return out_path


def read_table(path: Path, fmt: str) -> pd.DataFrame:
    """Read a DataFrame previously written by :func:`write_table`.

    For ``fmt="csv"``, columns are coerced back to ``bool``/numeric dtypes
    where CSV's text round-trip would otherwise leave them as strings
    (``pandas`` already infers numeric columns; only literal ``"True"``/
    ``"False"`` columns need an explicit boolean cast).

    Args:
        path: Path to the table file, as returned by :func:`write_table`.
        fmt: One of ``"parquet"``, ``"csv"``, ``"feather"``.

    Returns:
        The loaded DataFrame.

    Raises:
        ValueError: If ``fmt`` is not a supported format.
    """
    path = Path(path)
    if fmt == "parquet":
        return pd.read_parquet(path)
    if fmt == "csv":
        df = pd.read_csv(path)
        return _coerce_bool_columns(df)
    if fmt == "feather":
        return pd.read_feather(path)
    raise ValueError(f"Unsupported table format: {fmt!r}")


def _coerce_bool_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Cast object columns of literal ``"True"``/``"False"`` strings to bool.

    CSV has no boolean dtype: pandas reads a written ``bool`` column back
    as the strings ``"True"``/``"False"`` (or, if any values are missing,
    an object column mixing those strings with ``NaN``). This restores the
    original ``bool`` dtype for such columns; every other column is left
    untouched.

    Args:
        df: DataFrame just read from CSV.

    Returns:
        ``df`` with any all-``"True"``/``"False"`` object columns cast to
        ``bool``.
    """
    for col in df.columns:
        if df[col].dtype == object and set(df[col].dropna().unique()) <= {"True", "False"}:
            df[col] = df[col].map({"True": True, "False": False}).astype(bool)
    return df


def write_partitioned(
    df: pd.DataFrame,
    root: Path,
    fmt: str,
    partition_cols: list[str],
    partition: bool,
) -> None:
    """Write a DataFrame, optionally split into per-group partitions.

    For ``fmt="parquet"`` with ``partition=True``, writes a
    ``pyarrow.dataset`` hive-style partitioned dataset under ``root``,
    partitioned by ``partition_cols``. Otherwise (``csv``/``feather``, or
    ``partition=False``), writes plain files under ``root``: one per
    distinct combination of ``partition_cols`` values (named by joining
    those values with ``__``, e.g. ``S1__basic.csv``) when
    ``partition=True``, or a single file (``data.<ext>``) when
    ``partition=False``.

    Args:
        df: Table to write.
        root: Destination directory (created if missing).
        fmt: One of ``"parquet"``, ``"csv"``, ``"feather"``.
        partition_cols: Column names to partition by.
        partition: Whether to split output by ``partition_cols`` at all.

    Raises:
        ValueError: If ``fmt`` is not a supported format.
    """
    if fmt not in _EXTENSIONS:
        raise ValueError(f"Unsupported table format: {fmt!r}")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    if not partition:
        write_table(df, root / "data", fmt)
        return

    if fmt == "parquet":
        table = pa.Table.from_pandas(df, preserve_index=False)
        ds.write_dataset(
            table,
            base_dir=str(root),
            format="parquet",
            partitioning=partition_cols,
            partitioning_flavor="hive",
            existing_data_behavior="overwrite_or_ignore",
        )
        return

    for key, group in df.groupby(partition_cols, sort=True):
        key_values = key if isinstance(key, tuple) else (key,)
        name = "__".join(str(v) for v in key_values)
        write_table(group, root / name, fmt)


__all__ = ["read_table", "write_partitioned", "write_table"]
