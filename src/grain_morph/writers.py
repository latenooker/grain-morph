"""Tabular output writers/readers for grain-morph pipeline results.

Supports the three formats exposed by ``Config.output.format``
(``"parquet"``, ``"csv"``, ``"feather"``), plus an optional
``pyarrow.dataset``-partitioned layout for parquet.

Parquet and feather are the **lossless** formats — dtypes (including
``bool``) round-trip exactly. CSV is a **lossy, text-based inspection
format** (matching the design doc's "portable/inspectable fallback"):
everything is written as text, and :func:`read_table` relies on pandas'
own type inference — plus a narrow, best-effort boolean coercion — to
recover something close to the original dtypes. See :func:`read_table`
for the CSV caveats this implies.
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

    ``fmt="csv"`` writes plain text: it is the portable/inspectable
    fallback, not a dtype-preserving format (see :func:`read_table`).
    ``"parquet"`` and ``"feather"`` preserve dtypes exactly.

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

    ``"parquet"`` and ``"feather"`` are lossless: the returned frame has
    exactly the dtypes it was written with. ``fmt="csv"`` is lossy —
    everything on disk is text, so dtypes are *reconstructed* rather than
    preserved: numeric columns come back via pandas' own ``read_csv``
    type inference, and object columns holding only the literal strings
    ``"True"``/``"False"`` are cast to ``bool`` by :func:`_coerce_bool_columns`
    (a best-effort heuristic, not a guarantee — see that function's
    docstring for the ambiguity this accepts). Don't rely on CSV for
    strict dtype round-tripping; use parquet or feather instead.

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

    This is a deliberately accepted ambiguity, not an oversight: CSV
    cannot distinguish "a bool column that happened to serialize as
    True/False" from "a genuine string column whose only two values are
    the literal words True/False" — both look identical on disk. This
    function (and, on pandas builds new enough to type-infer bool columns
    directly in ``read_csv`` — the object-dtype check below is a no-op —
    pandas itself) resolves that ambiguity in favor of bool, which is
    correct for this pipeline's actual columns (e.g. ``qc_pass``) but
    would silently misclassify a hypothetical genuine two-valued string
    column holding exactly ``{"True", "False"}``. Anything that must
    survive that edge case losslessly should use parquet or feather
    instead of CSV.

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

    ``partition=False`` writes a single file (``root/data.<ext>``)
    regardless of ``fmt``, and ``partition_cols`` is unused.

    When ``partition=True``, the two branches below are **intentionally
    different layouts**, not just different file-naming schemes — in
    particular they disagree on whether ``partition_cols`` end up as
    columns *inside* each file on disk:

    - ``fmt="parquet"``: writes one ``pyarrow.dataset`` hive-style
      partitioned dataset under ``root`` (via
      ``pyarrow.dataset.write_dataset``), e.g.
      ``root/sample_id=S1/camera=basic/part-0.parquet``. The
      ``partition_cols`` values are encoded **only in the directory
      path** — each individual part file's own schema does *not*
      include them. They are reconstructed as columns only when the
      dataset is read back as a whole, e.g. via
      ``pandas.read_parquet(root)`` or
      ``pyarrow.dataset.dataset(root, partitioning="hive")`` — never by
      reading a single part file directly with
      :func:`read_table`/``pd.read_parquet`` on one leaf path.
    - ``fmt="csv"`` / ``fmt="feather"``: writes one flat file per
      distinct combination of ``partition_cols`` values, named by
      joining those values with ``__`` (e.g. ``root/S1__basic.csv``).
      Unlike the parquet case, ``partition_cols`` remain **embedded as
      ordinary columns inside each file** — there is no separate
      directory-encoded layer to strip them into, so each file is
      independently readable (and re-groupable) via
      :func:`read_table` alone.

    This asymmetry is deliberate: it's the standard hive-partitioned
    dataset convention for parquet (what most parquet-consuming tools
    expect), while csv/feather have no equivalent partitioned-dataset
    reader in pandas, so keeping the partition columns inline is what
    makes each per-group file self-describing.

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
