from __future__ import annotations

import pandas as pd

from grain_morph.writers import read_table, write_partitioned, write_table


def test_roundtrip_all_formats(tmp_path):
    df = pd.DataFrame(
        {"grain_uid": ["f:1", "f:2"], "area_um2": [1.5, 2.5], "qc_pass": [True, False]}
    )
    for fmt in ("parquet", "csv", "feather"):
        p = write_table(df, tmp_path / f"t_{fmt}", fmt)
        back = read_table(p, fmt)
        assert list(back["grain_uid"]) == ["f:1", "f:2"]
        assert back["area_um2"].tolist() == [1.5, 2.5]


def test_write_partitioned_parquet_dataset_roundtrip(tmp_path):
    """Hive-partitioned parquet strips partition_cols from each part file's
    own schema (they live only in the directory path) but reconstructs them
    when the dataset directory is read back as a whole — the integrity
    Task 12 relies on when it reads a full partitioned parquet dataset.
    """
    df = pd.DataFrame(
        {
            "sample_id": ["S1", "S1", "S2"],
            "camera": ["basic", "basic", "zoom"],
            "grain_uid": ["S1:1", "S1:2", "S2:1"],
            "area_um2": [1.0, 2.0, 3.0],
        }
    )
    root = tmp_path / "parquet_partitioned"
    write_partitioned(df, root, "parquet", ["sample_id", "camera"], True)

    # A single part file does NOT carry the partition columns...
    part_files = sorted(root.rglob("*.parquet"))
    assert part_files, "expected at least one written parquet part file"
    leaf = read_table(part_files[0], "parquet")
    assert "sample_id" not in leaf.columns
    assert "camera" not in leaf.columns

    # ...but reading the dataset directory as a whole reconstructs them
    # from the hive-style path, with correct values per row.
    back = pd.read_parquet(root).sort_values("grain_uid").reset_index(drop=True)
    assert back["sample_id"].astype(str).tolist() == ["S1", "S1", "S2"]
    assert back["camera"].astype(str).tolist() == ["basic", "basic", "zoom"]
    assert back["area_um2"].tolist() == [1.0, 2.0, 3.0]


def test_csv_roundtrip_bool_and_ambiguous_string_column(tmp_path):
    """Documents CSV's lossy round-trip rather than hiding it: a genuine
    ``qc_pass``-style bool column survives as bool, but so does a genuine
    two-valued *string* column whose only values happen to be the literal
    words "True"/"False" — CSV text can't tell the two apart, and
    `_coerce_bool_columns` deliberately resolves that ambiguity in favor of
    bool. This is a known, accepted CSV limitation (use parquet/feather for
    lossless dtype round-trips), not a bug to fix here.
    """
    df = pd.DataFrame(
        {
            "qc_pass": [True, False, True],
            "ambiguous_label": ["True", "False", "True"],
        }
    )
    p = write_table(df, tmp_path / "bool_ambiguity", "csv")
    back = read_table(p, "csv")

    assert back["qc_pass"].dtype == bool
    assert back["qc_pass"].tolist() == [True, False, True]

    # Documented limitation, not a bug: a genuine string column holding
    # only "True"/"False" is indistinguishable from a bool column once
    # round-tripped through CSV text, so it comes back as bool too.
    assert back["ambiguous_label"].dtype == bool
    assert back["ambiguous_label"].tolist() == [True, False, True]
