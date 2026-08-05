from __future__ import annotations

import pandas as pd

from grain_morph.writers import read_table, write_table


def test_roundtrip_all_formats(tmp_path):
    df = pd.DataFrame(
        {"grain_uid": ["f:1", "f:2"], "area_um2": [1.5, 2.5], "qc_pass": [True, False]}
    )
    for fmt in ("parquet", "csv", "feather"):
        p = write_table(df, tmp_path / f"t_{fmt}", fmt)
        back = read_table(p, fmt)
        assert list(back["grain_uid"]) == ["f:1", "f:2"]
        assert back["area_um2"].tolist() == [1.5, 2.5]
