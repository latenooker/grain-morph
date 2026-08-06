# Per-frame summary + `detect --overview` — design spec

Date: 2026-08-05
Status: **draft, pending user review**

Covers handoff features **1** (per-frame morphometrics summary) and **2**
(`detect --overview` annotated images). Feature 3 (omit-µm / optional
calibration) is deferred; these two are built against the **current** schema
(`_um` columns always present) and will need a small follow-up when Feature 3
lands — an accepted trade-off of doing 1 & 2 first.

## 1. Purpose

- **Feature 1:** add a per-`frame_id` roll-up to `aggregate`. Today `aggregate`
  only summarizes per `(sample_id, camera)`; users want per-frame granularity,
  including a per-QC-criterion count so it's visible *what* tripped on each
  frame.
- **Feature 2:** let `detect` emit annotated overview PNGs (QC-colored grain
  outlines over each source frame) as a run output, without the on-demand
  `overlay` command's explicit `--frames` list.

## 2. Feature 1 — per-frame summary table

### 2.1 Approach

Reuse the existing per-group summariser verbatim, grouped by `frame_id` instead
of `(sample_id, camera)`. `aggregate._summarize_group` already produces exactly
the wanted fields — counts, **one count per QC flag**, D10/D50/D90 of the size
columns, and mean/SD of every shape descriptor present — so no new statistics
code is written; only the grouping key changes.

### 2.2 Changes (all in `aggregate.py`)

Extract the shared group-apply + integer-cast bookkeeping (currently inside
`_build_summary`) into one helper, so both tables share it:

```python
def _apply_group_summary(grains: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """groupby(keys) -> _summarize_group per group -> cast count cols to int64."""
    table = (
        grains.groupby(keys, sort=True)
        .apply(_summarize_group, include_groups=False)
        .reset_index()
    )
    fixed = {"n_total", "n_accepted", "n_rejected"}
    count_cols = [c for c in table.columns if c in fixed or c.startswith("n_flag_")]
    table[count_cols] = table[count_cols].astype("int64")
    return table

def _build_summary(grains):   return _apply_group_summary(grains, ["sample_id", "camera"])
def _build_per_frame(grains): return _apply_group_summary(grains, ["sample_id", "camera", "frame_id"])
```

`aggregate_run` adds `"per_frame"` to its returned dict (now `summary`,
`rejection_by_ecd`, `accepted`, `per_frame`), built from the same `working`
frame (with `qc_pass` recomputed) as the other tables.

**No CLI change needed.** `cli.py::aggregate` already writes every table in the
returned dict (`for name, table in tables.items(): write_table(table, out_dir /
name, ...)`), so the new table lands at `OUT/per_frame` automatically.

### 2.3 Row shape

One row per `(sample_id, camera, frame_id)`:

```
sample_id, camera, frame_id,
n_total, n_accepted, n_rejected,
n_flag_border, n_flag_defocus, n_flag_sliver, n_flag_too_small,
n_flag_no_polygon, n_flag_possible_agglomerate,
ecd_um_D10/D50/D90, feret_min_um_D10/D50/D90,
<shape>_mean, <shape>_sd  (for each shape descriptor present)
```

### 2.4 Semantics of the per-criterion counts (documented, intentional)

- `n_flag_*` counts grains where that flag is **set**, matching the existing
  `summary` table's convention. Flags are **not** mutually exclusive (a grain
  can be both `flag_defocus` and `flag_border`), so the per-criterion counts can
  sum to **more** than `n_rejected`. This is a "what tripped on this frame"
  breakdown, not a partition of the rejects.
- `flag_possible_agglomerate` is counted but is **not** in the default
  `disqualifying_flags` (default: defocus, border, too_small, sliver,
  no_polygon), so it appears as a per-frame signal without contributing to
  `n_rejected` — a useful free diagnostic given the open agglomerate-detection
  gap in `followups.md`.

### 2.5 Tests

- `aggregate_run` returns a `per_frame` table.
- One row per `(sample_id, camera, frame_id)`.
- `n_accepted + n_rejected == n_total` per row.
- Size percentiles ordered (`D10 <= D50 <= D90`).
- A `n_flag_*` column exists for every `flag_*` column present in the input.
- CLI `aggregate` writes the `per_frame` table file.

## 3. Feature 2 — `detect --overview`

### 3.1 Approach

Add a `--overview` toggle to `detect`. When set, after `run_detect` completes,
render QC-colored overlay PNGs for **every frame that produced ≥1 detection**,
reusing the existing `overlay.make_overlays` renderer. Run it as a **post-detect
pass in the main process** — never inside the parallel `process_frame` workers —
to keep the hot path picklable/pure and matplotlib out of loky workers.

Because `grains` only contains rows for frames that produced detections, "all
unique `frame_id`s in `grains`" **is** "all frames with detections" — no
separate empty-frame filter is needed, and there is no cap (a run's frames are
mostly near-empty; the noise fix keeps the detected set small).

### 3.2 Changes (in `cli.py`)

New options on `detect`:

- `--overview/--no-overview` — default **off**.
- `--overview-factor` — integer downsample factor, default `_DEFAULT_OVERLAY_FACTOR` (4).

New CLI-level helper (testable without invoking Typer), calling the same read
helpers the `overlay` command already uses:

```python
def _write_overviews(run_dir: Path, frames_dir: Path, cfg: Config, factor: int) -> None:
    grains = _read_grains(run_dir / "grains", cfg)
    if grains is None:                       # zero-object run -> log + skip, no crash
        typer.echo("No grains detected; nothing to overlay.")
        return
    frame_ids = sorted(grains["frame_id"].astype(str).unique())   # every frame with >=1 detection
    contours = _read_overlay_contours(run_dir, frame_ids, cfg)
    paths = make_overlays(grains, contours, frames_dir, run_dir / "overviews",
                          frame_ids, factor, cfg, labels=None)
    typer.echo(f"Wrote {len(paths)} overview PNG(s) to {run_dir / 'overviews'}")
```

`detect` calls it after `run_detect(...)` when `overview` is true. Output goes to
`OUT/overviews/`, parallel to the `overlay` command's directory behavior.

**Placement:** the helper lives in `cli.py`, not `overlay.py` (as the handoff
floated), because the read helpers `_read_grains` / `_read_overlay_contours`
already live in `cli.py` and the existing import direction is `cli -> overlay`;
a plain function in `cli.py` is still unit-testable without Typer.

### 3.3 Caveat — contours dependency

Overlays only get outlines drawn when per-frame contour files exist on disk,
i.e. when the run had `cfg.output.save_contours` true (the same dependency the
`overlay` command documents). `--overview`'s help text will state this. If
save_contours was off, overviews render the frame images without outlines rather
than failing. (Confirm the `save_contours` default during planning.)

### 3.4 Tests

- `detect --overview` on a tiny run writes one PNG per detected frame to
  `OUT/overviews/`.
- Default (no flag) writes no `overviews/` directory.
- `_write_overviews` on a zero-grain run logs and writes nothing, without
  crashing.
- Overview count equals the number of distinct `frame_id`s in the grains table.

## 4. Out of scope

- Feature 3 (optional calibration / omit-µm). When it lands, both tables here
  gain a px-only path (`per_frame` summarizes px size columns when `_um` absent;
  overview labels/titles go px-only). Tracked in `handoff.md`.
- Sampling/capping the overview set (decided against: render all detected
  frames).
- The agglomerate-detection strengthening in `followups.md`.
