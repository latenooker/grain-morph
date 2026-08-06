# Per-frame summary + `detect --overview` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a per-`frame_id` roll-up table to `aggregate` (with a per-QC-criterion count), and a `detect --overview` flag that renders QC-colored overlay PNGs for every frame with detections.

**Architecture:** Feature 1 reuses `aggregate._summarize_group` grouped by `frame_id` via a shared `_apply_group_summary` helper; the new table rides the existing `aggregate_run` return dict and CLI write loop. Feature 2 adds a main-process post-detect pass (`cli._write_overviews`) that reuses `overlay.make_overlays`, wired to a `--overview` toggle on `detect`.

**Tech Stack:** Python 3.11+, pandas, numpy, Typer (CLI), pytest, `typer.testing.CliRunner`, `tests.synth.make_frame`.

## Global Constraints

- `from __future__ import annotations` at the top of every module (already present in all files touched here — do not remove).
- Modern type hints (`str | None`, `list[str]`); Google-style docstrings on every public function; private symbols prefixed `_`.
- **No new external dependencies.**
- Built against the **current** schema (`_um` columns always present). Feature 3 (omit-µm) is out of scope.
- Every commit message ends with these two trailers (append to each commit in this plan):
  ```
  Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_018tBqGZ6kNb9DtxdcivFef3
  ```
- Work happens on branch `feat/per-frame-and-overview` (already created; the spec is committed there).
- Run tests with `uv run pytest` (dev extras installed via `uv sync --extra dev`).

---

### Task 1: Per-frame summary table in `aggregate`

**Files:**
- Modify: `src/grain_morph/aggregate.py` (extract `_apply_group_summary`; add `_build_per_frame`; add `"per_frame"` to `aggregate_run`)
- Test: `tests/test_aggregate.py` (new per-frame unit test)
- Test: `tests/test_cli.py` (new: CLI writes `per_frame` table)

**Interfaces:**
- Consumes: existing `_summarize_group(group) -> pd.Series`, `_recompute_qc_pass`, `Config`.
- Produces:
  - `_apply_group_summary(grains: pd.DataFrame, keys: list[str]) -> pd.DataFrame`
  - `_build_per_frame(grains: pd.DataFrame) -> pd.DataFrame` (one row per `(sample_id, camera, frame_id)`)
  - `aggregate_run(...)` return dict now includes key `"per_frame"` alongside `"summary"`, `"rejection_by_ecd"`, `"accepted"`.

- [ ] **Step 1: Write the failing unit test**

Add to `tests/test_aggregate.py`:

```python
def _grains_with_frames():
    """Two frames of one (sample, camera), each with 100 grains."""
    rng = np.random.default_rng(1)
    parts = []
    for frame in ("S1_b_0000001", "S1_b_0000002"):
        n = 100
        parts.append(pd.DataFrame({
            "sample_id": ["S1"] * n,
            "camera": ["basic"] * n,
            "frame_id": [frame] * n,
            "ecd_um": rng.uniform(50, 500, n),
            "feret_min_um": rng.uniform(40, 450, n),
            "solidity": rng.uniform(0.9, 1.0, n),
            "flag_defocus": rng.random(n) < 0.1,
            "flag_border": rng.random(n) < 0.05,
            "flag_too_small": [False] * n,
            "flag_sliver": [False] * n,
            "flag_no_polygon": [False] * n,
        }))
    return pd.concat(parts, ignore_index=True)


def test_per_frame_table_shape_counts_and_flag_columns():
    out = aggregate_run(_grains_with_frames(), load_config(None))
    pf = out["per_frame"]
    # one row per (sample_id, camera, frame_id)
    assert len(pf) == 2
    assert set(pf["frame_id"]) == {"S1_b_0000001", "S1_b_0000002"}
    assert {"sample_id", "camera", "frame_id"}.issubset(pf.columns)
    # counts consistent
    assert (pf["n_total"] == 100).all()
    assert (pf["n_accepted"] + pf["n_rejected"] == pf["n_total"]).all()
    # per-criterion count present for every flag in the input
    for flag in ("flag_defocus", "flag_border", "flag_too_small",
                 "flag_sliver", "flag_no_polygon"):
        assert f"n_{flag}" in pf.columns
    # size percentiles ordered (every frame has accepted grains here)
    assert (pf["ecd_um_D10"] <= pf["ecd_um_D50"]).all()
    assert (pf["ecd_um_D50"] <= pf["ecd_um_D90"]).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_aggregate.py::test_per_frame_table_shape_counts_and_flag_columns -v`
Expected: FAIL with `KeyError: 'per_frame'`.

- [ ] **Step 3: Refactor `_build_summary` onto a shared helper and add `_build_per_frame`**

In `src/grain_morph/aggregate.py`, replace the body of `_build_summary` (currently the groupby-apply + int-cast at lines ~204-227) with a call to a new shared helper, and add the per-frame builder:

```python
def _apply_group_summary(grains: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Apply :func:`_summarize_group` per group of `keys`, tidy count dtypes.

    Args:
        grains: Per-grain table with `qc_pass` already recomputed.
        keys: Grouping columns (e.g. `["sample_id", "camera"]`).

    Returns:
        One row per distinct `keys` combination, with the count columns
        (`n_total`, `n_accepted`, `n_rejected`, `n_flag_*`) cast back to
        `int64` (see note below on why the apply upcasts them to float).
    """
    table = (
        grains.groupby(keys, sort=True)
        .apply(_summarize_group, include_groups=False)
        .reset_index()
    )
    # `_summarize_group` mixes always-integer counts with float/`nan` stat
    # columns into one per-group Series, which upcasts the whole thing to
    # float64; cast the count columns back to int64 for a tidy result.
    fixed_count_cols = {"n_total", "n_accepted", "n_rejected"}
    count_cols = [c for c in table.columns if c in fixed_count_cols or c.startswith("n_flag_")]
    table[count_cols] = table[count_cols].astype("int64")
    return table


def _build_summary(grains: pd.DataFrame) -> pd.DataFrame:
    """Per `(sample_id, camera)` summary table (see module docstring).

    Args:
        grains: Per-grain table with `qc_pass` already recomputed.

    Returns:
        One row per `(sample_id, camera)`, columns per :func:`_summarize_group`.
    """
    return _apply_group_summary(grains, ["sample_id", "camera"])


def _build_per_frame(grains: pd.DataFrame) -> pd.DataFrame:
    """Per `frame_id` summary table — same columns as `summary`, per frame.

    One row per `(sample_id, camera, frame_id)`; carries the per-QC-criterion
    counts (`n_flag_*`), size percentiles, and shape mean/SD that
    :func:`_summarize_group` produces. `n_flag_*` counts grains where each
    flag is *set* (not mutually exclusive; can sum to more than `n_rejected`).

    Args:
        grains: Per-grain table with `qc_pass` already recomputed; must have a
            `frame_id` column.

    Returns:
        One row per `(sample_id, camera, frame_id)`.
    """
    return _apply_group_summary(grains, ["sample_id", "camera", "frame_id"])
```

- [ ] **Step 4: Add `"per_frame"` to the `aggregate_run` return dict**

In `aggregate_run` (the `return {...}` near the end), add the new table and document it. The return becomes:

```python
    return {
        "summary": _build_summary(working),
        "rejection_by_ecd": _build_rejection_by_ecd(working),
        "accepted": accepted,
        "per_frame": _build_per_frame(working),
    }
```

Also add a `"per_frame"` bullet to `aggregate_run`'s Returns docstring and to the module docstring's table list (mirror the existing `"summary"` wording: one row per `(sample_id, camera, frame_id)`, with per-flag counts).

- [ ] **Step 5: Run the per-frame test and the existing aggregate tests**

Run: `uv run pytest tests/test_aggregate.py -v`
Expected: PASS (new test passes; pre-existing `test_summary_*` still pass — `_build_summary` output is unchanged).

- [ ] **Step 6: Write the failing CLI test that `per_frame` is written**

Add to `tests/test_cli.py`:

```python
def test_aggregate_cli_writes_per_frame_table(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    iio.imwrite(run / "S1_b_back.bmp", make_frame(size=(128, 128), objects=[]).image)
    obj = [{"kind": "ellipse", "cx": 64, "cy": 64, "a": 18, "b": 18,
            "angle": 0.0, "blur_sigma": 0.0}]
    iio.imwrite(run / "S1_b_0000001.bmp", make_frame(size=(128, 128), objects=obj).image)
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump({
        "calibration": {"um_per_px": {"basic": 5.0, "zoom": 1.0}},
        "output": {"format": "csv"},
    }))
    out = tmp_path / "out"
    res = runner.invoke(app, ["detect", str(run), str(out), "--config", str(cfg), "--jobs", "1"])
    assert res.exit_code == 0, res.output

    agg_out = tmp_path / "agg"
    res = runner.invoke(app, ["aggregate", str(out / "grains"), str(agg_out), "--config", str(cfg)])
    assert res.exit_code == 0, res.output
    assert (agg_out / "per_frame.csv").exists()
```

- [ ] **Step 7: Run the CLI test**

Run: `uv run pytest tests/test_cli.py::test_aggregate_cli_writes_per_frame_table -v`
Expected: PASS (no production change needed — `cli.aggregate` already writes every table in the dict).

- [ ] **Step 8: Commit**

```bash
git add src/grain_morph/aggregate.py tests/test_aggregate.py tests/test_cli.py
git commit -m "feat(aggregate): per-frame summary table with per-QC-criterion counts"
```
(Append the two global-constraint trailers.)

---

### Task 2: `_write_overviews` post-detect helper

**Files:**
- Modify: `src/grain_morph/cli.py` (add `_write_overviews`, placed after `_read_overlay_contours`)
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: existing `_read_grains(path, cfg) -> pd.DataFrame | None`, `_read_overlay_contours(run_dir, frame_ids, cfg) -> pd.DataFrame`, `make_overlays(...)`, `Config`, `_DEFAULT_OVERLAY_FACTOR`.
- Produces: `_write_overviews(run_dir: Path, frames_dir: Path, cfg: Config, factor: int) -> None` — renders one `{frame_id}_overlay.png` per frame with ≥1 detection into `run_dir / "overviews"`; no-ops (with a logged message) on a zero-grain run.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_cli.py` (imports `runner`, `app`, `make_frame`, `iio`, `yaml` already present; `_detect_with_no_objects` helper already defined in this file):

```python
def test_write_overviews_renders_all_detected_frames(tmp_path):
    from grain_morph.cli import _write_overviews
    from grain_morph.config import load_config

    run = tmp_path / "run"
    run.mkdir()
    iio.imwrite(run / "S1_b_back.bmp", make_frame(size=(128, 128), objects=[]).image)
    obj = [{"kind": "ellipse", "cx": 64, "cy": 64, "a": 18, "b": 18,
            "angle": 0.0, "blur_sigma": 0.0}]
    iio.imwrite(run / "S1_b_0000001.bmp", make_frame(size=(128, 128), objects=obj).image)
    iio.imwrite(run / "S1_b_0000002.bmp", make_frame(size=(128, 128), objects=obj).image)
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump({"calibration": {"um_per_px": {"basic": 5.0, "zoom": 1.0}}}))
    out = tmp_path / "out"
    res = runner.invoke(app, ["detect", str(run), str(out), "--config", str(cfg_path), "--jobs", "1"])
    assert res.exit_code == 0, res.output

    _write_overviews(out, run, load_config(cfg_path), factor=4)
    pngs = sorted((out / "overviews").glob("*.png"))
    assert len(pngs) == 2


def test_write_overviews_zero_grains_no_crash(tmp_path):
    from grain_morph.cli import _write_overviews
    from grain_morph.config import load_config

    out, cfg_path = _detect_with_no_objects(tmp_path)
    _write_overviews(out, tmp_path / "run", load_config(cfg_path), factor=4)
    assert not (out / "overviews").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -k write_overviews -v`
Expected: FAIL with `ImportError: cannot import name '_write_overviews'`.

- [ ] **Step 3: Implement `_write_overviews`**

In `src/grain_morph/cli.py`, add after `_read_overlay_contours`:

```python
def _write_overviews(run_dir: Path, frames_dir: Path, cfg: Config, factor: int) -> None:
    """Render QC-colored overview PNGs for every frame with a detection.

    A main-process post-`detect` pass (never inside the parallel
    `process_frame` workers): reads the just-written grains + contours and
    calls :func:`grain_morph.overlay.make_overlays` for every distinct
    `frame_id` present (which, since only detected grains are stored, is
    exactly the set of frames with at least one detection). Outlines are only
    drawn where per-frame contour files exist, i.e. when the run had
    `cfg.output.save_contours` true (default).

    Args:
        run_dir: Completed `detect` output directory (has `grains/` and,
            for outlines, `contours/`).
        frames_dir: Directory to search for the source frame images.
        cfg: Resolved pipeline configuration.
        factor: Integer factor the full-res composite is downsampled by.
    """
    grains = _read_grains(run_dir / "grains", cfg)
    if grains is None:
        typer.echo("No grains detected; nothing to overlay.")
        return
    frame_ids = sorted(grains["frame_id"].astype(str).unique())
    contours = _read_overlay_contours(run_dir, frame_ids, cfg)
    paths = make_overlays(
        grains, contours, frames_dir, run_dir / "overviews", frame_ids, factor, cfg, labels=None
    )
    typer.echo(f"Wrote {len(paths)} overview PNG(s) to {run_dir / 'overviews'}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -k write_overviews -v`
Expected: PASS (both).

- [ ] **Step 5: Commit**

```bash
git add src/grain_morph/cli.py tests/test_cli.py
git commit -m "feat(cli): _write_overviews post-detect overlay renderer"
```
(Append the two global-constraint trailers.)

---

### Task 3: Wire `--overview` into the `detect` command

**Files:**
- Modify: `src/grain_morph/cli.py` (`detect` command: add options + call `_write_overviews`)
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `_write_overviews(...)` from Task 2, `_DEFAULT_OVERLAY_FACTOR`.
- Produces: `detect` gains `--overview/--no-overview` (default off) and `--overview-factor` (default `_DEFAULT_OVERLAY_FACTOR`); when `--overview` is set, writes `OUT/overviews/`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_cli.py`:

```python
def _detect_one_object_args(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    iio.imwrite(run / "S1_b_back.bmp", make_frame(size=(128, 128), objects=[]).image)
    obj = [{"kind": "ellipse", "cx": 64, "cy": 64, "a": 18, "b": 18,
            "angle": 0.0, "blur_sigma": 0.0}]
    iio.imwrite(run / "S1_b_0000001.bmp", make_frame(size=(128, 128), objects=obj).image)
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump({"calibration": {"um_per_px": {"basic": 5.0, "zoom": 1.0}}}))
    out = tmp_path / "out"
    return run, out, cfg


def test_detect_overview_writes_pngs(tmp_path):
    run, out, cfg = _detect_one_object_args(tmp_path)
    res = runner.invoke(
        app, ["detect", str(run), str(out), "--config", str(cfg), "--jobs", "1", "--overview"]
    )
    assert res.exit_code == 0, res.output
    assert list((out / "overviews").glob("*.png"))


def test_detect_without_overview_writes_no_overviews_dir(tmp_path):
    run, out, cfg = _detect_one_object_args(tmp_path)
    res = runner.invoke(
        app, ["detect", str(run), str(out), "--config", str(cfg), "--jobs", "1"]
    )
    assert res.exit_code == 0, res.output
    assert not (out / "overviews").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -k detect_overview -v`
Expected: `test_detect_overview_writes_pngs` FAILs (no `overviews/` written; `--overview` is an unknown option → non-zero exit). `test_detect_without_overview_writes_no_overviews_dir` may already pass.

- [ ] **Step 3: Add the options and the post-detect call to `detect`**

In `src/grain_morph/cli.py`, add two parameters to the `detect` signature (after `force`) and call the helper after `run_detect`:

```python
    overview: Annotated[
        bool,
        typer.Option(
            "--overview/--no-overview",
            help=(
                "After detection, render QC-colored overview PNGs (one per frame "
                "with >=1 detection) into OUT/overviews/. Outlines require the run's "
                "cfg.output.save_contours (default true)."
            ),
        ),
    ] = False,
    overview_factor: Annotated[
        int,
        typer.Option("--overview-factor", help="Integer factor to downsample overview PNGs by."),
    ] = _DEFAULT_OVERLAY_FACTOR,
```

Then, at the end of the `detect` body:

```python
    cfg = load_config(config)
    run_detect(frames_dir, out_dir, cfg, n_jobs=jobs, force=force)
    if overview:
        _write_overviews(out_dir, frames_dir, cfg, overview_factor)
```

Also add `overview` and `overview_factor` to the `detect` docstring's Args (one line each, matching the option help).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -k detect_overview -v`
Expected: PASS (both).

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS (no regressions).

- [ ] **Step 6: Commit**

```bash
git add src/grain_morph/cli.py tests/test_cli.py
git commit -m "feat(detect): --overview flag renders per-run overlay PNGs"
```
(Append the two global-constraint trailers.)

---

## Self-Review

**Spec coverage:**
- Feature 1 per-frame table → Task 1. Row shape (§2.3) incl. `n_flag_*` per-criterion counts → Task 1 Steps 1/3/4. `n_total` vocabulary + reuse `_summarize_group` → Task 1 Step 3. CLI writes it automatically → Task 1 Steps 6-7. Semantics (§2.4) documented in `_build_per_frame` docstring → Task 1 Step 3.
- Feature 2 `detect --overview` → Tasks 2 + 3. Post-detect main-process pass, all frames with detections → Task 2 Step 3. Options `--overview`/`--overview-factor` (default 4) → Task 3 Step 3. `OUT/overviews/` output → Tasks 2/3. Zero-grain no-crash → Task 2 Step 1. save_contours caveat (§3.3) → documented in `_write_overviews` docstring + `--overview` help.
- Out of scope (§4: Feature 3, sampling/cap, agglomerate) → no tasks, as intended.

**Placeholder scan:** No TBD/TODO; every code and test step has literal content.

**Type consistency:** `_apply_group_summary(grains, keys)`, `_build_per_frame(grains)`, `_write_overviews(run_dir, frames_dir, cfg, factor)`, and the `aggregate_run` `"per_frame"` key are named identically across their definition and consumption. `make_overlays` call matches its 8-arg signature `(grains, contours, frames_root, out_dir, frame_ids, factor, cfg, labels)`.
