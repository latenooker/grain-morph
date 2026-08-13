# Design — `groundtruth`: lightweight grain-labeling GUI

Status: approved (2026-08-13). Branch: `feat/groundtruth-gui`.

## Motivation

QC groundtruthing (hand-labeling grains include/exclude/special to validate and
tune the QC gate) was previously done with scratch scripts (see `handoff.md`
P_01/P_17 hand-labeling). Make it a first-class, resumable tool: sample grains
from a completed `detect` run, show each one zoomed with its mask, and capture a
class per grain by keystroke.

## Inputs / outputs

- Reads a completed **`detect` OUT dir**: `grains/` (identity + metrics + QC
  flags, incl. absolute `frame_path`, `label`, `centroid_x/y`) and `contours/`
  (`grain_uid → wkt` subpixel polygon). No re-detection.
- `--frames-dir` optional: relocate source frames if the stored absolute
  `frame_path`s are stale (e.g. the external drive remounted elsewhere) — the
  frame image is loaded from `frames_dir/<frame_id>.<ext>` when given.
- Writes **`groundtruth.csv`** (default `run_dir/groundtruth.csv`), keyed by
  `grain_uid`: `label` (int), `label_name`, predicted `qc_status` + `flags`,
  `ecd_px`, `sample_id`, `camera`, `frame_id`, `labeled_at`. Keyed by
  `grain_uid` so it joins back to `grains` to score hand-label vs predicted.
- Public **`run_groundtruth(run_dir, *, n_frames=None, frac_frames=None,
  n_grains=None, frac_grains=None, seed=0, config=None, out=None,
  frames_dir=None) -> pandas.DataFrame`** (opens the window, returns the labels
  frame) + thin CLI **`grain-morph groundtruth`**.

## Sampling — Latin-hypercube over the continuous QC-driving metrics

The QC flags are thresholds on continuous per-grain metrics already in the
grains table; the sample spans those raw values so labels inform each cut.

1. **Frame pool:** `n_frames`/`frac_frames` → seed-randomly pick that many
   distinct `frame_id`s; candidates = their grains. Else all grains.
2. **Target** `N` = `n_grains` (or `round(frac_grains·|candidates|)`), capped at
   `|candidates|`.
3. **Axes** (default, config-overridable via `groundtruth.lhs_axes`):
   `ecd_px` (→ `flag_too_small`, also the size axis), `edge_width_px` +
   `contrast` (→ `flag_defocus`), `qc_aspect_ratio` (→ `flag_sliver`),
   `qc_solidity` (→ `flag_possible_agglomerate`). Each axis is empirical-
   quantile-ranked to [0,1) over candidates (robust to scale/outliers).
4. **LHS design:** `scipy.stats.qmc.LatinHypercube(d=len(axes), seed=seed)
   .random(M)`; greedily match each design point to the nearest **unused**
   candidate in quantile space (`scipy.spatial.cKDTree` / argmin). Space-filling
   ⇒ both sides of every threshold are covered.
5. **Boolean-flag reservation:** `flag_border` (`touches_border`) and
   `flag_no_polygon` (`has_polygon` False) have no continuous axis. Reserve up to
   `groundtruth.reserved.border` / `.no_polygon` slots (seed-randomly, when
   present in the pool) before the LHS fills the rest; `M = N − reserved_taken`.
   Grains missing any continuous axis value (NaN) are excluded from the LHS and
   only reachable via reservation.
6. Realized per-axis quantile coverage + reserved counts are `log()`ged; a
   stratum that ran dry is noted, never silently dropped.

Determinism: identical `(run_dir, spec, seed)` ⇒ identical selection.

## GUI (matplotlib, one grain at a time)

- Crop of the source frame around the polygon bbox + `groundtruth.crop_pad_px`
  padding (clamped to frame bounds); WKT polygon overlaid (outline + `mask_alpha`
  translucent fill). Status bar: `grain_uid`, sample/camera, ECD (px, µm if
  calibrated), current assigned label, progress `i/N`.
- **Predicted QC status is hidden by default** (unbiased ground truth); key `i`
  reveals accept/reject + flags on demand (`groundtruth.show_predicted_status`
  can default it on).
- **Keys** (class map configurable; default): `0`=exclude, `1`=include,
  `2`=special → assign **and** advance; `m` toggle mask; `n`/`p` (or →/←)
  next/prev without labeling; `u` unset current; `i` reveal predicted status;
  `h` help; `q`/`esc` save+quit.
- **Autosave after every keystroke** → crash-safe and **resumable**: relaunch
  loads the existing `groundtruth.csv`, preserves labels, and starts at the first
  unlabeled grain.

## Config — new `groundtruth` section (`configs/default.yaml` + model)

`classes` (list of `{key, value, name}`, default 0/1/2 =
exclude/include/special), `lhs_axes` (the five columns above), `reserved`
(`{border, no_polygon}`), `crop_pad_px`, `mask_alpha`,
`show_predicted_status` (bool). Read through `Config`, per the no-magic-numbers
rule.

## Module layout / reuse

- New `src/grain_morph/groundtruth.py`: sampling, label store, `_LabelSession`
  state machine, `_crop_bbox`, and the matplotlib driver `run_groundtruth`.
- Relocate the two table readers `_read_grains` and the contour reader from
  `cli.py` into `io.py` (`read_grains`, `read_contours`) so `cli` and
  `groundtruth` share them without a circular import; `cli.py` keeps the
  `groundtruth` typer command (imports `run_groundtruth`), `groundtruth.py`
  imports only `io`. Existing CLI tests guard the move.

## Testing (TDD on the pure parts; GUI kept thin)

The matplotlib event loop isn't unit-testable, so logic is factored out:

- **Sampler** (`_sample_grains`): seeded determinism; `N` selected & capped;
  frame-pool respected; every present QC-driving range spanned (min & max
  quantile grains selectable); reserved border/no-polygon honored; NaN grains
  excluded from LHS.
- **Label store**: write → append → **resume** round-trip keyed by `grain_uid`;
  autosave writes after each `assign`; malformed/missing file handled.
- **`_LabelSession`**: `assign`/`next`/`prev`/`toggle_mask`/`unset`/index
  transitions and "start at first unlabeled" — no display.
- **`_crop_bbox`**: bbox from polygon + pad, clamped to frame shape.
- CLI: `grain-morph groundtruth --help` registers; headless (`Agg`) launch
  raises a clear "needs an interactive backend" error, not a traceback.

## Docs

README (new command + `groundtruth` config rows), `docs/tutorial.md` (a
"groundtruth your QC gate" step on the fixtures), and a new `docs/groundtruth.md`
(sampling method, keys, output schema, how to score the gate against labels).

## Non-goals

Editing polygons/masks; multi-user label reconciliation; active-learning loops;
anything that writes back into the `detect` output. Labels live only in
`groundtruth.csv`.
