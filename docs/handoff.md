# Handoff — three pending features

Requested during the 2026-08-05 session, deferred (a detection-noise bug took
priority). Recommended order: **3 → 1 → 2** (feature 3 changes the output schema
that 1 and 2 build on). Each should go through the normal brainstorm → plan →
TDD → review flow; the design decisions below are already settled, so planning
should be quick.

Current state at handoff: `main` has the full pipeline + `overlay` command +
tuned defocus thresholds (5.25/0.70) + the empty-frame noise fix. Calibration is
a placeholder (basic 20 / zoom 8 µm/px); real µm/px is not yet available (it's a
device optical spec — see `followups.md` and the session on calibration).

---

## Feature 3 — Omit µm sizes until µm/px is confirmed (do first)

**Why:** µm/px is unknown (placeholder), so every `_um` column currently ships a
misleading physical size. The QC gate is entirely in image pixels
(`edge_width_px`, `min_ecd_px`), so it's calibration-independent — only the size
*outputs* depend on calibration.

**Decision:** make calibration **optional**. When a camera has no
`calibration.um_per_px` set, emit **px-native** size columns only and omit the
`_um` columns for that camera's grains; do **not** fail loudly. When calibration
*is* set, emit both (as today).

**Where / what:**
- `measure.py::measure_polygon` — currently emits `area_px` (only px one) plus
  `area_um2, ecd_um, feret_max_um, feret_min_um, major_axis_um, minor_axis_um,
  perimeter_um`. Add px-native columns computed **before** scaling:
  `ecd_px` (note: `qc.qc_metrics` already computes an `ecd_px`; reconcile —
  measure should own the size px columns), `feret_max_px`, `feret_min_px`,
  `major_axis_px`, `minor_axis_px`, `perimeter_px`. Then `_um = _px * um_per_px`
  only when calibration is present.
- `config.py` — allow `calibration.um_per_px.{basic,zoom}` to stay `null`
  without `um_per_px(camera)` raising; add a helper like
  `Config.has_calibration(camera) -> bool`.
- `pipeline.py::process_frame` / row assembly — if the frame's camera is
  uncalibrated, populate px columns and leave `_um` columns absent (or NaN with
  a documented convention; prefer *absent* so nobody trusts a fake number).
  Remove/relax the up-front fail-loud calibration check (Task 12) — uncalibrated
  now means px-only, not an error. **Document this contract change** (README +
  the calibration-required note).
- `aggregate.py` / `report.py` — percentiles/summaries must work on px columns
  when `_um` is absent (e.g. summarize `ecd_px`/`feret_min_px`).

**Acceptance / tests:**
- `detect` with no calibration → grains table has px size columns, no `_um`
  columns, and the run does **not** abort.
- `detect` with calibration → both px and `_um` columns (unchanged behavior).
- `aggregate` produces size summaries from px columns when `_um` absent.

**Caveat:** biggest of the three — touches measure/config/pipeline/aggregate/
report/cli + the README calibration contract. Do it first so features 1 & 2
build on the final schema.

---

## Feature 1 — Per-frame morphometrics summary table

**Why:** user wants per-frame granularity ("a table with morphometrics per
frame"). The per-grain table already carries `frame_id`; there is no per-frame
roll-up (`aggregate` only summarizes per `(sample_id, camera)`).

**Decision:** add a per-frame summary — one row per `frame_id`.

**Where / what:**
- `aggregate.py::aggregate_run` — add a `"per_frame"` DataFrame to its returned
  dict (keys currently `summary`, `rejection_by_ecd`, `accepted`). One row per
  `frame_id` with: `sample_id, camera, n_detected, n_accepted, n_rejected`,
  a count per `flag_*`, and accepted-grain size stats (D10/D50/D90 and/or
  median of `ecd` and `feret_min`) + means of the shape descriptors
  (`aspect_ratio, solidity, circularity, wadell_roundness, wadell_sphericity,
  curvature_entropy`). Reuse the existing per-group summary logic, grouped by
  `frame_id` instead of `(sample_id, camera)`.
- Respect feature 3: summarize px columns when `_um` absent.
- `cli.py::aggregate` — write the `per_frame` table alongside `summary` /
  `rejection_by_ecd` / `accepted` via `write_table`.

**Acceptance / tests:** `aggregate_run` returns a `per_frame` table with one row
per frame, `n_accepted + n_rejected == n_detected` per row, size percentiles
ordered; CLI writes it.

---

## Feature 2 — `detect --overview` toggle (annotated overview images per run)

**Why:** user wants overview images with grain annotations emitted as a run
output, not only via the on-demand `overlay` command (which requires explicit
`--frames`). This was directly useful for diagnosing the noise over-detection.

**Decision:** add a `--overview` flag to `detect`. Reuse the existing
`grain_morph.overlay.make_overlays` renderer (QC-colored outlines, full-res-then-
downsample, label auto-suppression). Render as a **post-detect pass** over the
just-written grains + contours — NOT inside the parallel `process_frame` workers
(keep the hot path clean and `process_frame` picklable/pure; matplotlib in loky
workers is avoidable overhead).

**Where / what:**
- `cli.py::detect` — add `--overview/--no-overview` (default off),
  `--overview-factor` (default 4), and a scope control (default: cap/sample,
  since a run can be thousands of frames — reuse a stratified sample like the
  contact sheets, or `--overview-frames all` to force every frame). After
  `run_detect` completes, read `OUT/grains` + `OUT/contours` and call
  `make_overlays` for the selected frames into `OUT/overviews/`.
- Alternatively expose a thin `run_overview(out_dir, frames_root, cfg, ...)` in
  `overlay.py` that the CLI calls, so the logic is testable without the CLI.
- Respect feature 3 (px-only labels/titles when `_um` absent).

**Caveat — density:** on this data, near-empty frames are the norm and a few are
noise-heavy; with the noise fix, overviews will be clean. Still, cap/sample by
default so a big run doesn't write thousands of PNGs silently (log what was
sampled — the "no silent caps" rule).

**Acceptance / tests:** `detect --overview` on a tiny run writes overview PNGs to
`OUT/overviews/`; default-off writes none; a capped run logs the sample size.

---

## Notes carried from this session
- `overlay` command, tuned defocus thresholds, and the empty-frame noise fix are
  on `main`. `docs/followups.md` has the agglomerate-detection gap and the
  defocus-tuning provenance.
- Calibration is a device optical spec, not in the software manual / CSV /
  X-Plorer exports; get it from the hardware manual / Retsch datasheet or a
  reticle image (empirical cross-calibration was ruled out — our frames are a
  subset of the run the instrument sized over).
