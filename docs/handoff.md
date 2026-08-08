# Handoff — state of grain-morph

Last updated: 2026-08-08. This is the rolling baton-pass: what's done, what's
open, and where to look. Deep detail on the open QC/perf items lives in
[`followups.md`](followups.md); this doc orients you and prioritizes.

## Where things stand

`main` is the single active branch and is fully pushed to
`github.com/latenooker/grain-morph`. The pipeline runs end-to-end
(detect → aggregate → report/overlay), tested (95 passing, 1 realdata skip),
ruff + mypy clean.

**Calibration is still a placeholder** (`basic 20 / zoom 8 µm/px` in test runs);
real µm/px is a device optical spec not yet available. So **every `*_um`
column/percentile in existing outputs is placeholder-scaled — not physical.** All
pixel-native, ratio, and shape metrics, and the entire QC gate, are
calibration-independent and valid. This is what Feature 3 (below) addresses.

## Done this session (2026-08-06 → 08, all merged to `main`)

- **Per-frame summary table** (`aggregate` → `per_frame`) and **`detect
  --overview`** — the original Feature 1 & 2. Merged via PR #1 (commit `92408bc`).
- **Documentation set** for collaborators: `docs/index.md` (hub),
  `docs/tutorial.md` (runnable on committed fixtures), `delineation.md`,
  `morphometrics.md`, `qc.md`, `aggregation.md`, a revised `README.md`, and
  `CONTRIBUTING.md`. Each carries a **custom-vs-imported** provenance table.
- **Run-scoped blank pairing** (`fix/blank-pairing-run`, commit `e1d5a0b`) —
  `ParsedName.run` + blank key `(sample_id, run, camera)`; non-breaking (default
  schema keeps `run=None`). Closed the one real correctness risk in the
  deployment note.
- Extensive **QC investigation + morphometric bug review** — captured as the
  prioritized open work below and detailed in `followups.md`.

## Open work — prioritized

Each should go through the normal brainstorm → plan → TDD → review flow. Items
1–4 are fully worked out in `followups.md` (with hand-label evidence and
projected numbers); start there for detail.

### 1. Per-camera defocus/size QC gates  *(highest correctness payoff; spec'd)*
Hand-labeling (P_01 + P_17) showed the single global defocus cut is wrong per
camera:
- **basic:** `defocus_edge_width_px` 5.25 → **~7.0** (edge width is the right
  focus axis).
- **zoom:** edge width is size-confounded — gate on **size** instead,
  `min_ecd_px` ~10 → **~30**; keep the zoom edge cut generous (~17).
Make both `defocus_edge_width_px` and `min_ecd_px` **per-camera** in `config.py`
+ `qc.py` (dict per camera, fall back to the global scalar). Projected: +66 sharp
basic grains recovered, zoom composition corrected. See `followups.md`
§"Defocus … per-camera cut" and §"Hand-label results".

### 2. `flag_debris` — low-curvature-entropy fiber/debris flag  *(spec'd)*
`flag_debris = curvature_entropy < 0.5 AND NOT flag_border`, **non-disqualifying
by default**. Removes the 2 accepted debris escapes, labels ~43 already-rejected
fibers, guard spares border-clipped grains. **Interacts with #3** — it relies on
`curvature_entropy`'s current behavior. See `followups.md` §"flag_debris".

### 3. Fix `curvature_entropy`  *(bug; do with/after #2)*
As implemented it is **inverted vs its docstring** (high = smooth/regular, not
rough) because the histogram uses each grain's own curvature range — outlier-
sensitive and not cross-grain comparable. Also the boundary gradient isn't truly
periodic at the seam. Options: (a) keep the computation, correct the docs
(it's a useful regularity score, and #2 depends on it), or (b) redefine with a
**scale-normalized curvature over a fixed domain/bins** (verifiable: circle→0,
ellipse/N-gon have closed-form values) + reference unit tests. (b) changes values
and breaks #2, so sequence them. See `followups.md` and `morphometrics.md`
known-issues.

### 4. Performance — pipeline is disk-bound  *(profiled)*
Frame I/O from the external USB drive is ~80–85% of wall time (165 ms/frame vs
1 ms local). Fixes, in order: **stage the run to local SSD** before processing
(~5–10×), `lru_cache` the blank read, **parallelize the overview pass** (reuse
loky), crop `detect_objects` morphology to bounding boxes. See `followups.md`
§"Processing time is disk-bound".

### 5. Feret min → `shapely.minimum_rotated_rectangle`  *(minor, optional)*
Both Feret diameters are custom-but-consistent on the subpixel hull; min could
use the library's min-rotated-rectangle short side (same geometry). Max has no
clean library swap — leave it. See `morphometrics.md` known-issues.

### 6. Feature 3 — make calibration optional (omit µm until confirmed)  *(schema change; original pending feature)*
The only unshipped item from the original handoff. Detailed spec preserved below
because it lives nowhere else.

---

## Feature 3 spec — omit µm sizes until µm/px is confirmed

**Why:** µm/px is a placeholder, so every `_um` column currently ships a
misleading physical size. The QC gate is entirely in pixels, so only the size
*outputs* depend on calibration.

**Decision:** make calibration **optional**. When a camera has no
`calibration.um_per_px`, emit **px-native** size columns only and omit `_um`
columns for that camera's grains; do **not** fail loudly. When calibration is
set, emit both (as today).

**Where / what:**
- `measure.py::measure_polygon` — add px-native columns computed *before*
  scaling: `feret_max_px`, `feret_min_px`, `major_axis_px`, `minor_axis_px`,
  `perimeter_px` (and reconcile `ecd_px`, which `qc.qc_metrics` also computes —
  measure should own the size px columns). Then `_um = _px * um_per_px` only when
  calibration is present.
- `config.py` — allow `calibration.um_per_px.{basic,zoom}` to stay `null` without
  `um_per_px(camera)` raising; add `Config.has_calibration(camera) -> bool`.
- `pipeline.py` — uncalibrated → px columns populated, `_um` absent; **relax the
  up-front fail-loud calibration check** (`_validate_calibration`). **Ripple to
  watch:** `um_per_px` is currently a required `float` threaded through
  `_build_row` and the **contour record schema** (`grain_uid, wkt, um_per_px`),
  so it must become `float | None` end-to-end.
- `aggregate.py` / `report.py` — percentiles/summaries must work on px columns
  when `_um` is absent (e.g. `ecd_px`/`feret_min_px`). The per-frame table (done)
  and overview labels also need the px-only path.

**Design tension to decide:** "absent vs NaN" for `_um` on uncalibrated grains is
only fully controllable when the *whole run* is uncalibrated — in a run mixing a
calibrated and an uncalibrated camera, the tables merge into one schema and the
uncalibrated rows' `_um` become **NaN on write**. Decide (and document) whether
mixed-calibration runs are allowed / warn.

**Acceptance:** `detect` with no calibration → px size columns, no `_um`, no
abort; with calibration → both (unchanged); `aggregate` summarizes px columns
when `_um` absent.

**Caveat:** biggest change — touches measure/config/pipeline/aggregate/report/cli
+ the contour schema + the README calibration contract.

---

## Key context for whoever picks this up

- **Two geometries:** subpixel **polygon** (most metrics) vs **raster mask**
  (ellipse fit + QC geometry); some metrics have both, under distinct names (the
  QC flags threshold on the raster `qc_solidity`/`qc_aspect_ratio`, not the
  reported polygon columns — see `morphometrics.md` Dual geometry).
- **Flag, never drop.** `detect` records all flags; the accept/reject gate is
  recomputed at `aggregate` from `cfg.qc.disqualifying_flags` — re-gateable
  without re-detecting.
- **Build philosophy:** assemble from maintained packages; custom code is glue +
  standard formulas + a short list of bespoke metrics. Keep it that way.
- **Identity is in the filename, never the folder** — inputs aren't reliably
  foldered per sample; `filename.sample_regex` is the single source of identity
  truth (now run-aware). See `followups.md` §"Deployment".
- **Out of scope but noted:** mineral discrimination (quartz/feldspar/mica) — mica
  is shape-tractable, quartz/feldspar isn't by silhouette; `wadell_sphericity` is
  bimodal (a real platy population, confounded with fiber debris). `followups.md`.

## Where things live (not in git)

- **Test-run outputs:** `/Volumes/LEXAR/Camsizer/PPX/grain-morpho-test` (P_01) and
  `…-P_17_cs` (P_17) — placeholder-calibrated CSV runs used for the QC analysis.
- **Full-res dev frames:** `dev_data/` (gitignored). Committed runnable subset:
  `tests/fixtures/real/` (used by the tutorial).
- Analysis scripts/figures from the QC investigation were scratch (not committed).
