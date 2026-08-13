# Tutorial — your first run (10 minutes)

A hands-on walk from raw frames to summary tables, using the small example
frames **committed in this repo** — no external data or hardware needed. If you
just want the concepts, read [`index.md`](index.md) instead.

## 0. Install

See the [README](../README.md#install). In short, from the repo root:

```bash
uv sync --extra dev        # or: pip install -e ".[dev]"
```

This installs the `grain-morph` console script. Prefix commands with `uv run`
if you used `uv` (`uv run grain-morph …`).

## 1. The example data

`tests/fixtures/real/` holds 11 real CAMSIZER X2 frames, **4× downsampled** so
they're small enough to commit (~260 KB each). Two samples, two cameras:

```bash
ls tests/fixtures/real/
# P_01_cs_003_b_0001189.bmp   ← sample P_01_cs, run 003, basic camera, frame 0001189
# P_01_cs_003_b_back.bmp      ← a blank/background frame (for flat-fielding)
# P_17_cs_002_z_0010022.bmp   ← sample P_17_cs, zoom camera
# ...
```

The filename encodes identity: `{sample}_{camera}_{frame-or-"back"}`. `_b_` =
basic camera, `_z_` = zoom; `_back` = a blank frame the pipeline divides out to
correct uneven illumination.

> **All frames for one sample in a single directory, with no identity in the
> filenames?** Pass it in instead: `detect … --sample-id P_01_cs --camera basic`
> stamps every frame in the directory with that identity, so the filenames only
> need to differ (a blank is recognized by a `back` in its name — see
> `filename.blank_regex`). You can override just one of the two and let the
> filename supply the other. See the [config reference](../README.md#config-reference).

> **Because these are downsampled, the µm sizes below are illustrative, not
> physical.** On real full-resolution frames you'd use your instrument's true
> `um_per_px`. Everything about *how to run and read the pipeline* is identical.

## 2. A minimal config

Calibration is **required** — the pipeline refuses to invent physical sizes. Put
your per-camera µm/px in a small YAML that's deep-merged over the packaged
defaults. We also ask for CSV output so you can open the tables in anything:

```bash
cat > tutorial.yaml <<'YAML'
calibration:
  um_per_px: { basic: 20.0, zoom: 8.0 }   # illustrative values for the fixtures
output:
  format: csv
YAML
```

Every other knob keeps its default from `configs/default.yaml` (see the
[config reference](../README.md#config-reference)).

## 3. Detect + measure + QC-flag (Stage 1)

```bash
uv run grain-morph detect tests/fixtures/real out --config tutorial.yaml
```

This processes every frame and writes to `out/`:

```
out/
├── grains/          one table per frame — every detected grain, fully measured + flagged
├── contours/        the subpixel boundary polygon (WKT) per grain
├── manifest.csv     one row per frame (status, #objects) — also what makes re-runs resumable
├── summary.json     run totals
└── run_config.yaml  the exact resolved config this run used (provenance)
```

Look at the run totals:

```bash
cat out/summary.json
# {"n_frames": 7, "n_frames_by_status": {"ok": 7}, "n_objects": 10,
#  "rejection_counts": {"flag_border": 1, "flag_defocus": 1, "flag_too_small": 1}, ...}
```

7 frames had grains, 10 objects total, a few already flagged. **Nothing is
dropped** — flagged grains are recorded, not discarded (see step 6).

## 4. Look at one grain

Each `grains/*.csv` row is one grain with ~110 columns. The ones you'll use most:

| Column | What it is |
|---|---|
| `grain_uid`, `frame_id`, `camera` | identity |
| `ecd_um`, `feret_max_um`, `feret_min_um` | size (equivalent-circle diameter, max/min caliper) |
| `aspect_ratio`, `solidity`, `circularity`, `wadell_roundness`, `wadell_sphericity` | shape |
| `edge_width_px`, `contrast` | focus/sharpness (drive `flag_defocus`) |
| `flag_*`, `qc_pass` | QC flags and the pass/fail verdict |

Full definitions with formulas + which library computes each are in
[`morphometrics.md`](morphometrics.md).

## 5. Aggregate into summaries (Stage 2)

```bash
uv run grain-morph aggregate out/grains out/agg --config tutorial.yaml
```

Writes four tables to `out/agg/`:

- **`summary.csv`** — one row per `(sample, camera)`: counts, a count per QC
  flag, D10/D50/D90 of `ecd_um` and `feret_min_um`, and mean/SD of every shape
  descriptor — **over accepted grains only**.
- **`per_frame.csv`** — the same, one row per frame.
- **`rejection_by_ecd.csv`** — rejection rate by size class, so a size-dependent
  gate is visible.
- **`accepted.csv`** — just the grains that passed.

```bash
column -s, -t out/agg/summary.csv | cut -c1-120   # peek (CSV → aligned columns)
```

The gate is applied *here*, not at detect — so you can re-run `aggregate` with a
different `qc.disqualifying_flags` to tighten or loosen it **without
re-detecting**. See [`aggregation.md`](aggregation.md).

## 6. See what the pipeline saw

Overlay the detected outlines on a frame, colored by QC outcome (green = kept,
red = rejected):

```bash
uv run grain-morph overlay out tests/fixtures/real out/overlays \
    --frames P_01_cs_003_b_0001189
```

Or render overlays for **every** detected frame in one shot by adding
`--overview` to the original `detect` call (writes `out/overviews/`). For the
full QC review set (focus scatter, rejected-grain contact sheets):

```bash
uv run grain-morph report out/grains tests/fixtures/real out/report --config tutorial.yaml
```

## 6b. Groundtruth the QC gate (optional)

Want to check whether the gate is accepting/rejecting the right grains? Label a
sample by hand:

```bash
uv run grain-morph groundtruth out --n-grains 12 --seed 0 --config tutorial.yaml
```

A window shows one grain at a time (zoomed, with its polygon mask) drawn from a
Latin-hypercube sample across the QC-driving metrics. Press `1` to include, `0`
to exclude, `2` for a special case (`m` toggles the mask, `i` reveals the
predicted status, `q` saves and quits). Labels autosave to `out/groundtruth.csv`,
keyed by `grain_uid`, so you can join them back to `out/grains` and see where the
gate agrees or disagrees. Details: [`groundtruth.md`](groundtruth.md). *(Needs a
desktop matplotlib backend; on a fixtures-only machine this is just to see the
shape of the workflow.)*

## 7. The mental model

- **Two stages.** `detect` (expensive, parallel, per-frame) records *everything*
  and enforces *nothing*. `aggregate` (cheap) applies the QC gate — re-runnable.
- **Flag, never drop.** Every grain is measured and flagged; the accept/reject
  decision is a late, reversible filter (`qc.disqualifying_flags`).
- **Two geometries.** A subpixel **polygon** (most metrics) and a **raster mask**
  (ellipse fit + QC geometry). See [`delineation.md`](delineation.md).
- **Pixels are truth; µm are only as good as your calibration.**

## Where to go next

- Concepts + doc map: [`index.md`](index.md)
- Every metric's formula + provenance: [`morphometrics.md`](morphometrics.md)
- QC flags and tuning: [`qc.md`](qc.md) and the
  [README QC section](../README.md#how-the-qc-gate-works-and-how-to-tune-it)
- Hand-labeling to validate the gate: [`groundtruth.md`](groundtruth.md)
- Known limitations & open work: [`followups.md`](followups.md)
- Contributing code: [`../CONTRIBUTING.md`](../CONTRIBUTING.md)
