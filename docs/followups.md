# Follow-ups

Known gaps and deferred improvements, recorded so they aren't lost.

## Agglomerate detection misses convexly-fused grains (open)

`flag_possible_agglomerate` fires on `solidity < qc.agglomerate_solidity_max`
(default 0.90). This misses touching grains that fuse into a **fairly convex**
blob, because the necks between them aren't deep enough to drop solidity below
the threshold.

Concrete example (2026-08-05 labeling, `P_01_cs_003_b_0004511` label 3): a
clear 2–3 grain agglomerate measured `solidity = 0.959`, so it was **not**
flagged as an agglomerate. It was only rejected incidentally, via an inflated
`edge_width_px = 11.0` (the "edge" traversed the internal neck between grains),
which tripped `flag_defocus`. A *sharp* agglomerate would slip through as
accepted.

**Options to strengthen it (not yet implemented):**
- Count distance-transform / local-thickness maxima inside the object — >1
  strong peak suggests multiple grains.
- Use convexity-defect depth/count (the depth of the concavities), not just the
  area-based solidity ratio.
- Treat an anomalously high `edge_width_px` relative to grain size as a weak
  agglomerate signal.
- Consider enabling `--split-agglomerates` (distance-transform + watershed,
  already opt-in) by default, or surfacing it more prominently.

Until then: convexly-fused agglomerates may pass QC. The `--split-agglomerates`
opt-in exists for the "split, then judge each product" workflow.

## Defocus thresholds — tuning provenance (informational)

`qc.defocus_edge_width_px = 5.25` and `qc.defocus_contrast_min = 0.70`
(configs/default.yaml) were set on 2026-08-05 by fitting to hand-labeled grains
from `OK_sand_2_005` (basic) and 7 frames across `P_01`/`P_17` (basic), all
blank-division flat-fielded.

Findings from that exercise:
- `edge_width_px` is the primary, scale-independent focus signal (large *sharp*
  grains read low edge_width; large *soft* grains read high). `contrast` was
  demoted to 0.70 — an extreme-debris catch only — because the eroded-core
  contrast estimate reads artificially low on small sharp grains and was
  wrongly rejecting them.
- The accept/reject boundary is a **gray zone ~4.8–5.5 px**, not a sharp line:
  fine-sand `OK_sand` labels lean ~4.5–4.8 while coarse `P_01`/`P_17` labels
  lean ~5.3–5.9, and a mild-defocus grain (`P_01_cs_003_b_0004258` label 5)
  was labeled "intermediate". `5.25` is a deliberate compromise in that band;
  a single global px cut cannot satisfy every sample perfectly. Re-tune per
  study if needed (the `report` focus-scatter + the overlay command are the
  tools).
- `edge_width_px`/`min_ecd_px` are in image pixels, so these cuts are
  independent of the µm/px calibration.

## Defocus edge-width cut is not camera-scale-invariant — per-camera cut needed (open)

`qc.defocus_edge_width_px` is a single global cut, but `edge_width_px` is an
**absolute pixel blur**: for a focused grain it is ~`blur_µm / µm_per_px`, so a
camera with a smaller µm/px (the **zoom** CCD) reads a *higher* `edge_width_px`
for the *same* physical sharpness. The 5.25 cut was fit on **basic**-camera
labels only (see "Defocus thresholds — tuning provenance" above), so it is too
strict for zoom and over-rejects otherwise-sharp zoom (and coarse-basic) grains.
NB: that section's "scale-independent" claim holds across grain *size within one
camera*, not across cameras with different µm/px.

Concrete example (P_01_cs_003 test run, placeholder calib basic 20 / zoom 8
µm/px): `P_01_cs_003_z_0035950:1` (zoom) was rejected **solely** on
`edge_width_px = 5.74 > 5.25`, despite `contrast = 0.82` (sharp). All 6 zoom
objects in that sample were rejected, mostly on defocus.

**Evaluated but rejected — size-ratio normalization** (`edge_width_px / ecd_px`,
i.e. "edge width : core area"): the `µm/px` cancels, giving a calibration-free
ratio ~`blur_µm / size_µm` — elegant, and it *does* rescue the large zoom grain
above (ratio 0.038, in the accepted cluster). But it trades the cross-camera
scale error for a **size** error: a fixed blur is a larger fraction of a small
grain, so small *sharp* grains climb to ratio 0.16–0.27 — *above* genuinely
defocused large grains (`edge_width` 11–12 px → ratio ~0.12). The ordering
inverts, so **no single ratio cut separates the classes** (verified on the
63-grain P_01_cs run). Normalizing by *area* squares the size term and is worse.
So the ratio is at best a secondary cross-check for very large grains, not the
primary gate.

**Recommended fix — make the threshold per-camera, not the metric.** Each camera
has a fixed µm/px, so an absolute `edge_width_px` cut already *is* an absolute-µm
cut *within* a camera. Add a per-camera `defocus_edge_width_px` (e.g.
`{basic: 5.25, zoom: <tuned>}`), falling back to the global value when a camera
is unlisted. Physically zoom needs a *higher* px cut (~2.5× under the placeholder
20/8 → ~13 px, under which none of the P_01 zoom grains would be edge-rejected —
`contrast` still catches the genuinely soft ones). Setting the zoom cut properly
wants a few hand-labeled **zoom** grains rather than the placeholder ratio;
`P_17_cs_002` (75 zoom frames) is a better labeling source than P_01 (6 zoom
objects).

### Hand-label results (2026-08-05) — basic ≠ zoom failure mode

Labeled raw-vs-delineated crops across the edge_width range on P_01_cs_003 +
P_17_cs_002 (operator call, placeholder calib basic 20 / zoom 8 µm/px):

- **Basic — edge_width IS the right axis.** Sharp through `edge_width ≈ 6.66`,
  unusable from `≈ 7.34` up (a big, obviously-defocused grain at 10–12 px is the
  clear reject). So **`defocus_edge_width_px` basic 5.25 → ~7.0**. The old 5.25
  was slicing through a single visually-sharp population.
- **Zoom — edge_width is the WRONG axis; acceptability tracks GRAIN SIZE.** On
  zoom the *crispest* grain had the *highest* edge_width (8.34 px, `ecd = 823`),
  while the fuzziest were the *smallest* (`ecd` 14–19 px, edge 4–6). Operator
  labels: acceptable `ecd` 96–823, intermediate `ecd` 35, unusable `ecd` ≤19 —
  a clean split on **size**, none on edge_width. Cause: at high magnification a
  small grain is only ~15 px across, so its few-px edge is a large fraction and
  it *looks* soft regardless of true focus. Fix for zoom: gate on **size**, not
  edge — **`min_ecd_px` zoom ~10 → ~30** (drops the tiny fuzz, keeps ecd ≥35) —
  and let the zoom edge cut go generous (~17 px, the µm/px-scale-equivalent of
  basic's 7; non-binding on observed data).
- **Ratio verdict, refined.** `edge_width/ecd` separates the *zoom* labels
  cleanly (acceptable ≤0.07, unusable ≥0.23) — because on zoom the problem
  genuinely *is* small-grain softness — but still fails *basic*: a big defocused
  basic grain (edge 10.1, `ecd` 133) has ratio 0.076, *below* a mildly-soft
  smaller one at 0.099, so a ratio cut would accept the worse grain. Hence
  **per-camera, different gates** (basic: absolute edge_width; zoom: size), not
  one universal metric.

**Projected impact** (recomputing QC on the two test runs, basic edge>7 /
zoom min_ecd>30 / contrast<0.70): **+66 sharp basic grains recovered**
(P_17 basic 277→339, P_01 basic 33→37) that the 5.25 cut had wrongly rejected,
and the **zoom composition corrected** — it had been accepting the 3 unusably
tiny grains and rejecting the usable large ones; the size gate flips that
(P_01 zoom recovers `z_0035950`). Total accepted 313→380.

Implementation: make both `defocus_edge_width_px` and `min_ecd_px` per-camera in
config + `qc.py` (dict per camera, fall back to the global scalar when a camera
is unlisted). Numbers above are on placeholder calibration; the *basic ~7* /
*zoom size-gate* structure is calibration-independent (both are pixel cuts), but
re-confirm the exact values once real µm/px lands.

## Low curvature_entropy flags fiber/organic debris — proposed `flag_debris` (open)

`curvature_entropy` (as implemented) is **inverse to intuition**: it is *high*
for smooth/round grains and *low* for angular, straight-edged, or filamentary
boundaries — a mostly-straight boundary concentrates curvature near zero, so its
distribution is peaked → low entropy. Across accepted grains it correlates
positively with solidity/circularity/convexity (r≈+0.47 each) and is essentially
**size-independent** (r≈+0.01 with ecd). Treat it as a boundary-*regularity*
score, not a roughness score — the name misleads.

**Debris signal (2026-08-05 labeling).** The 10 lowest-`curvature_entropy`
objects across P_01_cs_003 + P_17_cs_002 were *all* non-grains — thin curved
fibers/hairs, overlapping strands, and a frame-edge fragment (operator
confirmed: "all are bad"). Real grains do not appear until CE ≈ 0.63; the only
*accepted* grains below that are two genuine debris escapes (CE 0.252 & 0.317,
ecd 10–12) that slip past QC because the size/sliver flags happen not to fire.

**Border confound — the guard that matters.** Low CE is *also* produced by
**border-clipped real grains**: a straight frame-cut edge looks like a straight
fiber to the curvature distribution. In the CE∈[0.30,0.66] band (90 grains),
**42 are border-clips** (real grains, already owned by `flag_border`) and 48 are
non-border debris candidates (44 of them fiber/tiny: AR≥3 or ecd<20). So a raw
CE cut mislabels border-clipped grains as debris.

**Proposed rule:** `flag_debris = curvature_entropy < 0.5 AND NOT flag_border`,
**non-disqualifying by default** (like `flag_possible_agglomerate`) — visible in
per-frame counts, doesn't change acceptance until added to `disqualifying_flags`.
Projected (both test runs): removes exactly the 2 accepted debris escapes
(1/sample, zero real-grain false positives), gives ~43 already-rejected fibers an
honest reason, and the `NOT flag_border` guard keeps 32 border-clipped real
grains (30 P_17 / 2 P_01) from being mislabeled. CE only exists for grains with a
polygon; pure-shape, so calibration-independent. Optionally tighten toward fibers
with `AND (aspect_ratio high OR ecd small)`.

## Processing time is disk-bound, not compute-bound (performance)

Profiled 2026-08-06 on the P_01_cs / P_17_cs test runs (frames on an external
USB drive, `n_jobs = -1` on a 14-core machine).

**#1 — Frame I/O dominates (~80–85% of wall time).** Reading a 2048×2040 BMP is
**~165 ms/frame from the USB drive vs ~1 ms/frame from local SSD (128×)**. In a
serial profile `imread` is 9.7 s of 23.2 s (42%); parallelized it's worse in
share because all 14 cores sit idle waiting on the one shared USB bus — which is
why parallel detection (0.21 s/frame) was barely faster than serial
(0.41 s/frame). Compute floor with I/O removed is ~0.24 s/frame → P_17 detection
would be **~8–10 s instead of 102 s**. **Fix: stage the run to local disk (one
bulk sequential copy) before processing.** Amortizes across detect + overview +
report, which each independently re-read every frame off the drive.

**#2 — The blank frame is re-read once per frame.** 112 `imread` calls for 56
frames — the single `_{cam}_back.bmp` is re-decoded once per frame of that
camera. Fix: `functools.lru_cache` on `_read_grayscale` by path (blank read once
per worker).

**#3 — Overview rendering (only with `--overview`): ~1.8 s/frame and serial.**
Dominated the P_17 run (347 × 1.8 ≈ 10 min vs 102 s for detection). It runs in
the main process (not parallelized) and re-reads every source frame from the
drive. Fix: parallelize the pass (reuse the loky pool) and read from the local
stage; a PIL raster path would beat matplotlib full-res render.

**#4 — `detect_objects` morphology: ~32% of the compute floor.** Full-resolution
scipy `binary_fill_holes` / erosion / dilation (2.6 s in `binary_erosion2`) +
`distance_transform_edt` (2.0 s) on 2048×2040 arrays. Secondary; only visible
after I/O is fixed. Could crop morphology to object bounding boxes.

Bottom line: disk-bound. The one change that matters is not reading frames off
the USB drive during compute; everything else is secondary.

## Mineral discrimination (quartz/feldspar/mica) — OUT OF SCOPE, noted

Not a main project goal; recorded because it came up and there is a partial
signal. `grain-morph` sees only backlit **silhouettes** (outline + interior
opacity), so prospects differ sharply by mineral:

- **Mica — tractable by shape.** Platy habit → tumbling flakes present as thin,
  high-aspect, low-sphericity outlines (classic CAMSIZER b/l discriminator).
  Existing columns suffice: `aspect_ratio`, `wadell_sphericity`,
  `feret_min/feret_max` (thinness), plus interior translucency via `contrast`
  (thin mica transmits more → brighter core). **NB:** `flag_sliver`
  (aspect_ratio > 3) removes exactly these grains, so any mica analysis must
  include flagged grains, not the accepted set.
- **Quartz vs feldspar — unreliable by silhouette alone.** Both are equant,
  near-identical outlines. The only morphological handle is cleavage: feldspar's
  two ~90° cleavages tend toward blocky, straight-faceted, near-90°-cornered
  shapes; quartz (conchoidal, no cleavage) is more irregular. So a targeted
  **`frac_right_angle` + `straight_edge_fraction`** feature (not generic
  angularity) is the best shot — but statistical at best, and erased by
  rounding. Reliable ID needs another modality (Na-cobaltinitrite staining,
  SEM-EDS, Raman, QEMSCAN, or optical) or labeled training grains.

**Exploratory finding (2026-08-06, P_01+P_17, 656 grains, unlabeled):**
`wadell_sphericity` is **bimodal** — main equant mode ~0.6 + a distinct
low-sphericity mode ~0.08–0.12 (valley ~0.2); corroborated by an `aspect_ratio`
secondary bump (~4–6) and a `thinness` shoulder (~0.1–0.25). A real
elongated/platy sub-population exists — **but it is confounded with the organic
fiber debris** (also thin, low-sphericity); the low-`curvature_entropy` signal
(see `flag_debris` note) would be the natural mica-vs-fiber separator. Nothing
here is confirmable without mineralogical ground truth.

## Deployment: identity is in the filename, never the folder structure

Operational constraint for real deployments: input **images and `.xle`
(X-Plorer) exports are not reliably organized one-folder-per-sample.** A folder
may mix samples/cameras/runs, and a single sample may be split across folders.
**The reliable identifier is always the file name** (sample tokens, camera code
`b`/`z`, and either a frame index or the `back` blank token).

**Current code already honors this — keep it that way.** `io.discover_frames`
uses `root.rglob("*")` (fully recursive, folder-flattening) and derives all
identity from the filename via `cfg.filename.sample_regex` / `camera_map`; blanks
are keyed by parsed `(sample_id, camera)`, not by directory. The committed
`tests/fixtures/real/` (P_01 and P_17 mixed in one folder) exercises exactly
this. **Do not add any folder-based grouping or per-sample-directory assumption**;
`filename.sample_regex` is the single source of identity truth — update it (and
`camera_map`) if a deployment's naming differs.

**Latent risk to fix before wide deployment — blank pairing ignores run.** Blanks
are keyed only on `(sample_id, camera)` (`io.py`), and the current filename schema
doesn't parse a *run* identifier (the `run` column is always null). If a discovery
root contains **multiple runs of the same sample+camera**, each with its own
`_back` blank, the blanks collide in the dict and the one kept is
filesystem-`rglob`-order-dependent — so a frame could be flat-fielded against the
wrong run's blank. Fix: capture `run` in `sample_regex`, include it in both the
`ParsedName` identity and the blank-pairing key (fall back to `(sample, camera)`
when a run token is absent), and — for the same reason — consider making blank
pairing prefer a same-run blank. Same rule applies to `.xle` ingestion: parse
identity (and run) from the filename, never the folder.
