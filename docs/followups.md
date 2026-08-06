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
