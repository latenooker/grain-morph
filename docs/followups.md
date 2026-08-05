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
