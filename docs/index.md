# grain-morph documentation

> **Status: work in progress.** This set is being written and revised; see each
> doc's banner. Pipeline behavior is authoritative in the code
> (`src/grain_morph/`) and defaults in `configs/default.yaml`.

Classical-CV morphometry + QC for backlit silhouette grain images (CAMSIZER X2
and similar). Given a directory of frames, it delineates grains, measures size
and shape, flags (never silently drops) unreliable ones, and produces per-sample
and per-frame summaries.

## Pipeline at a glance

```
frames ─▶ flat-field ─▶ detect / delineate ─▶ measure ─▶ QC metrics + flags ─▶ [persist per-grain table]
  (Stage 1: process_frame, run in parallel per frame — records everything, enforces nothing)

per-grain table ─▶ recompute qc_pass gate ─▶ summaries (summary / per_frame / rejection_by_ecd / accepted)
  (Stage 2: aggregate_run — the QC gate is applied here, re-gateable without re-detecting)
```

**Two-stage by design:** Stage 1 (detect) is expensive and records raw metrics +
flags for every object without deciding accept/reject. Stage 2 (aggregate)
applies the gate from `cfg.qc.disqualifying_flags` — so exploring a stricter or
looser QC gate is a cheap re-run of aggregation, not a re-detection.

## Document map

| Doc | Covers |
|---|---|
| [`delineation.md`](delineation.md) | flat-field → threshold → label → **subpixel boundary** → repair |
| [`morphometrics.md`](morphometrics.md) | every per-grain calculation + custom/imported provenance |
| [`qc.md`](qc.md) | focus/contrast metrics, flags, the accept/reject gate |
| [`aggregation.md`](aggregation.md) | per-(sample,camera) & per-frame summaries, ECD rejection binning |
| [`followups.md`](followups.md) | open work: per-camera defocus/size gates, `flag_debris`, performance |

## Commands

```bash
grain-morph detect  FRAMES_DIR OUT [--config c.yaml] [--jobs N] [--overview]
grain-morph aggregate OUT/grains AGG_OUT [--config c.yaml]
grain-morph report  OUT/grains FRAMES_DIR REP_OUT [--config c.yaml]
grain-morph overlay OUT FRAMES_DIR OVL_OUT --frames id1,id2 [--config c.yaml]
```

`detect --overview` renders QC-colored overlay PNGs for every frame with a
detection into `OUT/overviews/`.

## Conventions (repeated in each doc)

- **Two geometries:** a **subpixel polygon** (shapely, from marching-squares) and
  a **raster mask** (pixel labels). Most metrics use the polygon; ellipse-fit and
  QC geometry use the raster. Where both exist, both are persisted (see
  `morphometrics.md` → Dual geometry).
- **Units:** pixels natively; `*_um`/`*_um2` multiply by `um_per_px`, which is
  currently a **placeholder** — physical sizes are not yet real; all pixel/ratio/
  shape metrics are calibration-independent.
- **Build philosophy:** assemble from maintained packages (shapely, scikit-image,
  scipy, pyefd, wadell_rs, pandas); write glue/config/CLI/tests, not algorithms.
  Each doc marks **custom vs imported**; the honest accounting is that heavy
  algorithms are imported and most "custom" is standard formulas + orchestration,
  with a short list of genuinely bespoke metrics (focus metrics,
  `curvature_entropy`) called out.

## Known issues (see `morphometrics.md` / `followups.md`)

- `curvature_entropy` semantics are inverted vs its docstring and use an
  outlier-sensitive per-grain histogram range (a *regularity* score, not
  roughness); a verifiable fixed-domain redefinition is proposed.
- Non-periodic gradient at the boundary seam (minor).
- QC flags threshold on raster `qc_*` geometry, not the reported polygon columns.
- Defocus cut isn't camera-scale-invariant (per-camera gates proposed).
- Processing is disk-bound (external-drive frame I/O), not compute-bound.
