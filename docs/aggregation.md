# Aggregation — reference

> **Status: work in progress, subject to revision.** Companion to
> `morphometrics.md` / `delineation.md` / `qc.md`.

Stage 2: turn the per-grain table into per-group summaries and apply the QC
gate. Source: `src/grain_morph/aggregate.py`, one entry point `aggregate_run`.

## Where the QC gate actually lives

`aggregate_run(grains, cfg)` **recomputes** `qc_pass` from
`cfg.qc.disqualifying_flags` (`_recompute_qc_pass`) rather than trusting any
stored `qc_pass` — so the accept/reject decision is made *here*, at aggregation
time. Re-running with a different `cfg` re-gates without re-detecting (see
`qc.md`). All size/shape statistics below are computed over **accepted grains
only** — the point of the gate is to keep defocused/border/sliver artifacts out
of the reported distribution.

## The four tables (`aggregate_run` return dict)

| Key | One row per | Contents |
|---|---|---|
| `summary` | `(sample_id, camera)` | counts, per-flag counts, size percentiles, shape mean/SD |
| `per_frame` | `(sample_id, camera, frame_id)` | same columns as `summary`, per frame |
| `rejection_by_ecd` | `(sample_id, camera, ecd_bin)` | `n`, `n_rejected`, `rejection_rate` |
| `accepted` | — | the row subset of `grains` that passed the recomputed `qc_pass` |

`summary` and `per_frame` share one builder (`_apply_group_summary` +
`_summarize_group`), differing only in the grouping key — so their column
vocabulary is identical.

### `summary` / `per_frame` columns

- **Counts:** `n_total`, `n_accepted`, `n_rejected` (int).
- **Per-criterion counts:** `n_flag_*` for every `flag_*` present — grains where
  that flag is *set*. **Not mutually exclusive**, so they can sum to more than
  `n_rejected` (a grain can trip several). Includes `n_flag_possible_agglomerate`
  even though it isn't disqualifying — a free diagnostic.
- **Size percentiles** (over accepted): D10/D50/D90 of `ecd_um` and
  `feret_min_um` (`_SIZE_COLUMNS`), percent-finer convention.
- **Shape mean/SD** (over accepted): `{descriptor}_mean` / `{descriptor}_sd` for
  each of `aspect_ratio, solidity, convexity, circularity, extent, eccentricity,
  wadell_roundness, wadell_sphericity, curvature_entropy` present in the input.

### `rejection_by_ecd`

Rejection rate binned by ECD so a **size-dependent** QC gate is visible rather
than hidden. Fixed **Wentworth (1922)** sand-class edges (µm):
`62.5, 125, 250, 500, 1000, 2000, 4000` (+ a `0–62.5` and a `4000+` catch-all),
half-open `[lo, hi)`. Fixed (not data-derived) so bins are comparable across
runs. Columns: `sample_id, camera, ecd_bin, n, n_rejected, rejection_rate`.

## CLI

`grain-morph aggregate GRAINS OUT [--config …]` writes every returned table to
`OUT/{name}.{format}` (the CLI loops over the dict, so `per_frame` is written
automatically). Zero-grain runs echo a message and write nothing.

## Provenance

Imported: `pandas` (`groupby`/`cut`/`agg`), `numpy` (`percentile`). Custom: the
`qc_pass` recomputation, the Wentworth bin edges, the per-group summary
assembly, and the count-column dtype handling. No morphometric arithmetic
happens here — aggregation only *summarizes* columns produced upstream.

## Gotchas

- **Size percentiles are `NaN` for a group with zero accepted grains**
  (`_percentile_stats` on an empty series) — expected, not an error.
- **`_um` size percentiles are placeholder-scaled** until real `um_per_px`
  lands; the `n_*` counts and shape stats are unaffected.
- The gate uses `cfg.qc.disqualifying_flags`; changing it changes `n_accepted`
  and every accepted-only statistic — that is the intended re-gating knob.
