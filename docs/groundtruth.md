# Groundtruthing — reference

> **Status: work in progress.** Behavior is authoritative in
> `src/grain_morph/groundtruth.py`; defaults in `configs/default.yaml`
> (`groundtruth:`).

`grain-morph groundtruth` is a lightweight labeling GUI for **validating and
tuning the QC gate** by hand. It samples grains from a completed `detect` run,
shows each one zoomed with its polygon mask, and records a class per grain by
keystroke into a resumable `groundtruth.csv`. Nothing is re-detected and nothing
is written back into the `detect` output — labels live only in the CSV.

## Run it

```bash
grain-morph groundtruth OUT --n-frames 8 --n-grains 60 --seed 0 [--config c.yaml]
```

`OUT` is a completed `detect` output directory (it reads `OUT/grains/` and
`OUT/contours/`). Labels are written to `OUT/groundtruth.csv` (override with
`--out`). If the frames have moved since `detect` (the grain rows store absolute
`frame_path`s), point `--frames-dir` at their new location.

| Option | Meaning |
|---|---|
| `--n-frames` / `--frac-frames` | Size of the **frame pool** the grains are drawn from (default: all frames). |
| `--n-grains` / `--frac-grains` | Number / fraction of grains to label from that pool (default: all). |
| `--seed` | Makes the sample reproducible. |
| `--out` | Labels CSV path (default `OUT/groundtruth.csv`). |
| `--frames-dir` | Relocate source frames if the stored `frame_path`s are stale. |

## Keys

| Key | Action |
|---|---|
| `0` / `1` / `2` | Assign **exclude / include / special** and advance (classes are configurable). |
| `m` | Toggle the polygon-mask overlay. |
| `n` / `p` (or → / ←) | Next / previous grain without labeling. |
| `u` | Unset the current grain's label. |
| `i` | Reveal the **predicted** QC status (hidden by default to keep labels unbiased). |
| `q` / `esc` | Save and quit. |

Every keystroke autosaves, so a crash never loses labels and relaunching the
same command resumes at the first unlabeled grain.

## Sampling — Latin hypercube over the QC-driving metrics

The QC flags are thresholds on continuous per-grain metrics. To make hand labels
inform each cut, the sample is a space-filling **Latin hypercube** over exactly
those metrics (config `groundtruth.lhs_axes`, default):

| Axis (grain column) | QC flag it drives |
|---|---|
| `ecd_px` | `flag_too_small` (also the size axis) |
| `edge_width_px` | `flag_defocus` |
| `contrast` | `flag_defocus` |
| `qc_aspect_ratio` | `flag_sliver` |
| `qc_solidity` | `flag_possible_agglomerate` |

Each axis is empirical-quantile-ranked to `[0, 1)` (so its raw scale/outliers
don't matter), a `scipy.stats.qmc.LatinHypercube` design of the target size is
drawn, and each design point is matched to the nearest unused grain. The result
spans **both sides of every threshold** — e.g. grains with `edge_width_px` just
below and just above `defocus_edge_width_px` — which is what lets the labels
inform per-camera gate tuning (see [`followups.md`](followups.md)).

Two QC flags have **no continuous axis**: `flag_border` (`touches_border`) and
`flag_no_polygon` (`has_polygon`). A minimum number of each
(`groundtruth.reserved`, default `border: 2, no_polygon: 1`) is reserved from the
pool before the hypercube fills the rest; grains missing any continuous axis
value are reached only through that reservation. The realized per-axis and
reserved coverage is logged when sampling runs.

## Output schema (`groundtruth.csv`)

Keyed by `grain_uid`, so it joins straight back to the grains table.

| Column | Meaning |
|---|---|
| `grain_uid` | `"{frame_id}:{label}"` — join key to `grains`. |
| `label`, `label_name` | The assigned class integer + name (e.g. `1`, `include`). |
| `qc_status` | The gate's **predicted** outcome at label time (`accept`/`reject`). |
| `flags` | Comma-joined active QC flags for the grain. |
| `ecd_px`, `sample_id`, `camera`, `frame_id` | Identity/size for slicing. |
| `labeled_at` | UTC ISO timestamp. |

## Scoring the gate against labels

Because both tables are keyed by `grain_uid`, comparing the hand `label` to the
predicted `qc_status` is a join:

```python
import pandas as pd
grains = pd.read_parquet("OUT/grains")          # predicted flags per grain
truth  = pd.read_csv("OUT/groundtruth.csv")     # hand labels
joined = truth.merge(grains, on="grain_uid", suffixes=("", "_g"))
# e.g. confusion of predicted accept/reject vs include(1)/exclude(0)
pd.crosstab(joined["qc_status"], joined["label_name"])
```

That crosstab (and slicing it by `camera` or by `edge_width_px` band) is the
evidence behind the per-camera defocus/size gate changes proposed in
`followups.md`.

## Config (`groundtruth:` in `configs/default.yaml`)

| Key | Default | Description |
|---|---|---|
| `classes` | `0=exclude, 1=include, 2=special` | Keystroke-assignable label classes. |
| `lhs_axes` | the five metrics above | Continuous metrics the hypercube spans. |
| `reserved` | `{border: 2, no_polygon: 1}` | Minimum grains reserved for the boolean-flag QC cases. |
| `crop_pad_px` | `24` | Padding around a grain's polygon bbox in its zoomed view. |
| `mask_alpha` | `0.35` | Opacity of the translucent polygon-mask fill. |
| `show_predicted_status` | `false` | Whether predicted QC status shows by default (toggle live with `i`). |

## Custom-vs-imported

| Piece | Implementation |
|---|---|
| Sample design | **imported** — `scipy.stats.qmc.LatinHypercube` |
| Quantile ranking, nearest-point matching, reservation | **custom glue** (standard operations) |
| GUI | **imported** — matplotlib (image display + key events) |
| Mask overlay geometry | **imported** — shapely polygon from stored WKT |

## Requirements / notes

- Needs an **interactive matplotlib backend** (a desktop session: macOS, TkAgg,
  or QtAgg). Under a headless/file-only backend it fails with a clear message
  rather than a traceback — e.g. run `MPLBACKEND=QtAgg grain-morph groundtruth …`.
- The mask is the run's **subpixel contour** (`OUT/contours/`), so `detect` must
  have run with `output.save_contours` (the default).
