# `tests/fixtures/real/` — provenance

Real CAMSIZER X2 frames, anti-aliased-downsampled and committed so
`tests/test_integration_real.py` can exercise the pipeline against real
image content (real noise, real illumination gradients, real grain
silhouettes) without shipping multi-megabyte source BMPs in the repo.

## How these were generated

```bash
grain-morph make-fixtures dev_data/images_P_17_cs tests/fixtures/real --factor 4
grain-morph make-fixtures dev_data/images_P_01_cs tests/fixtures/real --factor 4
```

`make-fixtures` anti-aliased-downsamples every image in `SRC_DIR` by
`--factor` (`skimage.transform.downscale_local_mean`, uint8, filenames
preserved) into `DEST_DIR`. Source directories are gitignored dev data
(`dev_data/`, never committed); only the downsampled output below is
committed.

Both commands above wrote all 10 frames from each source sample (4 Basic +
4 Zoom data frames + `_b_back`/`_z_back` blanks) into `tests/fixtures/real/`
— 20 files total, each ~260 KB (`du -h tests/fixtures/real/*`). That full
set was then **pruned** to the 11 files below, kept because they are the
frames that actually detect grains at `--factor 4` (verified by running
`grain-morph detect` over the full 20-file set and inspecting
`manifest.parquet`'s `status`/`n_objects` columns — see "Frames dropped and
why" below).

## Source paths

- `dev_data/images_P_01_cs/` — sample `P_01_cs_003`
- `dev_data/images_P_17_cs/` — sample `P_17_cs_002`

(Both gitignored; full-resolution dev-only data, never committed.)

## Downsample factor

`--factor 4` for every file. Each source BMP is ~4.0 MB; at factor 4 (16x
fewer pixels) each downsampled BMP is ~260 KB, comfortably under the ~300 KB
budget for committed fixtures.

**Any config used against these fixtures must scale `um_per_px` by the same
factor (4x)** relative to the real per-camera calibration for
`images_P_01_cs`/`images_P_17_cs` — downsampling by N increases the
physical size one pixel represents by N, so a pixel in these fixtures
covers 4x the micrometers a pixel in the source frames does. (The
integration test does not attempt to reproduce the real calibration; it
uses arbitrary positive placeholders — see
`tests/test_integration_real.py` — since it only checks that the pipeline
*runs* on real image content, not that its output is scientifically
accurate.)

## Frame indices kept

11 files, ≤2 Basic + ≤2 Zoom data frames per sample, plus both blanks per
sample:

| Sample | Camera | Kept data frames | Blank |
|---|---|---|---|
| `P_01_cs_003` | Basic (`b`) | `0001189`, `0001684` | `P_01_cs_003_b_back.bmp` |
| `P_01_cs_003` | Zoom (`z`) | `0035950` | `P_01_cs_003_z_back.bmp` |
| `P_17_cs_002` | Basic (`b`) | `0000938`, `0001038` | `P_17_cs_002_b_back.bmp` |
| `P_17_cs_002` | Zoom (`z`) | `0010022`, `0010212` | `P_17_cs_002_z_back.bmp` |

Only one Zoom frame survived pruning for `P_01_cs_003` (see below) —
`make-fixtures`/the brief's "≤2" budget allows fewer than 2 per camera when
fewer than 2 of a sample's frames actually detect grains at this
downsample factor.

Confirmed via `grain-morph detect tests/fixtures/real out --config
<calibration.yaml> --jobs 1` (calibration:
`{basic: 20.0, zoom: 8.0}`, arbitrary positive placeholders): all 7 kept
data frames report `status="ok"` with `n_objects >= 1` in `manifest.
parquet` (`P_17_cs_002_b_0001038` alone has 4), for 10 detected grains
total across the pruned fixture set.

## Frames dropped and why

Of the original 20-file `make-fixtures` output, 9 data frames were dropped:

- `P_01_cs_003_b_0002073.bmp`, `P_01_cs_003_b_0002600.bmp` — redundant
  (already had 2 kept Basic frames for this sample).
- `P_01_cs_003_z_0036702.bmp`, `P_01_cs_003_z_0037023.bmp`,
  `P_01_cs_003_z_0037320.bmp` — each raised `ValueError: math domain error`
  in `grain_morph.measure.measure_polygon` (`math.sqrt(area_um2 /
  math.pi)` on an apparently negative `poly.area`), recorded as
  `status="error"` in the full-set `manifest.parquet`/`errors.parquet`.
  Likely a self-intersecting contour introduced by the downsampled image's
  coarser edges; **not fixed here** — out of scope for this task, which is
  about picking working fixtures, not pipeline correctness. Worth a
  follow-up issue.
- `P_17_cs_002_b_0000889.bmp`, `P_17_cs_002_z_0010119.bmp`,
  `P_17_cs_002_z_0010664.bmp` — same `math domain error`, same cause.
- `P_17_cs_002_b_0000987.bmp` — redundant (already had 2 kept Basic frames
  for this sample; `0000938`/`0001038` were preferred for their grain-count
  diversity, 1 and 4 objects respectively).

None of the 11 kept files were empty (0 objects) at factor 4; every one
that reports `status="ok"` also reports `n_objects >= 1`.
