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
set was then **pruned** to the 11 files below, kept to keep the committed
fixture set small while still grain-bearing and diverse for the integration
test — not because any frame fails or crashes (see "Frames dropped and why"
below; as of commit `f9cfd6b` every frame in the full 20-file set runs
clean).

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

Only one Zoom frame was kept for `P_01_cs_003`, not because fewer than 2
were available (all 4 of that sample's Zoom candidates produce objects as
of `f9cfd6b` — see "Frames dropped and why" below) but because one was
already enough to exercise the Zoom camera in the integration test, and the
other 3 were dropped along with the rest of the size-driven pruning.
`make-fixtures`/the brief's "≤2" budget is a ceiling, not a floor — it
allows fewer than 2 per camera whenever fewer are needed.

Confirmed via `grain-morph detect tests/fixtures/real out --config
<calibration.yaml> --jobs 1` (calibration:
`{basic: 20.0, zoom: 8.0}`, arbitrary positive placeholders): all 7 kept
data frames report `status="ok"` with `n_objects >= 1` in `manifest.
parquet` (`P_17_cs_002_b_0001038` alone has 4), for 10 detected grains
total across the pruned fixture set.

## Frames dropped and why

Of the original 20-file `make-fixtures` output, 9 data frames were dropped
to keep the committed fixture set small (11 files) rather than because any
of them fail:

- `P_01_cs_003_b_0002073.bmp`, `P_01_cs_003_b_0002600.bmp` — redundant
  (already had 2 kept Basic frames for this sample).
- `P_01_cs_003_z_0036702.bmp`, `P_01_cs_003_z_0037023.bmp`,
  `P_01_cs_003_z_0037320.bmp`, `P_17_cs_002_b_0000889.bmp`,
  `P_17_cs_002_z_0010119.bmp`, `P_17_cs_002_z_0010664.bmp` — not needed to
  hit the "≤2 Basic + ≤2 Zoom data frames per sample" budget once the
  frames above were chosen. Earlier (pre-`f9cfd6b`) these six *did* raise
  `ValueError: math domain error` in `grain_morph.measure.measure_polygon`
  on the coarse contours downsampling produces (a self-intersecting
  exterior ring, or a mis-associated/oversized interior hole ring, driving
  `poly.area` to zero or negative) — an entire frame's grains lost to the
  errors table over one bad polygon. That failure mode is **fixed** as of
  commit `f9cfd6b` (`fix(detect,measure): repair invalid/self-intersecting
  grain polygons`): invalid polygons are repaired via `shapely.make_valid`
  (largest-area component kept), and any contour that's still
  unrecoverable becomes `polygon=None`/`contour_ok=False`, which
  `qc`/`measure` already handle by NaN-filling and setting
  `flag_no_polygon` rather than crashing (flag-never-drop). Verified by
  running `make-fixtures --factor 4` on the full, un-pruned
  `dev_data/images_P_01_cs`/`images_P_17_cs` and then `grain-morph detect`
  over all 16 resulting data frames: all 16 report `status="ok"`, zero
  rows in `errors.parquet` (the file isn't even written), and zero
  `math domain error`s — the objects that used to crash the frame now
  simply carry `flag_no_polygon=True` and NaN shape measures. These six
  frames stayed out of the committed set purely because they weren't
  needed once two Basic + two (or, for `P_01_cs_003` Zoom, one) working
  frames per camera were already picked.
- `P_17_cs_002_b_0000987.bmp` — redundant (already had 2 kept Basic frames
  for this sample; `0000938`/`0001038` were preferred for their grain-count
  diversity, 1 and 4 objects respectively).

None of the 11 kept files were empty (0 objects) at factor 4; every one
that reports `status="ok"` also reports `n_objects >= 1`.
