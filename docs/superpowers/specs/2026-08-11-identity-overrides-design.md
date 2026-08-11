# Design — `detect` identity overrides (`--sample-id` / `--camera`)

Status: approved (self-review, 2026-08-11). Branch: `feat/identity-overrides`.

## Motivation

Today a frame's identity — `sample_id`, `camera`, blank-vs-data, frame index —
is parsed **solely** from its filename via `filename.sample_regex`
(`io.parse_name`). That's the right default for CAMSIZER exports, but it blocks
the common single-run layout: *all images for one sample sit in one directory*
and the filenames either don't encode the sample/camera or shouldn't dictate it.

Add optional overrides so the operator can supply `sample_id` and/or `camera`
directly. "Either or both" — override one and keep the other from the filename,
or override both and drop the filename convention entirely.

## Scope

- New CLI options on `detect`: `--sample-id`, `--camera`.
- Threaded core params (keyword-only, `None` default) through
  `run_detect → discover_frames → parse_name`.
- New config key `filename.blank_regex` (blank marker for the override path).
- Docs: README (Quickstart + config table), tutorial, delineation reference.

Non-goals: overrides for `aggregate`/`report`/`overlay` (they read identity from
the already-written grain table); a `--run` override; per-file identity maps.

## Behavior — override-first, regex-second

`parse_name(path, cfg, *, sample_id=None, camera=None)`:

1. `stem = Path(path).stem`; `match = re.match(sample_regex, stem)`;
   `groups = match.groupdict() if match else {}`.
2. `resolved_sample = sample_id if sample_id is not None else groups.get("sample")`.
3. `resolved_camera`: the `camera` override if given; else
   `camera_map[groups["cam"]]` when a `cam` group was captured; else `None`.
4. Frame/blank/run:
   - **regex matched** → unchanged: `run = groups.get("run")`,
     `is_blank = groups.get("frame") is None`, `frame_index = int(frame)` or `None`.
   - **regex did not match** (fallback, only reachable when overrides supply the
     missing identity) → `run = None`,
     `is_blank = re.search(blank_regex, stem) is not None`,
     `frame_index =` trailing digits of `stem` (`\d+$`) or `None`.
5. A file is a processable frame **iff both `resolved_sample` and
   `resolved_camera` are non-`None`** — else return `None` (skipped). With no
   overrides this reduces exactly to today's behavior (skip on no match).

This single rule covers every "either or both" combination:

| filenames | `--sample-id` | `--camera` | result |
|---|---|---|---|
| full tokens | — | — | today's behavior (unchanged) |
| full tokens | given | — | sample forced, camera from filename |
| full tokens | — | given | camera forced, sample from filename |
| full tokens | given | given | both forced (dir collapses to one sample/cam) |
| no tokens (won't match) | given | given | fallback: blank via `blank_regex`, index via trailing digits |
| no tokens | given only | — | camera unresolved → **skipped** (need `--camera` too) |
| custom regex, `cam`+`frame`, no `sample` group | given | — | matches; sample from override, camera from filename |

Overrides are **authoritative for every discovered frame**: pointing `detect` at
a directory that actually mixes samples with `--sample-id X` collapses them all
to `X`. Documented, not guarded.

## New config: `filename.blank_regex`

Consulted **only** in the fallback branch (regex-unmatched, override-supplied
identity) to tell a blank/background frame from a data frame. Default
`'(?i)(?:^|[_-])back$'` — matches `back`, `foo_back`, `foo-back`
case-insensitively, aligned with the CAMSIZER `back` convention. When the
`sample_regex` matches, blank detection still comes from its `frame`/`back`
alternation, so `blank_regex` and the regex are two blank sources that apply in
**mutually exclusive modes**; the docstring says so.

## Validation

`--camera` must be one of `cfg.filename.camera_map` values (e.g. `basic`/`zoom`,
the calibration keys). Rejected before discovery — `typer.BadParameter` at the
CLI, `ValueError` in `run_detect` for the library path. Calibration is still
enforced downstream by `_validate_calibration`, so overriding to an uncalibrated
camera fails loudly exactly as an uncalibrated filename-parsed camera does today.

## Downstream — no changes

`_validate_calibration`, hive-partitioning, and every grain/contour row read
`spec.parsed.{sample_id,camera}`, which the overrides now populate. Blank pairing
still keys on `(sample_id, run, camera)`; with overrides `run` is `None`, so it
reduces to `(sample_id, camera)`.

## Tests (TDD, real behavior)

- `test_io_parse.py`: override sample only (regex matches); override camera only;
  override both on a non-matching structureless stem → data frame with
  trailing-digit index; fallback blank via `blank_regex`; invalid camera raises;
  no-override path unchanged; custom sample-less regex + `--sample-id`.
- `test_io_discover.py`: structureless single dir + both overrides → all frames
  one sample/camera; a `back.bmp` pairs as the blank.
- `test_pipeline.py` + `test_cli.py`: `run_detect` / `grain-morph detect
  --sample-id --camera` end-to-end on structureless synthetic frames → grain rows
  carry the overridden identity; invalid `--camera` errors.

## Docs

- README: `detect` options in Quickstart; `filename.blank_regex` row + an
  identity-override note in the config section.
- `docs/tutorial.md`: a short "all images in one directory" callout.
- `docs/delineation.md` (identity subsection): document the override path and the
  two blank-detection modes.
