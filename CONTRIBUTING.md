# Contributing to grain-morph

Thanks for working on this. This guide covers how to set up, the conventions the
codebase follows, and the workflow for landing a change.

## Setup

```bash
uv sync --extra dev          # or: pip install -e ".[dev]"
```

Everything runs through the dev extras: `pytest`, `ruff`, `mypy`.

## The three checks (run before every PR)

```bash
uv run pytest                 # tests
uv run ruff check src tests   # lint
uv run mypy src               # types
```

The suite must be **green and pristine** — no new warnings in the output.
`tests/test_realdata_optin.py` is marked `realdata` and needs full-resolution
frames on an external drive; it **skips automatically** when the drive isn't
present, so a plain `pytest` is safe anywhere. Use `pytest -m "not realdata"` to
exclude it explicitly.

## Design philosophy (read this first)

**Assemble from maintained packages; write glue, config, CLI, and tests — not
algorithms.** The heavy lifting is delegated (shapely, scikit-image, scipy,
pyefd, wadell_rs, pandas); custom code is orchestration plus a short list of
genuinely bespoke metrics. Before writing a new algorithm, check whether a
maintained library already does it — and if you compute a quantity two ways,
keep them on the *same* geometry (see the Feret note in
[`docs/morphometrics.md`](docs/morphometrics.md)). The per-doc **custom vs
imported** tables exist to keep this honest; update them when you touch a
calculation.

## Code conventions

- `from __future__ import annotations` at the top of **every** module.
- Modern type hints (`str | None`, `list[str]`), not `Optional`/`List`.
- **Google-style docstrings** (Args / Returns / Raises) on every public function
  and class — this codebase documents *why*, not just what.
- Private symbols prefixed `_`.
- No magic numbers in source — every threshold/kernel/parameter lives in
  `configs/default.yaml` and is read through `Config`.
- PEP 8, `snake_case` functions, `UPPER_CASE` module constants.

## Test-driven, small commits

- Write the failing test first, watch it fail, implement the minimum to pass.
- Tests must verify **real behavior**, not mocks — the pipeline's tests run real
  detection/measurement on synthetic and downsampled-real frames.
- Keep commits focused; imperative subject line explaining *why*.

## Workflow

- Branch off `main` (never commit straight to `main`; never use `master`).
- Open a PR against `main`; make sure the three checks pass on the merged result.
- AI-assisted commits carry a `Co-Authored-By:` trailer per the project's
  attribution convention.

## Documentation

- Reference docs live in [`docs/`](docs/) and are currently marked
  **work-in-progress**. If you change a calculation, a flag, a default, or the
  output schema, update the matching doc (`morphometrics`, `qc`, `delineation`,
  `aggregation`) and the [`README`](README.md) config table in the same PR.
- Deferred work and known issues are tracked in
  [`docs/followups.md`](docs/followups.md) — add to it rather than losing a
  finding. Notable open items: `curvature_entropy` is semantically inverted vs
  its docstring (a regularity, not roughness, score) and slated for a verifiable
  redefinition; the defocus cut isn't camera-scale-invariant; processing is
  disk-bound. Check there before assuming a rough edge is a bug.

## Where things are

| Path | What |
|---|---|
| `src/grain_morph/` | the package (detect, flatfield, measure, qc, aggregate, cli, …) |
| `configs/default.yaml` | every tunable parameter |
| `tests/` | test suite; `tests/fixtures/real/` is committed runnable data |
| `docs/` | reference documentation + `tutorial.md` + `followups.md` |
| `dev_data/` | full-res dev frames — **gitignored**, never committed |
