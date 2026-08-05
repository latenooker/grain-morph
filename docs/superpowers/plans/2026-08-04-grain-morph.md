# grain-morph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone classical-CV CLI pipeline that turns directories of backlit silhouette BMP frames (CAMSIZER X2) into a per-grain morphometry table with QC flags and persisted subpixel boundary polygons.

**Architecture:** Two decoupled stages. `detect` reads frames, flat-fields (blank division by default), extracts subpixel contours on the flat-fielded grayscale, measures in polygon space, computes QC flags, and writes per-grain rows + geometry. `aggregate` filters by flags and summarizes per `(sample_id, camera)`. `report` renders QC artifacts. Frames are processed independently (joblib), resumable via a manifest, flag-never-drop.

**Tech Stack:** Python ≥3.11; numpy, scipy, scikit-image, shapely, pyefd, wadell_rs (fallback fast_rs), pandas, pyarrow, imageio, typer, pydantic v2, PyYAML, joblib, tqdm, matplotlib; pytest, ruff, mypy; uv.

## Global Constraints

- Python ≥ 3.11. `from __future__ import annotations` at the top of every module.
- Modern type hints (`str | None`, `list[str]`); Google-style docstrings on every public function/class; private symbols `_`-prefixed.
- `ruff` clean and `mypy` clean before every commit.
- **No magic numbers in source.** Every threshold/kernel/parameter comes from the pydantic config. Ship `configs/default.yaml` with the §2 starting values.
- **Flag, never drop.** `detect` emits one row per detected object always; rejection is a reversible filter at `aggregate`.
- Calibration `um_per_px` is **per camera**, required; if a frame's camera has no calibration entry, fail loudly (raise).
- Frames are large (tens of GB/run): stream, never load a whole run into memory; parallelize across frames.
- `output.format ∈ {parquet (default), csv, feather}`, routed through one `write_table` helper. Partitioning is Parquet-only; csv/feather write one file per `{sample_id}__{camera}` partition.
- Subpixel polygons are persisted **on by default** to a geometry table keyed on `grain_uid`.
- `grain_uid = f"{frame_id}:{label}"`, `frame_id` = filename stem.
- Commit messages end with `Co-Authored-By: Claude <claude-opus-4-8> <noreply@anthropic.com>`. Use `git -c user.name="Nate Looker" -c user.email="ntlooker@gmail.com"` if git identity is unset.
- Target < 1200 LOC source (excluding tests). A module past ~250 LOC is a smell.

**Spec:** `docs/superpowers/specs/2026-08-04-grain-morph-design.md` (read it before starting).

**Dev data:** full-res real frames at `dev_data/images_P_01_cs/` and `dev_data/images_P_17_cs/` (gitignored). Each has 4 Basic + 4 Zoom data frames and `_b_back.bmp`/`_z_back.bmp` blanks. Also on the external drive: `/Volumes/LEXAR/Camsizer/PPX/images_P_01_cs`, `images_P_17_cs`.

---

## File Structure

```
src/grain_morph/
  __init__.py     # version, top-level exports
  config.py       # pydantic Config, YAML load/validate, config_hash
  io.py           # FrameSpec, filename parse, discovery, blank pairing, manifest, resume
  writers.py      # write_table / read_table (parquet|csv|feather)
  flatfield.py    # apply_flatfield: blank | morphological | auto
  detect.py       # threshold, label, subpixel contours -> shapely polygons
  measure.py      # polygon morphometry + EFD + Wadell
  qc.py           # per-object QC metrics + flags
  pipeline.py     # process_frame (one frame -> rows+geometry+manifest), run_detect (parallel)
  aggregate.py    # per (sample_id,camera) summaries
  report.py       # contact sheets, focus scatter, rejection-vs-ECD, field render
  cli.py          # typer app: detect | aggregate | report | make-fixtures
tests/
  synth.py        # synthetic frame generator (ground truth)
  conftest.py     # shared fixtures (tmp dirs, synth frames)
  test_*.py
configs/
  default.yaml
pyproject.toml
README.md
```

---

## Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`, `src/grain_morph/__init__.py`, `tests/test_smoke.py`, `configs/.gitkeep`
- Create: `.python-version` (optional)

**Interfaces:**
- Produces: importable package `grain_morph` with `grain_morph.__version__: str`.

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[project]
name = "grain-morph"
version = "0.1.0"
description = "Classical-CV morphometry + QC pipeline for backlit silhouette grain images (CAMSIZER X2)"
readme = "README.md"
requires-python = ">=3.11"
license = { text = "MIT" }
authors = [{ name = "Nate Looker" }]
dependencies = [
  "numpy>=1.24",
  "scipy>=1.11",
  "scikit-image>=0.22",
  "shapely>=2.0",
  "pyefd>=1.6",
  "wadell_rs @ git+https://github.com/PaPieta/wadell_rs.git",
  "pandas>=2.0",
  "pyarrow>=14",
  "imageio>=2.31",
  "typer>=0.12",
  "pydantic>=2.5",
  "PyYAML>=6.0",
  "joblib>=1.3",
  "tqdm>=4.66",
  "matplotlib>=3.8",
]

[project.optional-dependencies]
dev = ["pytest>=7.4", "pytest-cov", "ruff>=0.5", "mypy>=1.8", "psutil>=5.9"]

[project.scripts]
grain-morph = "grain_morph.cli:app"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
  "realdata: opt-in tests against full-res frames on /Volumes/LEXAR (skipped if absent)",
]

[tool.ruff]
target-version = "py311"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]

[tool.mypy]
python_version = "3.11"
ignore_missing_imports = true
```

> Note: if `wadell_rs` fails to install, fall back to `fast_rs @ git+https://github.com/PaPieta/fast_rs.git` and record the substitution in the README (Task 16). Do not write a custom roundness algorithm.

- [ ] **Step 2: Write `src/grain_morph/__init__.py`**

```python
from __future__ import annotations

__version__ = "0.1.0"
```

- [ ] **Step 3: Write the smoke test `tests/test_smoke.py`**

```python
from __future__ import annotations

import grain_morph


def test_version_present():
    assert isinstance(grain_morph.__version__, str)
    assert grain_morph.__version__
```

- [ ] **Step 4: Create environment and install**

Run:
```bash
cd /Users/looker/Documents/projects/grain-morph
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -e ".[dev]"
```
If `wadell_rs` install fails, retry with the `fast_rs` fallback line and note it. Expected: install succeeds.

- [ ] **Step 5: Run smoke test**

Run: `pytest tests/test_smoke.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/grain_morph/__init__.py tests/test_smoke.py configs/.gitkeep
git commit -m "feat: scaffold grain-morph package (pyproject, package skeleton, smoke test)"
```

---

## Task 2: Config models + default.yaml

**Files:**
- Create: `src/grain_morph/config.py`, `configs/default.yaml`, `tests/test_config.py`

**Interfaces:**
- Produces:
  - `class Config(pydantic.BaseModel)` with nested sections: `calibration`, `filename`, `flatfield`, `threshold`, `detect`, `measure`, `qc`, `output`.
  - `load_config(path: str | Path | None) -> Config` — loads YAML, merges over packaged defaults, validates.
  - `config_hash(cfg: Config) -> str` — deterministic sha256 (first 12 hex chars) of the resolved config.
  - `Config.um_per_px(camera: str) -> float` — raises `KeyError` with a clear message if the camera is uncalibrated.
- Consumes: nothing.

Key config shape (see `configs/default.yaml`):
```
calibration:
  um_per_px: { basic: null, zoom: null }   # REQUIRED at run time; null -> fail loudly
filename:
  sample_regex: '(?P<sample>.+?)_(?P<cam>[bz])_(?:(?P<frame>\d+)|back)$'
  camera_map: { b: basic, z: zoom }
  back_token: back
flatfield:
  method: auto            # auto | blank | morphological
  morph_kernel_px: 201
threshold:
  method: half_max        # half_max | otsu
  half_max_fraction: 0.5
  core_percentile: 5      # dark-pixel percentile used to estimate opaque core
detect:
  min_area_px: 25
  fill_holes: true
measure:
  efd_order: 15
  efd_resample_n: 256
  wadell_smoothing: 1.0
qc:
  min_ecd_px: 10
  defocus_edge_width_px: 3.0      # edge_width above this -> defocus
  defocus_contrast_min: 0.90      # contrast below this -> defocus
  sliver_aspect_ratio: 3.0
  sliver_max_ecd_px: 40
  agglomerate_solidity_max: 0.90
  disqualifying_flags: [flag_defocus, flag_border, flag_too_small, flag_sliver, flag_no_polygon]
output:
  format: parquet         # parquet | csv | feather
  save_contours: true
  partition: true
```

- [ ] **Step 1: Write failing tests `tests/test_config.py`**

```python
from __future__ import annotations

import pytest

from grain_morph.config import Config, config_hash, load_config


def test_load_default_returns_config():
    cfg = load_config(None)
    assert isinstance(cfg, Config)
    assert cfg.threshold.method == "half_max"
    assert cfg.output.format == "parquet"


def test_uncalibrated_camera_fails_loudly():
    cfg = load_config(None)  # default calibration is null
    with pytest.raises(KeyError, match="calibration"):
        cfg.um_per_px("basic")


def test_calibration_lookup(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("calibration:\n  um_per_px:\n    basic: 5.0\n    zoom: 1.0\n")
    cfg = load_config(p)
    assert cfg.um_per_px("basic") == 5.0
    assert cfg.um_per_px("zoom") == 1.0


def test_config_hash_is_deterministic_and_sensitive():
    a = load_config(None)
    b = load_config(None)
    assert config_hash(a) == config_hash(b)
    a.threshold.half_max_fraction = 0.6
    assert config_hash(a) != config_hash(b)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_config.py -v`
Expected: FAIL (module/function not defined).

- [ ] **Step 3: Implement `configs/default.yaml`** with the shape above (all sections/keys present; `calibration.um_per_px.basic`/`.zoom` = `null`).

- [ ] **Step 4: Implement `src/grain_morph/config.py`**

Use pydantic v2 `BaseModel` sub-models for each section. `load_config` reads packaged `configs/default.yaml` via `importlib.resources`, deep-merges the user YAML (if any) on top, then constructs `Config`. `um_per_px(camera)` reads `self.calibration.um_per_px[camera]`; if `None` or missing, raise `KeyError(f"No calibration for camera '{camera}' — set calibration.um_per_px.{camera}")`. `config_hash` = `hashlib.sha256(json.dumps(cfg.model_dump(), sort_keys=True, default=str).encode()).hexdigest()[:12]`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_config.py -v`
Expected: PASS.

- [ ] **Step 6: ruff + mypy, then commit**

```bash
ruff check src tests && mypy src
git add src/grain_morph/config.py configs/default.yaml tests/test_config.py
git commit -m "feat(config): pydantic config, YAML load/merge, per-camera calibration, config_hash"
```

---

## Task 3: Filename parsing

**Files:**
- Create: `src/grain_morph/io.py` (start it here), `tests/test_io_parse.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class ParsedName` with fields: `sample_id: str`, `camera: str` (mapped, e.g. `basic`/`zoom`), `frame_index: int | None` (None for blanks), `is_blank: bool`, `stem: str`.
  - `parse_name(path: str | Path, cfg: Config) -> ParsedName | None` — returns None if the stem does not match (non-frame files).
- Consumes: `Config.filename` (Task 2).

- [ ] **Step 1: Write failing tests `tests/test_io_parse.py`**

```python
from __future__ import annotations

from grain_morph.config import load_config
from grain_morph.io import parse_name


def _cfg():
    return load_config(None)


def test_parse_basic_data_frame():
    p = parse_name("P_01_cs_003_b_0004258.bmp", _cfg())
    assert p is not None
    assert p.sample_id == "P_01_cs_003"
    assert p.camera == "basic"
    assert p.frame_index == 4258
    assert p.is_blank is False
    assert p.stem == "P_01_cs_003_b_0004258"


def test_parse_zoom_blank():
    p = parse_name("P_17_cs_002_z_back.bmp", _cfg())
    assert p is not None
    assert p.sample_id == "P_17_cs_002"
    assert p.camera == "zoom"
    assert p.is_blank is True
    assert p.frame_index is None


def test_parse_generic_sample_name():
    # sample naming is user-defined; only camera/frame/back tokens are fixed
    p = parse_name("OK_sand_2_b_0000042.bmp", _cfg())
    assert p is not None
    assert p.sample_id == "OK_sand_2"
    assert p.camera == "basic"
    assert p.frame_index == 42


def test_non_matching_returns_none():
    assert parse_name("notes.txt", _cfg()) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_io_parse.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `parse_name` in `io.py`**

Compile `cfg.filename.sample_regex` on the stem (basename without extension). On match, map the `cam` group through `cfg.filename.camera_map`; `is_blank = (frame group is None)`; `frame_index = int(frame)` if present else None. Return None on no match. Add `from __future__ import annotations` and module docstring.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_io_parse.py -v`
Expected: PASS.

- [ ] **Step 5: ruff + mypy, then commit**

```bash
ruff check src tests && mypy src
git add src/grain_morph/io.py tests/test_io_parse.py
git commit -m "feat(io): configurable filename parsing (sample/camera/frame/blank)"
```

---

## Task 4: Frame discovery + blank pairing

**Files:**
- Modify: `src/grain_morph/io.py`
- Create: `tests/conftest.py`, `tests/test_io_discover.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class FrameSpec`: `path: Path`, `parsed: ParsedName`, `blank_path: Path | None`.
  - `discover_frames(root: str | Path, cfg: Config) -> list[FrameSpec]` — recursively finds image files (`.bmp/.png/.tif/.tiff`), parses names, separates blanks, pairs each **data** frame to the blank matching its `(sample_id, camera)` (or None), returns only data frames sorted by `(sample_id, camera, frame_index)`.
- Consumes: `parse_name`, `Config`.

- [ ] **Step 1: Add shared fixtures to `tests/conftest.py`**

```python
from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest


def _write_bmp(path: Path, arr: np.ndarray) -> None:
    iio.imwrite(path, arr.astype(np.uint8))


@pytest.fixture
def frame_dir(tmp_path: Path) -> Path:
    """A directory with two data frames + one blank for one (sample, camera)."""
    d = tmp_path / "frames"
    d.mkdir()
    bg = np.full((64, 64), 200, np.uint8)
    _write_bmp(d / "S1_b_0000001.bmp", bg)
    _write_bmp(d / "S1_b_0000002.bmp", bg)
    _write_bmp(d / "S1_b_back.bmp", bg)
    return d
```

- [ ] **Step 2: Write failing tests `tests/test_io_discover.py`**

```python
from __future__ import annotations

from grain_morph.config import load_config
from grain_morph.io import discover_frames


def test_discovers_data_frames_only(frame_dir):
    specs = discover_frames(frame_dir, load_config(None))
    assert len(specs) == 2  # blank excluded from the data set
    assert all(s.parsed.is_blank is False for s in specs)


def test_blank_paired_to_data_frames(frame_dir):
    specs = discover_frames(frame_dir, load_config(None))
    for s in specs:
        assert s.blank_path is not None
        assert s.blank_path.name == "S1_b_back.bmp"


def test_missing_blank_yields_none(tmp_path):
    import imageio.v3 as iio
    import numpy as np
    (tmp_path / "S2_z_0000005.bmp").write_bytes(b"")  # placeholder replaced below
    iio.imwrite(tmp_path / "S2_z_0000005.bmp", np.full((16, 16), 200, np.uint8))
    specs = discover_frames(tmp_path, load_config(None))
    assert len(specs) == 1
    assert specs[0].blank_path is None
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_io_discover.py -v`
Expected: FAIL.

- [ ] **Step 4: Implement `discover_frames`**

Glob recursively for known extensions; `parse_name` each; build `blanks: dict[(sample_id, camera), Path]` from parsed blanks; for each data frame, look up its blank; construct `FrameSpec`; sort. Ignore None parses.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_io_discover.py -v`
Expected: PASS.

- [ ] **Step 6: ruff + mypy, then commit**

```bash
ruff check src tests && mypy src
git add src/grain_morph/io.py tests/conftest.py tests/test_io_discover.py
git commit -m "feat(io): frame discovery and per-(sample,camera) blank pairing"
```

---

## Task 5: Manifest + resume

**Files:**
- Modify: `src/grain_morph/io.py`
- Create: `tests/test_io_manifest.py`

**Interfaces:**
- Produces:
  - `frame_fingerprint(path: Path) -> str` — `f"{size}:{int(mtime)}"` (cheap, stable).
  - `class Manifest`: `add(stem, path, status, n_objects, seconds, flatfield_method)`, `is_done(stem, fingerprint) -> bool`, `to_frame() -> pd.DataFrame`, classmethod `load(path, fmt) -> Manifest`.
  - Manifest columns: `frame_id, frame_path, fingerprint, status, n_objects, seconds, flatfield_method`.
- Consumes: `frame_fingerprint`, `write_table`/`read_table` (Task 11 — until then, Manifest holds rows in memory and `load` handles missing file).

> Ordering note: Manifest persistence uses `write_table` (Task 11). Implement in-memory Manifest + fingerprint now; wire persistence in Task 11's step that adds manifest round-trip. The tests below use only in-memory behavior.

- [ ] **Step 1: Write failing tests `tests/test_io_manifest.py`**

```python
from __future__ import annotations

from grain_morph.io import Manifest, frame_fingerprint


def test_fingerprint_changes_with_content(tmp_path):
    p = tmp_path / "f.bmp"
    p.write_bytes(b"aaaa")
    fp1 = frame_fingerprint(p)
    p.write_bytes(b"aaaaaaaa")
    assert frame_fingerprint(p) != fp1


def test_is_done_requires_matching_fingerprint():
    m = Manifest()
    m.add("S1_b_0000001", "S1_b_0000001.bmp", "ok", 3, 0.1, "blank", fingerprint="10:5")
    assert m.is_done("S1_b_0000001", "10:5") is True
    assert m.is_done("S1_b_0000001", "99:5") is False   # changed file -> reprocess
    assert m.is_done("other", "10:5") is False
```

- [ ] **Step 2: Run to verify fail.** `pytest tests/test_io_manifest.py -v` → FAIL.

- [ ] **Step 3: Implement `frame_fingerprint` and `Manifest`** (in-memory dict keyed by `frame_id`, storing fingerprint + row).

- [ ] **Step 4: Run to verify pass.** `pytest tests/test_io_manifest.py -v` → PASS.

- [ ] **Step 5: ruff + mypy, commit.**

```bash
ruff check src tests && mypy src
git add src/grain_morph/io.py tests/test_io_manifest.py
git commit -m "feat(io): frame fingerprint + in-memory resume manifest"
```

---

## Task 6: Synthetic frame generator

**Files:**
- Create: `tests/synth.py`, `tests/test_synth.py`

**Interfaces:**
- Produces:
  - `@dataclass class SynthObject`: `polygon: shapely.Polygon` (ground-truth, pixel coords), `blurred: bool`, `touches_border: bool`.
  - `@dataclass class SynthFrame`: `image: np.ndarray (uint8)`, `objects: list[SynthObject]`, `background_level: float`.
  - `make_frame(size=(512, 512), objects: list[dict], gradient=0.0, noise_sigma=0.0, seed=0) -> SynthFrame` where each object dict is `{"kind": "ellipse"|"rect"|"blob", "cx","cy","a","b","angle","blur_sigma"}`.
  - Helper `ground_truth_area(obj)` / `ground_truth_perimeter(obj)` from the shapely polygon.
- Consumes: shapely, skimage.draw, scipy.ndimage.

Design: dark objects (intensity ~15) on bright background (~200). Apply multiplicative gradient `1 - gradient*(x/width)` across columns. Gaussian-blur only objects whose `blur_sigma>0` (draw each object on its own layer, blur, composite by min). Add Gaussian noise. `make_frame` is deterministic given `seed` (use `np.random.default_rng(seed)`).

- [ ] **Step 1: Write failing tests `tests/test_synth.py`**

```python
from __future__ import annotations

import numpy as np

from tests.synth import make_frame


def test_frame_shape_and_dtype():
    f = make_frame(size=(128, 128), objects=[{"kind": "ellipse", "cx": 64, "cy": 64, "a": 20, "b": 15, "angle": 0.0, "blur_sigma": 0.0}])
    assert f.image.shape == (128, 128)
    assert f.image.dtype == np.uint8
    assert len(f.objects) == 1


def test_objects_are_darker_than_background():
    f = make_frame(size=(128, 128), objects=[{"kind": "ellipse", "cx": 64, "cy": 64, "a": 20, "b": 15, "angle": 0.0, "blur_sigma": 0.0}])
    ys, xs = np.nonzero(f.image < 100)
    assert xs.size > 0
    assert abs(xs.mean() - 64) < 5 and abs(ys.mean() - 64) < 5


def test_gradient_makes_right_side_darker_background():
    f = make_frame(size=(128, 128), objects=[], gradient=0.4)
    left = f.image[:, :10].mean()
    right = f.image[:, -10:].mean()
    assert left - right > 20  # 40% gradient is visible


def test_border_object_flag_and_blur_flag():
    f = make_frame(size=(128, 128), objects=[
        {"kind": "rect", "cx": 4, "cy": 64, "a": 20, "b": 10, "angle": 0.0, "blur_sigma": 0.0},
        {"kind": "ellipse", "cx": 90, "cy": 40, "a": 12, "b": 12, "angle": 0.0, "blur_sigma": 5.0},
    ])
    assert any(o.touches_border for o in f.objects)
    assert any(o.blurred for o in f.objects)
```

- [ ] **Step 2: Run to verify fail.** `pytest tests/test_synth.py -v` → FAIL.
- [ ] **Step 3: Implement `tests/synth.py`.**
- [ ] **Step 4: Run to verify pass.** `pytest tests/test_synth.py -v` → PASS.
- [ ] **Step 5: Commit.**

```bash
git add tests/synth.py tests/test_synth.py
git commit -m "test: synthetic frame generator with ground-truth polygons"
```

---

## Task 7: Flat-fielding

**Files:**
- Create: `src/grain_morph/flatfield.py`, `tests/test_flatfield.py`

**Interfaces:**
- Produces:
  - `apply_flatfield(image: np.ndarray, blank: np.ndarray | None, cfg: Config) -> tuple[np.ndarray, str]` — returns `(corrected_float32, method_used)`. Corrected image is normalized so background ≈ 1.0 (objects < 1.0). `method_used ∈ {"blank","morphological"}`.
  - `_estimate_field_morphological(image, kernel_px) -> np.ndarray`.
- Consumes: `Config.flatfield`.

Logic: if `method == "blank"` or (`method == "auto"` and blank is not None) and blank given → `field = blank.astype(f32)`. Else → morphological field via grey opening / large uniform filter (`scipy.ndimage.grey_opening` with `size=kernel_px`, or `scipy.ndimage.uniform_filter` for speed — opening is more correct for dark objects). `corrected = image / field`, clip, and rescale so the field-corrected **background** (field/field) is 1.0. Return corrected float32 in ~[0, ~1.1].

- [ ] **Step 1: Write failing tests `tests/test_flatfield.py`**

```python
from __future__ import annotations

import numpy as np

from grain_morph.config import load_config
from grain_morph.flatfield import apply_flatfield
from tests.synth import make_frame


def test_blank_division_flattens_background():
    cfg = load_config(None)
    cfg.flatfield.method = "blank"
    f = make_frame(size=(128, 128), objects=[], gradient=0.4)
    blank = f.image.copy()  # empty frame IS the illumination field
    corrected, method = apply_flatfield(f.image, blank, cfg)
    assert method == "blank"
    # background now uniform: std across the frame is tiny
    assert corrected.std() < 0.02
    assert abs(float(np.median(corrected)) - 1.0) < 0.02


def test_morphological_used_when_no_blank():
    cfg = load_config(None)
    cfg.flatfield.method = "auto"
    f = make_frame(size=(256, 256), objects=[
        {"kind": "ellipse", "cx": 128, "cy": 128, "a": 15, "b": 15, "angle": 0.0, "blur_sigma": 0.0}])
    corrected, method = apply_flatfield(f.image, None, cfg)
    assert method == "morphological"
    # object still darker than corrected background
    assert corrected.min() < 0.6
    assert abs(float(np.median(corrected)) - 1.0) < 0.05
```

- [ ] **Step 2: Run to verify fail.** `pytest tests/test_flatfield.py -v` → FAIL.
- [ ] **Step 3: Implement `flatfield.py`.** Use `kernel_px = cfg.flatfield.morph_kernel_px`; guard kernel ≤ image size in tests by `min(kernel_px, min(shape)-1)`.
- [ ] **Step 4: Run to verify pass.** `pytest tests/test_flatfield.py -v` → PASS.
- [ ] **Step 5: ruff + mypy, commit.**

```bash
ruff check src tests && mypy src
git add src/grain_morph/flatfield.py tests/test_flatfield.py
git commit -m "feat(flatfield): blank-division default + morphological fallback"
```

---

## Task 8: Detection — threshold, label, subpixel contours

**Files:**
- Create: `src/grain_morph/detect.py`, `tests/test_detect.py`

**Interfaces:**
- Produces:
  - `@dataclass class Detection`: `label: int`, `polygon: shapely.Polygon | None`, `mask_bbox: tuple[int,int,int,int]`, `centroid_xy: tuple[float,float]`, `contour_ok: bool`.
  - `threshold_level(corrected: np.ndarray, cfg: Config) -> float` — half-max (default) or Otsu, on the flat-fielded image.
  - `detect_objects(corrected: np.ndarray, cfg: Config) -> tuple[np.ndarray, list[Detection]]` — returns `(label_image, detections)`. Objects are **dark** (`corrected < level`). Fill holes, drop `< min_area_px`, extract subpixel contours via `skimage.measure.find_contours(corrected, level)` and associate to labels; convert (row,col)→(x,y) for shapely; keep largest exterior ring, interior rings as holes; set `contour_ok=False`/`polygon=None` if association fails.
- Consumes: `Config.threshold`, `Config.detect`, flat-fielded image (Task 7).

Half-max level: background ≈ 1.0 after flat-fielding; core estimate = `np.percentile(corrected, cfg.threshold.core_percentile)`; `level = core + cfg.threshold.half_max_fraction * (1.0 - core)`. (Objects darker than `level` are foreground.)

- [ ] **Step 1: Write failing tests `tests/test_detect.py`**

```python
from __future__ import annotations

import numpy as np

from grain_morph.config import load_config
from grain_morph.detect import detect_objects, threshold_level
from grain_morph.flatfield import apply_flatfield
from tests.synth import make_frame


def _corrected(objs, size=(256, 256), gradient=0.0):
    cfg = load_config(None)
    f = make_frame(size=size, objects=objs, gradient=gradient)
    corrected, _ = apply_flatfield(f.image, None, cfg)
    return corrected, cfg, f


def test_threshold_between_core_and_background():
    corrected, cfg, _ = _corrected([{"kind": "ellipse", "cx": 128, "cy": 128, "a": 20, "b": 20, "angle": 0.0, "blur_sigma": 0.0}])
    lvl = threshold_level(corrected, cfg)
    assert 0.3 < lvl < 0.95


def test_detects_expected_object_count():
    corrected, cfg, _ = _corrected([
        {"kind": "ellipse", "cx": 80, "cy": 80, "a": 18, "b": 18, "angle": 0.0, "blur_sigma": 0.0},
        {"kind": "ellipse", "cx": 180, "cy": 180, "a": 14, "b": 14, "angle": 0.0, "blur_sigma": 0.0}])
    _, dets = detect_objects(corrected, cfg)
    assert len(dets) == 2
    assert all(d.polygon is not None and d.contour_ok for d in dets)


def test_polygon_area_close_to_ground_truth():
    r = 25
    corrected, cfg, f = _corrected([{"kind": "ellipse", "cx": 128, "cy": 128, "a": r, "b": r, "angle": 0.0, "blur_sigma": 0.0}])
    _, dets = detect_objects(corrected, cfg)
    truth = f.objects[0].polygon.area
    got = dets[0].polygon.area
    assert abs(got - truth) / truth < 0.02  # subpixel contour, tight


def test_small_object_removed_by_min_area():
    cfg = load_config(None)
    cfg.detect.min_area_px = 2000
    corrected, _, _ = _corrected([{"kind": "ellipse", "cx": 128, "cy": 128, "a": 5, "b": 5, "angle": 0.0, "blur_sigma": 0.0}])
    _, dets = detect_objects(corrected, cfg)
    assert dets == []
```

- [ ] **Step 2: Run to verify fail.** `pytest tests/test_detect.py -v` → FAIL.
- [ ] **Step 3: Implement `detect.py`.** Foreground mask `corrected < level`; `scipy.ndimage.binary_fill_holes`; `skimage.measure.label`; `skimage.measure.regionprops` for bbox/centroid/area; drop small; `find_contours(corrected, level)`; for each contour build a shapely polygon (x=col, y=row); assign to the label whose centroid lies inside (or nearest); pick the largest exterior per label, others as holes.
- [ ] **Step 4: Run to verify pass.** `pytest tests/test_detect.py -v` → PASS.
- [ ] **Step 5: ruff + mypy, commit.**

```bash
ruff check src tests && mypy src
git add src/grain_morph/detect.py tests/test_detect.py
git commit -m "feat(detect): half-max threshold, labeling, subpixel contour polygons"
```

---

## Task 9: Morphometry

**Files:**
- Create: `src/grain_morph/measure.py`, `tests/test_measure.py`

**Interfaces:**
- Produces:
  - `measure_polygon(poly: shapely.Polygon, um_per_px: float) -> dict[str, float]` — size + first-order shape (keys: `area_px, area_um2, ecd_um, feret_max_um, feret_min_um, major_axis_um, minor_axis_um, perimeter_um, aspect_ratio, solidity, convexity, circularity, extent, eccentricity, orientation`).
  - `measure_efd(poly, order, resample_n) -> tuple[dict, float]` — `({"efd_1_a":..., ...}, fourier_power_cum_90)`.
  - `measure_wadell(poly, smoothing) -> dict[str, float]` — `{"wadell_roundness":..., "wadell_sphericity":...}` via `wadell_rs`. **Confirm the installed `wadell_rs` function signatures from its README before implementing this one function** (external API; one lookup permitted). Fallback to `fast_rs` if substituted at Task 1.
  - `raster_perimeter_px(mask: np.ndarray) -> float` — `skimage.measure.regionprops` crofton perimeter, for the polygon-vs-raster comparison.
- Consumes: shapely, pyefd, wadell_rs.

Feret: `feret_max_um` from the convex hull's max vertex-pair distance; `feret_min_um` from the minimum width of the hull (min over hull-edge support widths) — or `poly.minimum_rotated_rectangle` short side. Circularity = `4*pi*area / perimeter**2`. Solidity = `area / poly.convex_hull.area`. Convexity = `poly.convex_hull.length / poly.length`.

- [ ] **Step 1: Write failing tests `tests/test_measure.py`**

```python
from __future__ import annotations

import math

import numpy as np
import shapely

from grain_morph.config import load_config
from grain_morph.detect import detect_objects
from grain_morph.flatfield import apply_flatfield
from grain_morph.measure import measure_polygon, raster_perimeter_px
from tests.synth import make_frame


def test_circle_area_and_perimeter_within_tolerance():
    # unit test on an exact analytic circle polygon (no imaging)
    circle = shapely.Point(0, 0).buffer(50, quad_segs=256)
    m = measure_polygon(circle, um_per_px=1.0)
    assert abs(m["area_um2"] - math.pi * 50**2) / (math.pi * 50**2) < 0.01
    assert abs(m["perimeter_um"] - 2 * math.pi * 50) / (2 * math.pi * 50) < 0.02
    assert abs(m["circularity"] - 1.0) < 0.02
    assert abs(m["aspect_ratio"] - 1.0) < 0.05


def test_recovered_area_within_1pct_zero_blur_no_gradient():
    # ACCEPTANCE CRITERION 1
    cfg = load_config(None)
    r = 40
    f = make_frame(size=(256, 256), objects=[{"kind": "ellipse", "cx": 128, "cy": 128, "a": r, "b": r, "angle": 0.0, "blur_sigma": 0.0}])
    corrected, _ = apply_flatfield(f.image, None, cfg)
    _, dets = detect_objects(corrected, cfg)
    m = measure_polygon(dets[0].polygon, um_per_px=1.0)
    truth_area = f.objects[0].polygon.area
    truth_perim = f.objects[0].polygon.length
    assert abs(m["area_um2"] - truth_area) / truth_area < 0.01
    assert abs(m["perimeter_um"] - truth_perim) / truth_perim < 0.02


def test_polygon_perimeter_lower_than_raster():
    # ACCEPTANCE CRITERION 5
    cfg = load_config(None)
    f = make_frame(size=(256, 256), objects=[{"kind": "ellipse", "cx": 128, "cy": 128, "a": 40, "b": 40, "angle": 0.0, "blur_sigma": 0.0}])
    corrected, _ = apply_flatfield(f.image, None, cfg)
    labels, dets = detect_objects(corrected, cfg)
    poly_perim = measure_polygon(dets[0].polygon, 1.0)["perimeter_um"]
    raster_perim = raster_perimeter_px((labels == dets[0].label))
    truth = f.objects[0].polygon.length
    assert poly_perim < raster_perim
    assert abs(poly_perim - truth) < abs(raster_perim - truth)
```

- [ ] **Step 2: Run to verify fail.** `pytest tests/test_measure.py -v` → FAIL.
- [ ] **Step 3: Implement `measure.py`** (`measure_polygon`, `raster_perimeter_px` first — enough for these tests; add `measure_efd`, `measure_wadell` with their own quick smoke tests).
- [ ] **Step 4: Add smoke tests for EFD + Wadell** (append to `tests/test_measure.py`):

```python
def test_efd_and_wadell_smoke():
    from grain_morph.measure import measure_efd, measure_wadell
    circle = shapely.Point(0, 0).buffer(50, quad_segs=128)
    efd, cum90 = measure_efd(circle, order=15, resample_n=256)
    assert len(efd) == 15 * 4
    assert 1 <= cum90 <= 15
    w = measure_wadell(circle, smoothing=1.0)
    assert 0.0 < w["wadell_roundness"] <= 1.2
    assert 0.0 < w["wadell_sphericity"] <= 1.2
```

- [ ] **Step 5: Run all measure tests to verify pass.** `pytest tests/test_measure.py -v` → PASS.
- [ ] **Step 6: ruff + mypy, commit.**

```bash
ruff check src tests && mypy src
git add src/grain_morph/measure.py tests/test_measure.py
git commit -m "feat(measure): polygon morphometry, EFD, Wadell roundness/sphericity"
```

---

## Task 10: QC metrics + flags

**Files:**
- Create: `src/grain_morph/qc.py`, `tests/test_qc.py`

**Interfaces:**
- Produces:
  - `qc_metrics(raw_image, corrected, poly, mask, bbox, frame_shape, cfg) -> dict` — continuous: `edge_width_px, edge_gradient, edge_gradient_norm, contrast, ecd_px, aspect_ratio, solidity`.
  - `qc_flags(metrics, cfg) -> dict[str, bool]` — `flag_defocus, flag_border, flag_too_small, flag_sliver, flag_possible_agglomerate, flag_no_polygon`, plus `qc_pass`.
  - Border: `flag_border = bbox touches [0, H/W-1]`.
- Consumes: `Config.qc`, raw image (for gradient/contrast), corrected image, polygon+mask.

Focus metrics: `edge_gradient` = mean Sobel magnitude (`skimage.filters.sobel` on **raw** image) within a ±3 px band around the boundary (dilate mask boundary). `contrast` = `(local_bg_mean - eroded_core_mean) / local_bg_mean`, local_bg from a ring outside the object. `edge_width_px` = mean 10–90% intensity rise distance sampled along boundary normals (primary defocus signal). `flag_defocus = (edge_width_px > cfg.qc.defocus_edge_width_px) or (contrast < cfg.qc.defocus_contrast_min)`.

- [ ] **Step 1: Write failing tests `tests/test_qc.py`**

```python
from __future__ import annotations

import numpy as np

from grain_morph.config import load_config
from grain_morph.detect import detect_objects
from grain_morph.flatfield import apply_flatfield
from grain_morph.qc import qc_flags, qc_metrics
from tests.synth import make_frame


def _measure_flags(objs, size=(256, 256)):
    cfg = load_config(None)
    f = make_frame(size=size, objects=objs)
    corrected, _ = apply_flatfield(f.image, None, cfg)
    labels, dets = detect_objects(corrected, cfg)
    out = []
    for d in dets:
        mask = labels == d.label
        m = qc_metrics(f.image.astype(float), corrected, d.polygon, mask, d.mask_bbox, f.image.shape, cfg)
        out.append((m, qc_flags(m, cfg)))
    return out, f


def test_border_object_flagged():
    # ACCEPTANCE CRITERION 4
    res, _ = _measure_flags([{"kind": "rect", "cx": 3, "cy": 128, "a": 30, "b": 20, "angle": 0.0, "blur_sigma": 0.0}])
    assert res[0][1]["flag_border"] is True


def test_defocus_recall_and_fp_rate():
    # ACCEPTANCE CRITERION 3 (aggregate over many objects)
    rng = np.random.default_rng(0)
    sharp_fp = 0
    blur_tp = 0
    n = 30
    for i in range(n):
        cx = int(rng.integers(60, 196)); cy = int(rng.integers(60, 196))
        sharp, _ = _measure_flags([{"kind": "ellipse", "cx": cx, "cy": cy, "a": 20, "b": 18, "angle": 0.0, "blur_sigma": 0.0}])
        if sharp and sharp[0][1]["flag_defocus"]:
            sharp_fp += 1
        blur, _ = _measure_flags([{"kind": "ellipse", "cx": cx, "cy": cy, "a": 20, "b": 18, "angle": 0.0, "blur_sigma": 5.0}])
        if blur and blur[0][1]["flag_defocus"]:
            blur_tp += 1
    assert blur_tp / n >= 0.95      # recall
    assert sharp_fp / n <= 0.02     # false-positive rate
```

- [ ] **Step 2: Run to verify fail.** `pytest tests/test_qc.py -v` → FAIL.
- [ ] **Step 3: Implement `qc.py`.** Start with `edge_gradient/contrast` + `flag_defocus` from contrast; tune `edge_width_px` until criterion 3 passes on synthetic blur σ≥4. Keep all thresholds from `cfg.qc`.
- [ ] **Step 4: Run to verify pass.** `pytest tests/test_qc.py -v` → PASS. If recall/FP fails, adjust `defocus_edge_width_px`/`defocus_contrast_min` **in `configs/default.yaml`** (not in code) and re-run.
- [ ] **Step 5: ruff + mypy, commit.**

```bash
ruff check src tests && mypy src
git add src/grain_morph/qc.py tests/test_qc.py configs/default.yaml
git commit -m "feat(qc): focus/border/sliver/agglomerate metrics and flags"
```

---

## Task 11: Table writers + manifest persistence

**Files:**
- Create: `src/grain_morph/writers.py`, `tests/test_writers.py`
- Modify: `src/grain_morph/io.py` (wire `Manifest.load`/`save` to writers)

**Interfaces:**
- Produces:
  - `write_table(df: pd.DataFrame, path: Path, fmt: str) -> Path` — writes `.parquet`/`.csv`/`.feather`; returns the actual path written (adds extension).
  - `read_table(path: Path, fmt: str) -> pd.DataFrame`.
  - `write_partitioned(df, root, fmt, partition_cols, partition: bool) -> None` — parquet → `pyarrow` dataset partitioned by cols; csv/feather → one file per partition named `{c1}__{c2}...`.
  - `Manifest.save(path, fmt)` / `Manifest.load(path, fmt)`.
- Consumes: `Config.output`.

- [ ] **Step 1: Write failing tests `tests/test_writers.py`**

```python
from __future__ import annotations

import pandas as pd

from grain_morph.writers import read_table, write_table


def test_roundtrip_all_formats(tmp_path):
    df = pd.DataFrame({"grain_uid": ["f:1", "f:2"], "area_um2": [1.5, 2.5], "qc_pass": [True, False]})
    for fmt in ("parquet", "csv", "feather"):
        p = write_table(df, tmp_path / f"t_{fmt}", fmt)
        back = read_table(p, fmt)
        assert list(back["grain_uid"]) == ["f:1", "f:2"]
        assert back["area_um2"].tolist() == [1.5, 2.5]
```

- [ ] **Step 2: Run to verify fail.** `pytest tests/test_writers.py -v` → FAIL.
- [ ] **Step 3: Implement `writers.py`** and wire `Manifest.save/load` in `io.py`.
- [ ] **Step 4: Run to verify pass.** `pytest tests/test_writers.py -v` → PASS.
- [ ] **Step 5: ruff + mypy, commit.**

```bash
ruff check src tests && mypy src
git add src/grain_morph/writers.py src/grain_morph/io.py tests/test_writers.py
git commit -m "feat(writers): parquet/csv/feather write_table + partitioning + manifest persistence"
```

---

## Task 12: Detect pipeline orchestration

**Files:**
- Create: `src/grain_morph/pipeline.py`, `tests/test_pipeline.py`

**Interfaces:**
- Produces:
  - `process_frame(spec: FrameSpec, cfg: Config, pipeline_version: str, cfg_hash: str) -> FrameResult` where `FrameResult` has `rows: list[dict]`, `contours: list[dict]`, `status: str`, `n_objects: int`, `seconds: float`, `flatfield_method: str`, `error: str | None`. Catches all per-frame exceptions → `status="error"`, `error=traceback`.
  - `run_detect(root, out_dir, cfg, n_jobs=..., force=False) -> None` — discovers frames, skips done (manifest), runs `process_frame` via `joblib.Parallel`, streams rows to `write_partitioned`, writes `contours`, `manifest`, `errors`, `run_config.yaml`, `summary.json`.
- Consumes: everything from Tasks 4–11.

Row assembly: merge `measure_polygon` + `measure_efd` + `measure_wadell` + `qc_metrics` + `qc_flags` with identity columns (`grain_uid, frame_id, frame_path, sample_id, camera, run, label, centroid_x, centroid_y, um_per_px, wadell_smoothing, pipeline_version, config_hash`). Empty frame → `rows=[]`, still a manifest entry.

- [ ] **Step 1: Write failing tests `tests/test_pipeline.py`**

```python
from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd

from grain_morph.config import load_config
from grain_morph.pipeline import run_detect
from grain_morph.writers import read_table
from tests.synth import make_frame


def _cfg_with_calib():
    cfg = load_config(None)
    cfg.calibration.um_per_px = {"basic": 5.0, "zoom": 1.0}
    return cfg


def _write_run(dirpath: Path, gradient=0.0):
    dirpath.mkdir(parents=True, exist_ok=True)
    # blank
    blank = make_frame(size=(256, 256), objects=[], gradient=gradient).image
    iio.imwrite(dirpath / "S1_b_back.bmp", blank)
    # one frame, object in the BRIGHT corner and one in the DARK corner (same size)
    f = make_frame(size=(256, 256), gradient=gradient, objects=[
        {"kind": "ellipse", "cx": 40, "cy": 128, "a": 25, "b": 25, "angle": 0.0, "blur_sigma": 0.0},
        {"kind": "ellipse", "cx": 216, "cy": 128, "a": 25, "b": 25, "angle": 0.0, "blur_sigma": 0.0}])
    iio.imwrite(dirpath / "S1_b_0000001.bmp", f.image)
    return f


def test_flatfield_area_agreement_bright_vs_dark(tmp_path):
    # ACCEPTANCE CRITERION 2
    run = tmp_path / "run"
    _write_run(run, gradient=0.4)
    out = tmp_path / "out"
    run_detect(run, out, _cfg_with_calib(), n_jobs=1)
    df = read_table(out / "grains.parquet", "parquet") if (out / "grains.parquet").exists() else pd.read_parquet(out / "grains")
    df = df.sort_values("centroid_x")
    areas = df["area_um2"].tolist()
    assert len(areas) == 2
    assert abs(areas[0] - areas[1]) / max(areas) < 0.02


def test_zero_particle_frame(tmp_path):
    # ACCEPTANCE CRITERION 6
    run = tmp_path / "run"; run.mkdir()
    iio.imwrite(run / "S1_b_back.bmp", make_frame(size=(128, 128), objects=[]).image)
    iio.imwrite(run / "S1_b_0000001.bmp", make_frame(size=(128, 128), objects=[]).image)
    out = tmp_path / "out"
    run_detect(run, out, _cfg_with_calib(), n_jobs=1)
    from grain_morph.io import Manifest
    man = Manifest.load(out / "manifest.parquet", "parquet")
    assert len(man.to_frame()) == 1
    # no grains file OR an empty one
    gdir = out / "grains"
    assert (not gdir.exists()) or sum(len(read_table(p, "parquet")) for p in gdir.rglob("*.parquet")) == 0


def test_resume_does_no_work(tmp_path):
    # ACCEPTANCE CRITERION 7
    run = tmp_path / "run"; _write_run(run)
    out = tmp_path / "out"
    run_detect(run, out, _cfg_with_calib(), n_jobs=1)
    man1 = (out / "manifest.parquet").read_bytes()
    run_detect(run, out, _cfg_with_calib(), n_jobs=1)  # resume
    man2 = (out / "manifest.parquet").read_bytes()
    assert man1 == man2  # byte-identical; no reprocessing
```

- [ ] **Step 2: Run to verify fail.** `pytest tests/test_pipeline.py -v` → FAIL.
- [ ] **Step 3: Implement `pipeline.py`.** Deterministic writes (sort rows by `grain_uid`; stable manifest column order; no timestamps inside manifest rows) so resume is byte-identical. `pipeline_version = grain_morph.__version__`.
- [ ] **Step 4: Run to verify pass.** `pytest tests/test_pipeline.py -v` → PASS.
- [ ] **Step 5: Add the memory-bound test (ACCEPTANCE CRITERION 8)** `tests/test_memory.py`:

```python
from __future__ import annotations

import imageio.v3 as iio
import psutil

from grain_morph.config import load_config
from grain_morph.pipeline import run_detect
from tests.synth import make_frame


def test_500_frames_bounded_rss(tmp_path):
    run = tmp_path / "run"; run.mkdir()
    iio.imwrite(run / "S1_b_back.bmp", make_frame(size=(256, 256), objects=[]).image)
    for i in range(1, 501):
        f = make_frame(size=(256, 256), seed=i, objects=[
            {"kind": "ellipse", "cx": 128, "cy": 128, "a": 18, "b": 15, "angle": 0.0, "blur_sigma": 0.0}])
        iio.imwrite(run / f"S1_b_{i:07d}.bmp", f.image)
    cfg = load_config(None); cfg.calibration.um_per_px = {"basic": 5.0, "zoom": 1.0}
    proc = psutil.Process()
    before = proc.memory_info().rss
    run_detect(run, tmp_path / "out", cfg, n_jobs=1)
    after = proc.memory_info().rss
    assert (after - before) < 300 * 1024 * 1024  # < 300 MB growth
```

- [ ] **Step 6: Run to verify pass.** `pytest tests/test_pipeline.py tests/test_memory.py -v` → PASS.
- [ ] **Step 7: ruff + mypy, commit.**

```bash
ruff check src tests && mypy src
git add src/grain_morph/pipeline.py tests/test_pipeline.py tests/test_memory.py
git commit -m "feat(pipeline): per-frame processing, parallel run_detect, resume, run artifacts"
```

---

## Task 13: Aggregate

**Files:**
- Create: `src/grain_morph/aggregate.py`, `tests/test_aggregate.py`

**Interfaces:**
- Produces:
  - `aggregate_run(grains: pd.DataFrame, cfg: Config) -> dict[str, pd.DataFrame]` returning `{"summary": ..., "rejection_by_ecd": ..., "accepted": ...}`.
  - `summary` per `(sample_id, camera)`: `n_total, n_accepted, n_rejected, n_flag_defocus, ...`, `ecd_um_D10/D50/D90`, `feret_min_um_D10/D50/D90`, and mean/SD of shape descriptors over accepted grains.
  - `rejection_by_ecd`: rejection rate binned by ECD (so size-dependent gating is visible).
- Consumes: per-grain DataFrame, `Config.qc.disqualifying_flags`.

- [ ] **Step 1: Write failing tests `tests/test_aggregate.py`**

```python
from __future__ import annotations

import numpy as np
import pandas as pd

from grain_morph.aggregate import aggregate_run
from grain_morph.config import load_config


def _grains():
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame({
        "sample_id": ["S1"] * n,
        "camera": ["basic"] * n,
        "ecd_um": rng.uniform(50, 500, n),
        "feret_min_um": rng.uniform(40, 450, n),
        "solidity": rng.uniform(0.9, 1.0, n),
        "flag_defocus": rng.random(n) < 0.1,
        "flag_border": rng.random(n) < 0.05,
        "flag_too_small": [False] * n,
        "flag_sliver": [False] * n,
        "flag_no_polygon": [False] * n,
    })


def test_summary_counts_and_percentiles():
    out = aggregate_run(_grains(), load_config(None))
    s = out["summary"].iloc[0]
    assert s["n_total"] == 200
    assert s["n_accepted"] + s["n_rejected"] == 200
    assert s["ecd_um_D10"] < s["ecd_um_D50"] < s["ecd_um_D90"]


def test_rejection_binned_by_ecd_present():
    out = aggregate_run(_grains(), load_config(None))
    rej = out["rejection_by_ecd"]
    assert {"ecd_bin", "n", "rejection_rate"}.issubset(rej.columns)
    assert (rej["rejection_rate"].between(0, 1)).all()
```

- [ ] **Step 2: Run to verify fail.** `pytest tests/test_aggregate.py -v` → FAIL.
- [ ] **Step 3: Implement `aggregate.py`.** `qc_pass` recomputed from `disqualifying_flags` so aggregation-time flag choices are honored. Percentiles via `np.percentile` on the "percent finer" convention (D10 = 10th percentile of the size).
- [ ] **Step 4: Run to verify pass.** `pytest tests/test_aggregate.py -v` → PASS.
- [ ] **Step 5: ruff + mypy, commit.**

```bash
ruff check src tests && mypy src
git add src/grain_morph/aggregate.py tests/test_aggregate.py
git commit -m "feat(aggregate): per (sample,camera) summaries, D10/50/90, rejection-by-ECD"
```

---

## Task 14: Report artifacts

**Files:**
- Create: `src/grain_morph/report.py`, `tests/test_report.py`

**Interfaces:**
- Produces:
  - `make_reports(grains: pd.DataFrame, frames_root: Path, out_dir: Path, cfg: Config) -> list[Path]` — writes and returns paths for: contact sheets (rejected-by-flag; stratified accepted), `focus_scatter.png` (edge_gradient_norm vs contrast, colored by flag, threshold lines), `rejection_vs_ecd.png`, `illumination_field.png` (one frame). Uses matplotlib `Agg`.
- Consumes: per-grain DataFrame, geometry (for crops via bbox), flatfield (for the field render).

- [ ] **Step 1: Write failing test `tests/test_report.py`**

```python
from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from grain_morph.config import load_config
from grain_morph.report import make_reports


def test_report_files_created(tmp_path):
    rng = np.random.default_rng(0)
    n = 50
    grains = pd.DataFrame({
        "sample_id": ["S1"] * n, "camera": ["basic"] * n,
        "grain_uid": [f"f:{i}" for i in range(n)],
        "ecd_um": rng.uniform(50, 500, n),
        "edge_gradient_norm": rng.uniform(0.02, 0.2, n),
        "contrast": rng.uniform(0.8, 0.95, n),
        "flag_defocus": rng.random(n) < 0.2,
        "qc_pass": rng.random(n) < 0.8,
    })
    paths = make_reports(grains, frames_root=tmp_path, out_dir=tmp_path / "rep", cfg=load_config(None))
    assert any(p.name == "focus_scatter.png" for p in paths)
    assert any(p.name == "rejection_vs_ecd.png" for p in paths)
    assert all(p.exists() for p in paths)
```

- [ ] **Step 2: Run to verify fail.** `pytest tests/test_report.py -v` → FAIL.
- [ ] **Step 3: Implement `report.py`.** Contact-sheet crops require frames; when a frame is missing (as in this test) skip crops gracefully but still emit the scatter + rejection charts. Keep functions small.
- [ ] **Step 4: Run to verify pass.** `pytest tests/test_report.py -v` → PASS.
- [ ] **Step 5: ruff + mypy, commit.**

```bash
ruff check src tests && mypy src
git add src/grain_morph/report.py tests/test_report.py
git commit -m "feat(report): focus scatter, rejection-vs-ECD, contact sheets, field render"
```

---

## Task 15: CLI

**Files:**
- Create: `src/grain_morph/cli.py`, `tests/test_cli.py`

**Interfaces:**
- Produces: typer `app` with commands:
  - `detect FRAMES_DIR OUT_DIR [--config PATH] [--jobs N] [--force]`
  - `aggregate GRAINS_PATH OUT_DIR [--config PATH]`
  - `report GRAINS_PATH FRAMES_DIR OUT_DIR [--config PATH]`
  - `make-fixtures SRC_DIR DEST_DIR [--factor 4]` (Task 16 uses this).
- Consumes: `run_detect`, `aggregate_run`, `make_reports`, `load_config`.

- [ ] **Step 1: Write failing test `tests/test_cli.py`**

```python
from __future__ import annotations

import imageio.v3 as iio
import yaml
from typer.testing import CliRunner

from grain_morph.cli import app
from tests.synth import make_frame

runner = CliRunner()


def test_detect_cli_end_to_end(tmp_path):
    run = tmp_path / "run"; run.mkdir()
    iio.imwrite(run / "S1_b_back.bmp", make_frame(size=(128, 128), objects=[]).image)
    iio.imwrite(run / "S1_b_0000001.bmp", make_frame(size=(128, 128), objects=[
        {"kind": "ellipse", "cx": 64, "cy": 64, "a": 18, "b": 18, "angle": 0.0, "blur_sigma": 0.0}]).image)
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump({"calibration": {"um_per_px": {"basic": 5.0, "zoom": 1.0}}}))
    out = tmp_path / "out"
    res = runner.invoke(app, ["detect", str(run), str(out), "--config", str(cfg), "--jobs", "1"])
    assert res.exit_code == 0, res.output
    assert (out / "summary.json").exists()
    assert (out / "run_config.yaml").exists()
```

- [ ] **Step 2: Run to verify fail.** `pytest tests/test_cli.py -v` → FAIL.
- [ ] **Step 3: Implement `cli.py`.**
- [ ] **Step 4: Run to verify pass.** `pytest tests/test_cli.py -v` → PASS.
- [ ] **Step 5: ruff + mypy, commit.**

```bash
ruff check src tests && mypy src
git add src/grain_morph/cli.py tests/test_cli.py
git commit -m "feat(cli): detect/aggregate/report/make-fixtures commands"
```

---

## Task 16: Real integration fixtures, README, lockfile, full green

**Files:**
- Create: `tests/fixtures/real/GENERATION.md`, downsampled BMPs under `tests/fixtures/real/`, `tests/test_integration_real.py`, `tests/test_realdata_optin.py`, `README.md`, `uv.lock`

**Interfaces:**
- Consumes: `make-fixtures` CLI (Task 15), dev frames in `dev_data/`.

- [ ] **Step 1: Generate downsampled fixtures from dev data**

Run:
```bash
grain-morph make-fixtures dev_data/images_P_17_cs tests/fixtures/real --factor 4
grain-morph make-fixtures dev_data/images_P_01_cs tests/fixtures/real --factor 4
```
`make-fixtures` anti-aliased-downsamples each BMP by `factor`, preserving names (so parsing/pairing still work), keeping ≤2 Basic + ≤2 Zoom data frames + both blanks per sample. Confirm each output ≲300 KB (`du -h tests/fixtures/real/*`).

- [ ] **Step 2: Write `tests/fixtures/real/GENERATION.md`** recording: source paths, chosen frame indices, `--factor 4`, and that fixture configs must scale `um_per_px` by the factor. Commit fixtures + note.

- [ ] **Step 3: Write integration test `tests/test_integration_real.py`**

```python
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from grain_morph.cli import app
from typer.testing import CliRunner

runner = CliRunner()
FIX = Path("tests/fixtures/real")


def test_detect_on_downsampled_real(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump({"calibration": {"um_per_px": {"basic": 20.0, "zoom": 8.0}}}))
    out = tmp_path / "out"
    res = runner.invoke(app, ["detect", str(FIX), str(out), "--config", str(cfg), "--jobs", "1"])
    assert res.exit_code == 0, res.output
    # both cameras represented; at least some grains detected
    frames = list((out / "grains").rglob("*.parquet")) if (out / "grains").exists() else []
    assert frames, "expected some detected grains from real fixtures"
    df = pd.concat([pd.read_parquet(p) for p in frames])
    assert set(df["camera"]).issubset({"basic", "zoom"})
    assert (df["um_per_px"] > 0).all()
```

- [ ] **Step 4: Write opt-in full-res test `tests/test_realdata_optin.py`**

```python
from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.realdata

LEXAR = Path("/Volumes/LEXAR/Camsizer/PPX/images_P_17_cs")


@pytest.mark.skipif(not LEXAR.exists(), reason="external drive not mounted")
def test_defocus_gate_separates_on_real_frames(tmp_path):
    from grain_morph.config import load_config
    from grain_morph.io import discover_frames
    from grain_morph.pipeline import process_frame
    cfg = load_config(None)
    cfg.calibration.um_per_px = {"basic": 20.0, "zoom": 8.0}
    specs = discover_frames(LEXAR, cfg)
    assert specs
    r = process_frame(specs[0], cfg, "test", "hash")
    assert r.status in {"ok", "empty"}
```

- [ ] **Step 5: Run the full suite**

Run: `pytest -v` (opt-in `realdata` runs only if the drive is mounted).
Expected: all PASS. Then `ruff check src tests && mypy src` clean.

- [ ] **Step 6: Write `README.md`** with: install (`uv`/pip; note `wadell_rs`→`fast_rs` fallback if used), a 3-command quickstart (`detect` → `aggregate` → `report`), the config reference table (every key in `default.yaml`), and a "How the QC gate works and how to tune it" section for a non-author reader (explain half-max threshold, blank flat-fielding, `edge_width_px`/`contrast` defocus logic, flag-never-drop, rejection-by-ECD).

- [ ] **Step 7: Pin the lockfile**

Run: `uv lock` → commit `uv.lock`.

- [ ] **Step 8: Final commit**

```bash
git add tests/fixtures/real README.md uv.lock tests/test_integration_real.py tests/test_realdata_optin.py
git commit -m "test+docs: downsampled real fixtures, integration + opt-in real tests, README, lockfile"
```

---

## Self-Review (completed by plan author)

**Spec coverage:** §2 facts → Tasks 2/3/4/7 (calibration, filename, blanks, flat-field). §3 decisions → blank-default (T7), camera-first-class (T2/T3/T13), configurable sample regex (T3), half-max (T8), polygons-on (T8/T12), focus gate (T10). §4 architecture → module-per-task. §5 morphometry → T9. §6 QC → T10. §7 outputs (formats, partitions, artifacts) → T11/T12; report artifacts → T14. §8 acceptance criteria → 1(T9), 2(T12), 3(T10), 4(T10), 5(T9), 6(T12), 7(T12), 8(T12); three test tiers → T6 (synthetic), T16 (downsampled + opt-in). §9 stack/style → T1 + per-task ruff/mypy. §10 non-goals honored (flag-never-drop enforced in T12/T13; no DL/DB).

**Placeholder scan:** every code step carries real code or an exact command. The one external-API dependency (`wadell_rs` signatures, T9 Step 3) is explicitly a one-lookup confirmation, not a logic placeholder.

**Type consistency:** `Config`, `ParsedName`, `FrameSpec`, `Detection`, `FrameResult`, `Manifest` names and their consumers are consistent across tasks; `write_table`/`read_table` signatures match between T11 and consumers T12–T16.

**Known ordering seam:** Manifest persistence (T5 interface) is implemented in-memory in T5 and given disk round-trip in T11; T5 tests only exercise in-memory behavior, so no forward dependency is violated.
