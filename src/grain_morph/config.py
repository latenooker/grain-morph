"""Pydantic configuration models, YAML load/merge, and config hashing.

All pipeline thresholds, kernel sizes, and other tunables live in
``configs/default.yaml`` — never as literals in source. :func:`load_config`
loads that packaged default and deep-merges an optional user YAML file on
top of it, producing a validated :class:`Config`.
"""

from __future__ import annotations

import hashlib
import importlib.resources
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict


class _StrictModel(BaseModel):
    """Base model that rejects unknown keys, so config typos fail loudly."""

    model_config = ConfigDict(extra="forbid")


class CalibrationConfig(_StrictModel):
    """Per-camera micrometers-per-pixel calibration.

    Attributes:
        um_per_px: Camera name -> microns per pixel, or ``None`` if that
            camera has not yet been calibrated.
    """

    um_per_px: dict[str, float | None]


class FilenameConfig(_StrictModel):
    """Rules for parsing sample/camera/frame identity out of filenames.

    Attributes:
        sample_regex: Regex applied to a frame's filename stem, capturing
            named groups ``sample``, ``cam``, and either ``frame`` or the
            literal ``back`` (a blank/background frame).
        camera_map: Single-letter camera code (as captured by ``cam``) ->
            full camera name (matching the keys of
            ``calibration.um_per_px``).
    """

    sample_regex: str
    camera_map: dict[str, str]


class FlatfieldConfig(_StrictModel):
    """Flat-field correction settings.

    Attributes:
        method: ``"auto"`` picks blank-division when a blank frame is
            available and falls back to ``"morphological"`` otherwise;
            ``"blank"`` and ``"morphological"`` force one method.
        morph_kernel_px: Structuring-element size (pixels) for the
            morphological background estimate.
    """

    method: Literal["auto", "blank", "morphological"]
    morph_kernel_px: int


class ThresholdConfig(_StrictModel):
    """Grayscale thresholding settings applied to the flat-fielded image.

    Attributes:
        method: ``"half_max"`` or ``"otsu"``.
        half_max_fraction: Fraction of the background-to-core intensity
            range used as the half-max threshold level.
        core_percentile: Dark-pixel percentile used to estimate the opaque
            core intensity for the half-max method.
        min_object_depth: Fractional darkening below the flat-fielded
            background (1.0) at or beyond which a pixel counts as opaque
            object-core. Backlit grains are near-opaque, so a real object
            contributes many such pixels; an object-free frame has essentially
            none, even after blank-division amplifies sensor noise. A frame
            with fewer than ``detect.min_area_px`` opaque pixels is treated as
            object-free and ``half_max`` returns a level that detects nothing
            (guards against Otsu tracing noise on empty frames).
    """

    method: Literal["half_max", "otsu"]
    half_max_fraction: float
    core_percentile: float
    min_object_depth: float


class DetectConfig(_StrictModel):
    """Object detection settings.

    Attributes:
        min_area_px: Minimum connected-component area (pixels) to keep.
        fill_holes: Whether to fill interior holes before labeling.
    """

    min_area_px: int
    fill_holes: bool


class MeasureConfig(_StrictModel):
    """Morphometry settings.

    Attributes:
        efd_order: Number of elliptic Fourier descriptor harmonics.
        efd_resample_n: Number of points the boundary is resampled to
            before computing EFDs.
        wadell_smoothing: Smoothing factor applied before Wadell
            roundness/circularity estimation.
        curvature_resample_n: Number of points the boundary is resampled
            to before computing curvature entropy.
        curvature_smoothing: Gaussian smoothing sigma (in resampled-point
            units) applied to the boundary before differentiating for
            curvature entropy.
        curvature_bins: Number of histogram bins used to estimate the
            curvature distribution for curvature entropy.
    """

    efd_order: int
    efd_resample_n: int
    wadell_smoothing: float
    curvature_resample_n: int
    curvature_smoothing: float
    curvature_bins: int


class QCConfig(_StrictModel):
    """Quality-control thresholds used to flag (never drop) detections.

    Attributes:
        min_ecd_px: Minimum equivalent circular diameter (pixels).
        defocus_edge_width_px: Edge-width values above this indicate
            defocus.
        defocus_contrast_min: Contrast values below this indicate defocus.
        sliver_aspect_ratio: Aspect ratio above which an object may be
            flagged a sliver.
        sliver_max_ecd_px: Only objects at or below this ECD are eligible
            for the sliver flag.
        agglomerate_solidity_max: Solidity below this indicates a probable
            agglomerate.
        disqualifying_flags: Flag names that ``aggregate`` treats as
            disqualifying by default.
    """

    min_ecd_px: float
    defocus_edge_width_px: float
    defocus_contrast_min: float
    sliver_aspect_ratio: float
    sliver_max_ecd_px: float
    agglomerate_solidity_max: float
    disqualifying_flags: list[str]


class OutputConfig(_StrictModel):
    """Output table settings.

    Attributes:
        format: Tabular output format: ``"parquet"``, ``"csv"``, or
            ``"feather"``.
        save_contours: Whether to persist subpixel boundary polygons.
        partition: Whether to partition output by ``(sample_id, camera)``.
    """

    format: Literal["parquet", "csv", "feather"]
    save_contours: bool
    partition: bool


class RuntimeConfig(_StrictModel):
    """Execution/parallelization settings for `grain_morph.pipeline.run_detect`.

    Attributes:
        n_jobs: Number of worker processes `joblib.Parallel` uses to
            process frames (`-1` = all cores). `run_detect`'s `n_jobs`
            argument (and the CLI `--jobs` flag) overrides this per call.
    """

    n_jobs: int


class Config(_StrictModel):
    """Fully resolved, validated grain-morph pipeline configuration.

    Attributes:
        calibration: Per-camera micrometers-per-pixel calibration.
        filename: Filename-parsing rules.
        flatfield: Flat-field correction settings.
        threshold: Thresholding settings.
        detect: Object detection settings.
        measure: Morphometry settings.
        qc: Quality-control thresholds.
        output: Output table settings.
        runtime: Execution/parallelization settings.
    """

    calibration: CalibrationConfig
    filename: FilenameConfig
    flatfield: FlatfieldConfig
    threshold: ThresholdConfig
    detect: DetectConfig
    measure: MeasureConfig
    qc: QCConfig
    output: OutputConfig
    runtime: RuntimeConfig

    def um_per_px(self, camera: str) -> float:
        """Look up the calibrated micrometers-per-pixel scale for a camera.

        Args:
            camera: Camera name (e.g. ``"basic"`` or ``"zoom"``).

        Returns:
            The calibrated micrometers-per-pixel value.

        Raises:
            KeyError: If ``camera`` has no calibration entry, or its
                calibration is still ``null``.
        """
        value = self.calibration.um_per_px.get(camera)
        if value is None:
            raise KeyError(
                f"No calibration for camera '{camera}' — set calibration.um_per_px.{camera}"
            )
        return value


def _load_default_dict() -> dict[str, Any]:
    """Read the packaged ``configs/default.yaml`` into a plain dict.

    Returns:
        The parsed default configuration.
    """
    resource = importlib.resources.files("grain_morph.configs") / "default.yaml"
    data = yaml.safe_load(resource.read_text())
    return data if isinstance(data, dict) else {}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto ``base`` without mutating either.

    Args:
        base: The base mapping (e.g. packaged defaults).
        override: The mapping to layer on top; wins on conflicts.

    Returns:
        A new mapping with ``override`` deep-merged onto ``base``.
    """
    merged = dict(base)
    for key, value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(base_value, value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path | None) -> Config:
    """Load, merge, and validate the pipeline configuration.

    Reads the packaged ``configs/default.yaml``, then — if ``path`` is
    given — deep-merges the user YAML at ``path`` on top of it (user values
    win; unspecified keys keep their default).

    Args:
        path: Path to a user override YAML file, or ``None`` to use only
            the packaged defaults.

    Returns:
        The validated, fully resolved :class:`Config`.
    """
    merged = _load_default_dict()
    if path is not None:
        user_dict = yaml.safe_load(Path(path).read_text()) or {}
        merged = _deep_merge(merged, user_dict)
    return Config(**merged)


def config_hash(cfg: Config) -> str:
    """Compute a short, deterministic hash of a resolved config.

    Args:
        cfg: The config to hash.

    Returns:
        The first 12 hex characters of the sha256 digest of the config's
        sorted JSON representation.
    """
    payload = json.dumps(cfg.model_dump(), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]
