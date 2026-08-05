"""Filename parsing and (later) image I/O for the grain-morph pipeline.

Frame filenames encode a sample identifier, a camera code, and either a
frame index or a literal token marking a blank/background frame. The
regex and tokens used to parse this identity are configurable via
``cfg.filename`` (see :class:`grain_morph.config.FilenameConfig`) rather
than hardcoded here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from grain_morph.config import Config

_IMAGE_EXTENSIONS = (".bmp", ".png", ".tif", ".tiff")


@dataclass(frozen=True)
class ParsedName:
    """Identity information decoded from a frame filename.

    Attributes:
        sample_id: Sample identifier captured by the ``sample`` regex group.
        camera: Full camera name, mapped from the captured ``cam`` code via
            ``cfg.filename.camera_map`` (e.g. ``"basic"``/``"zoom"``).
        frame_index: Frame number, or ``None`` for blank/background frames.
        is_blank: Whether this filename identifies a blank/background frame.
        stem: The filename stem (basename without extension) that was
            parsed.
    """

    sample_id: str
    camera: str
    frame_index: int | None
    is_blank: bool
    stem: str


@dataclass(frozen=True)
class FrameSpec:
    """A discovered data frame, paired with its matching blank (if any).

    Attributes:
        path: Path to the data frame's image file.
        parsed: Identity decoded from ``path``'s filename.
        blank_path: Path to the blank/background frame sharing this frame's
            ``(sample_id, camera)``, or ``None`` if no such blank was found.
    """

    path: Path
    parsed: ParsedName
    blank_path: Path | None


def parse_name(path: str | Path, cfg: Config) -> ParsedName | None:
    """Parse sample/camera/frame identity out of a frame's filename.

    Applies ``cfg.filename.sample_regex`` to the filename stem (basename
    without extension), then maps the captured camera code through
    ``cfg.filename.camera_map``.

    Args:
        path: Path (or bare filename) of the frame image.
        cfg: Resolved pipeline configuration providing ``filename`` rules.

    Returns:
        A :class:`ParsedName` on a successful match, or ``None`` if the
        stem does not match ``cfg.filename.sample_regex``.
    """
    stem = Path(path).stem
    match = re.match(cfg.filename.sample_regex, stem)
    if match is None:
        return None

    groups = match.groupdict()
    frame_group = groups.get("frame")
    return ParsedName(
        sample_id=groups["sample"],
        camera=cfg.filename.camera_map[groups["cam"]],
        frame_index=int(frame_group) if frame_group is not None else None,
        is_blank=frame_group is None,
        stem=stem,
    )


def discover_frames(root: str | Path, cfg: Config) -> list[FrameSpec]:
    """Recursively discover data frames under ``root`` and pair them to blanks.

    Walks ``root`` for files with a known image extension
    (``_IMAGE_EXTENSIONS``), parses each filename via :func:`parse_name`,
    and discards files that don't match ``cfg.filename.sample_regex``.
    Each remaining data (non-blank) frame is paired with the blank frame
    that shares its ``(sample_id, camera)``, if one was found.

    Args:
        root: Directory to search recursively for image files.
        cfg: Resolved pipeline configuration providing ``filename`` rules.

    Returns:
        Data-frame specs sorted by ``(sample_id, camera, frame_index)``.
    """
    root = Path(root)
    paths = sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
    )

    data_frames: list[tuple[Path, ParsedName]] = []
    blanks: dict[tuple[str, str], Path] = {}
    for path in paths:
        parsed = parse_name(path, cfg)
        if parsed is None:
            continue
        if parsed.is_blank:
            blanks[(parsed.sample_id, parsed.camera)] = path
        else:
            data_frames.append((path, parsed))

    specs = [
        FrameSpec(
            path=path,
            parsed=parsed,
            blank_path=blanks.get((parsed.sample_id, parsed.camera)),
        )
        for path, parsed in data_frames
    ]
    specs.sort(key=lambda s: (s.parsed.sample_id, s.parsed.camera, s.parsed.frame_index))
    return specs
