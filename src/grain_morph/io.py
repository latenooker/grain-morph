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
