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
from typing import Any

import pandas as pd

from grain_morph.config import Config
from grain_morph.writers import read_table, write_table

_IMAGE_EXTENSIONS = (".bmp", ".png", ".tif", ".tiff")
_MANIFEST_COLUMNS = (
    "frame_id",
    "frame_path",
    "fingerprint",
    "status",
    "n_objects",
    "seconds",
    "flatfield_method",
)


@dataclass(frozen=True)
class ParsedName:
    """Identity information decoded from a frame filename.

    Attributes:
        sample_id: Sample identifier captured by the ``sample`` regex group.
        run: Run/measurement identifier captured by an optional ``run`` regex
            group, or ``None`` when ``sample_regex`` defines no ``run`` group
            (the packaged default folds any run token into ``sample_id``
            instead). When present, it makes identity — and blank pairing —
            run-scoped, so multiple runs of one sample+camera in a single
            discovery root don't collide.
        camera: Full camera name, mapped from the captured ``cam`` code via
            ``cfg.filename.camera_map`` (e.g. ``"basic"``/``"zoom"``).
        frame_index: Frame number, or ``None`` for blank/background frames.
        is_blank: Whether this filename identifies a blank/background frame.
        stem: The filename stem (basename without extension) that was
            parsed.
    """

    sample_id: str
    run: str | None
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
            ``(sample_id, camera, run)``, or ``None`` if no such blank was
            found. Including ``run`` keeps pairing correct when a discovery
            root spans multiple runs of one sample+camera (see
            :class:`ParsedName`); with the default (run-less) schema ``run`` is
            ``None`` for every frame, so this reduces to ``(sample_id, camera)``.
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
        run=groups.get("run"),
        camera=cfg.filename.camera_map[groups["cam"]],
        frame_index=int(frame_group) if frame_group is not None else None,
        is_blank=frame_group is None,
        stem=stem,
    )


def discover_frames(root: str | Path, cfg: Config) -> list[FrameSpec]:
    """Recursively discover data frames under ``root`` and pair them to blanks.

    Walks ``root`` for files with a known image extension
    (``_IMAGE_EXTENSIONS``), skipping hidden files (name starts with
    ``.``) -- notably macOS AppleDouble sidecar files (e.g.
    ``._frame.bmp``), which `Path.rglob` would otherwise happily return
    and which `sample_regex` can accidentally match (its leading
    ``.+?`` group readily swallows a ``._`` prefix as part of the
    sample id), only for image decoding to fail later since the
    sidecar isn't real image data. These sidecars are routinely written
    by macOS when copying onto non-HFS+ volumes (e.g. the exFAT
    external drives this pipeline reads CAMSIZER output from), so this
    is an expected condition, not a hypothetical one. Each remaining
    file is parsed via :func:`parse_name`, which discards files that
    don't match ``cfg.filename.sample_regex``. Each remaining data
    (non-blank) frame is paired with the blank frame that shares its
    ``(sample_id, camera, run)``, if one was found -- so a discovery root that
    mixes multiple runs of one sample+camera pairs each run's frames to that
    run's own blank rather than to whichever blank was walked last.

    Args:
        root: Directory to search recursively for image files.
        cfg: Resolved pipeline configuration providing ``filename`` rules.

    Returns:
        Data-frame specs sorted by ``(sample_id, run, camera, frame_index)``.
    """
    root = Path(root)
    paths = sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in _IMAGE_EXTENSIONS
    )

    data_frames: list[tuple[Path, ParsedName]] = []
    blanks: dict[tuple[str, str | None, str], Path] = {}
    for path in paths:
        parsed = parse_name(path, cfg)
        if parsed is None:
            continue
        if parsed.is_blank:
            blanks[(parsed.sample_id, parsed.run, parsed.camera)] = path
        else:
            data_frames.append((path, parsed))

    specs = [
        FrameSpec(
            path=path,
            parsed=parsed,
            blank_path=blanks.get((parsed.sample_id, parsed.run, parsed.camera)),
        )
        for path, parsed in data_frames
    ]
    def _sort_key(s: FrameSpec) -> tuple[str, str, str, int | None]:
        p = s.parsed
        return (p.sample_id, p.run or "", p.camera, p.frame_index)

    specs.sort(key=_sort_key)
    return specs


def frame_fingerprint(path: Path) -> str:
    """Compute a cheap, stable fingerprint for a frame file.

    Combines file size and (truncated) modification time from
    ``path.stat()`` so that resume logic can detect whether a frame has
    changed on disk since it was last processed, without hashing file
    contents.

    Args:
        path: Path to the frame image file.

    Returns:
        A ``"{size}:{int(mtime)}"`` string.
    """
    stat = path.stat()
    return f"{stat.st_size}:{int(stat.st_mtime)}"


class Manifest:
    """In-memory record of per-frame processing results, keyed by frame ID.

    Used to support resumable pipeline runs: a frame is considered done
    only if it was previously recorded *and* its stored fingerprint still
    matches the current one (see :meth:`is_done`).
    """

    def __init__(self) -> None:
        """Initialize an empty manifest."""
        self._rows: dict[str, dict[str, Any]] = {}

    def add(
        self,
        frame_id: str,
        frame_path: str | Path,
        status: str,
        n_objects: int,
        seconds: float,
        flatfield_method: str,
        *,
        fingerprint: str,
    ) -> None:
        """Record (or overwrite) the processing result for a frame.

        Args:
            frame_id: Unique identifier for the frame (typically its
                filename stem).
            frame_path: Path to the frame's image file.
            status: Outcome of processing (e.g. ``"ok"``, ``"error"``).
            n_objects: Number of objects detected in the frame.
            seconds: Wall-clock processing time, in seconds.
            flatfield_method: Name of the flatfield correction method used.
            fingerprint: Value from :func:`frame_fingerprint` for
                ``frame_path`` at the time it was processed.
        """
        self._rows[frame_id] = {
            "frame_id": frame_id,
            "frame_path": str(frame_path),
            "fingerprint": fingerprint,
            "status": status,
            "n_objects": n_objects,
            "seconds": seconds,
            "flatfield_method": flatfield_method,
        }

    def is_done(self, frame_id: str, fingerprint: str) -> bool:
        """Check whether a frame has already been processed and is unchanged.

        Args:
            frame_id: Frame identifier to look up.
            fingerprint: Current fingerprint of the frame's file, from
                :func:`frame_fingerprint`.

        Returns:
            ``True`` if ``frame_id`` is recorded and its stored fingerprint
            equals ``fingerprint``; ``False`` otherwise (including when
            ``frame_id`` is unknown).
        """
        row = self._rows.get(frame_id)
        return row is not None and row["fingerprint"] == fingerprint

    def to_frame(self) -> pd.DataFrame:
        """Materialize the manifest as a :class:`pandas.DataFrame`.

        Returns:
            A DataFrame with columns ``frame_id, frame_path, fingerprint,
            status, n_objects, seconds, flatfield_method``, one row per
            recorded frame.
        """
        return pd.DataFrame(list(self._rows.values()), columns=list(_MANIFEST_COLUMNS))

    def save(self, path: str | Path, fmt: str) -> Path:
        """Persist the manifest to disk via :func:`grain_morph.writers.write_table`.

        Args:
            path: Destination path. The correct extension for ``fmt`` is
                appended if it isn't already present (see
                :func:`grain_morph.writers.write_table`).
            fmt: One of ``"parquet"``, ``"csv"``, ``"feather"``.

        Returns:
            The actual path written.
        """
        return write_table(self.to_frame(), Path(path), fmt)

    @classmethod
    def load(cls, path: str | Path, fmt: str) -> Manifest:
        """Reconstruct a manifest previously written by :meth:`save`.

        Args:
            path: Path to the manifest table, as returned by :meth:`save`.
            fmt: One of ``"parquet"``, ``"csv"``, ``"feather"``.

        Returns:
            A new :class:`Manifest` whose rows reproduce those in the
            saved table (``to_frame()`` on the result is equal to
            ``to_frame()`` on the manifest that was saved).
        """
        frame = read_table(Path(path), fmt)
        manifest = cls()
        for row in frame.to_dict("records"):
            manifest.add(
                str(row["frame_id"]),
                str(row["frame_path"]),
                str(row["status"]),
                int(row["n_objects"]),
                float(row["seconds"]),
                str(row["flatfield_method"]),
                fingerprint=str(row["fingerprint"]),
            )
        return manifest
