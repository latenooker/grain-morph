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
_TABLE_EXTENSIONS = {"parquet": ".parquet", "csv": ".csv", "feather": ".feather"}


def read_grains(path: str | Path, cfg: Config) -> pd.DataFrame | None:
    """Load a per-grain table from either a grains root or a single file.

    A `detect` run that found zero objects never creates `out_dir / "grains"`
    at all (see `pipeline.run_detect`) -- an ordinary, valid outcome, not an
    error. This returns `None` for that case (a missing `path`, or a grains
    directory with no per-frame table files under it) rather than raising, so
    callers can print a clear message and exit cleanly.

    Args:
        path: A grains root directory (e.g. `out_dir / "grains"`), or a single
            table file.
        cfg: Resolved configuration; `cfg.output.format` selects the per-frame
            file extension globbed for, and the format a single file is read as.

    Returns:
        The loaded per-grain table, or `None` if `path` doesn't exist or is a
        directory with no matching per-frame table files under it.
    """
    path = Path(path)
    if not path.exists():
        return None
    fmt = cfg.output.format
    if not path.is_dir():
        return read_table(path, fmt)
    if fmt == "parquet":
        # `pandas.read_parquet` on a directory transparently unions every part
        # file under it, hive-partitioned or not -- the one format whose leaf
        # files may *not* carry `sample_id`/`camera` inline (see
        # `writers.write_partitioned`), so only a whole-directory read
        # reconstructs them.
        if not any(path.rglob(f"*{_TABLE_EXTENSIONS[fmt]}")):
            return None
        return pd.read_parquet(path)
    # csv/feather grains directories are a flat set of per-frame files that each
    # carry every column inline, so unioning them is a per-file read + concat.
    files = sorted(path.rglob(f"*{_TABLE_EXTENSIONS[fmt]}"))
    if not files:
        return None
    return pd.concat([read_table(f, fmt) for f in files], ignore_index=True)


def read_contours(run_dir: str | Path, frame_ids: list[str], cfg: Config) -> pd.DataFrame:
    """Concatenate per-frame contour tables for the requested frames.

    Contours are never partitioned (`pipeline._write_or_clear_contour_frame`
    always writes `contours/{frame_id}.<ext>`), so each requested frame maps to
    exactly one candidate file path.

    Args:
        run_dir: Completed `detect` output directory (has `contours/`).
        frame_ids: Frame ids (stems) to read contours for.
        cfg: Resolved configuration; `cfg.output.format` selects the per-frame
            contour file's extension.

    Returns:
        `grain_uid, wkt, um_per_px` rows for every requested frame whose contour
        file exists; a frame with no contour file (`save_contours` off, or zero
        objects) is silently skipped. An empty, correctly-columned frame if none
        of the requested frames have one.
    """
    run_dir = Path(run_dir)
    ext = _TABLE_EXTENSIONS[cfg.output.format]
    frames = [
        read_table(path, cfg.output.format)
        for fid in frame_ids
        if (path := run_dir / "contours" / f"{fid}{ext}").exists()
    ]
    if not frames:
        return pd.DataFrame(columns=["grain_uid", "wkt", "um_per_px"])
    return pd.concat(frames, ignore_index=True)
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


def parse_name(
    path: str | Path,
    cfg: Config,
    *,
    sample_id: str | None = None,
    camera: str | None = None,
) -> ParsedName | None:
    """Parse sample/camera/frame identity out of a frame's filename.

    Resolution is **override-first, regex-second**. ``cfg.filename.sample_regex``
    is applied to the filename stem (basename without extension); each identity
    field then takes the explicit override when one is given, else the captured
    value. A file is a processable frame only if *both* ``sample_id`` and
    ``camera`` resolve to a non-``None`` value -- so with no overrides this
    reduces to the historical behavior (skip anything the regex doesn't match).

    When both overrides are supplied, a stem that the regex does *not* match is
    still processable: its blank/data status comes from ``cfg.filename.
    blank_regex`` and its frame index from the stem's trailing digits (this is
    the "all frames in one directory, no identity tokens in the filename" case).

    Args:
        path: Path (or bare filename) of the frame image.
        cfg: Resolved pipeline configuration providing ``filename`` rules.
        sample_id: If given, forces the sample id, ignoring any ``sample``
            capture. Supplied by ``detect``'s ``--sample-id`` override.
        camera: If given, forces the (already-mapped, e.g. ``"basic"``) camera,
            ignoring any ``cam`` capture. Supplied by ``detect``'s ``--camera``
            override.

    Returns:
        A :class:`ParsedName` when both sample and camera resolve, or ``None``
        (the file is skipped) otherwise.
    """
    stem = Path(path).stem
    match = re.match(cfg.filename.sample_regex, stem)
    groups = match.groupdict() if match is not None else {}

    resolved_sample = sample_id if sample_id is not None else groups.get("sample")
    if camera is not None:
        resolved_camera: str | None = camera
    elif groups.get("cam") is not None:
        resolved_camera = cfg.filename.camera_map[groups["cam"]]
    else:
        resolved_camera = None

    if resolved_sample is None or resolved_camera is None:
        return None

    if match is not None:
        frame_group = groups.get("frame")
        run = groups.get("run")
        is_blank = frame_group is None
        frame_index = int(frame_group) if frame_group is not None else None
    else:
        # Override path: regex didn't match, but the overrides supplied identity.
        run = None
        is_blank = re.search(cfg.filename.blank_regex, stem) is not None
        trailing = re.search(r"(\d+)$", stem)
        frame_index = int(trailing.group(1)) if trailing is not None else None

    return ParsedName(
        sample_id=resolved_sample,
        run=run,
        camera=resolved_camera,
        frame_index=frame_index,
        is_blank=is_blank,
        stem=stem,
    )


def discover_frames(
    root: str | Path,
    cfg: Config,
    *,
    sample_id: str | None = None,
    camera: str | None = None,
) -> list[FrameSpec]:
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
        sample_id: Optional sample-id override forwarded to :func:`parse_name`
            for every file (see its docstring); ``None`` keeps filename parsing.
        camera: Optional camera override forwarded to :func:`parse_name` for
            every file; ``None`` keeps filename parsing.

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
        parsed = parse_name(path, cfg, sample_id=sample_id, camera=camera)
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
