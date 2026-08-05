from __future__ import annotations

import numpy as np
import shapely
from shapely.geometry import Polygon

from grain_morph.config import load_config
from grain_morph.detect import _repair_polygon, detect_objects, threshold_level
from grain_morph.flatfield import apply_flatfield
from tests.synth import make_frame


def _corrected(objs, size=(256, 256), gradient=0.0):
    cfg = load_config(None)
    f = make_frame(size=size, objects=objs, gradient=gradient)
    corrected, _ = apply_flatfield(f.image, None, cfg)
    return corrected, cfg, f


def test_threshold_between_core_and_background():
    corrected, cfg, _ = _corrected(
        [
            {
                "kind": "ellipse",
                "cx": 128,
                "cy": 128,
                "a": 20,
                "b": 20,
                "angle": 0.0,
                "blur_sigma": 0.0,
            }
        ]
    )
    lvl = threshold_level(corrected, cfg)
    assert 0.3 < lvl < 0.95


def test_detects_expected_object_count():
    corrected, cfg, _ = _corrected(
        [
            {
                "kind": "ellipse",
                "cx": 80,
                "cy": 80,
                "a": 18,
                "b": 18,
                "angle": 0.0,
                "blur_sigma": 0.0,
            },
            {
                "kind": "ellipse",
                "cx": 180,
                "cy": 180,
                "a": 14,
                "b": 14,
                "angle": 0.0,
                "blur_sigma": 0.0,
            },
        ]
    )
    _, dets = detect_objects(corrected, cfg)
    assert len(dets) == 2
    assert all(d.polygon is not None and d.contour_ok for d in dets)


def test_polygon_area_close_to_ground_truth():
    r = 25
    corrected, cfg, f = _corrected(
        [{"kind": "ellipse", "cx": 128, "cy": 128, "a": r, "b": r, "angle": 0.0, "blur_sigma": 0.0}]
    )
    _, dets = detect_objects(corrected, cfg)
    truth = f.objects[0].polygon.area
    got = dets[0].polygon.area
    assert abs(got - truth) / truth < 0.02  # subpixel contour, tight


def test_small_object_removed_by_min_area():
    cfg = load_config(None)
    cfg.detect.min_area_px = 2000
    corrected, _, _ = _corrected(
        [{"kind": "ellipse", "cx": 128, "cy": 128, "a": 5, "b": 5, "angle": 0.0, "blur_sigma": 0.0}]
    )
    _, dets = detect_objects(corrected, cfg)
    assert dets == []


def test_area_accurate_on_clean_frame():
    r = 40
    corrected, cfg, f = _corrected(
        [{"kind": "ellipse", "cx": 128, "cy": 128, "a": r, "b": r, "angle": 0.0, "blur_sigma": 0.0}]
    )
    _, dets = detect_objects(corrected, cfg)
    truth = f.objects[0].polygon.area
    got = dets[0].polygon.area
    assert abs(got - truth) / truth < 0.01


def test_area_accurate_under_residual_gradient():
    # Regression: an Otsu-based dark-pixel split (not a fixed median cut) is
    # required so a residual illumination gradient after morphological
    # flat-fielding doesn't drag the half-max core estimate toward
    # background and inflate the recovered area.
    r = 40
    corrected, cfg, f = _corrected(
        [
            {
                "kind": "ellipse",
                "cx": 128,
                "cy": 128,
                "a": r,
                "b": r,
                "angle": 0.0,
                "blur_sigma": 0.0,
            }
        ],
        gradient=0.2,
    )
    _, dets = detect_objects(corrected, cfg)
    assert len(dets) == 1
    truth = f.objects[0].polygon.area
    got = dets[0].polygon.area
    assert abs(got - truth) / truth < 0.03


def test_repair_polygon_valid_polygon_passes_through_unchanged():
    # Already-valid, positive-area polygons must be returned as-is: the
    # 1%-area accuracy tests rely on valid polygons measuring unchanged.
    valid = shapely.Point(0, 0).buffer(10, quad_segs=16)
    assert _repair_polygon(valid) is valid


def test_repair_polygon_self_intersecting_exterior_is_repaired():
    # Real failure mode 1: a coarse downsampled marching-squares trace can
    # self-intersect (a "bowtie" ring). shapely reports its raw `.area`
    # as 0 for this case (the two lobes' signed areas cancel), which
    # would otherwise reach `measure_polygon` as a spuriously-empty
    # object instead of the true ~50-unit combined outline.
    bowtie = Polygon([(0, 0), (10, 10), (10, 0), (0, 10)])
    assert not bowtie.is_valid
    assert bowtie.area == 0.0

    repaired = _repair_polygon(bowtie)

    assert repaired is not None
    assert repaired.is_valid
    assert repaired.area > 0.0


def test_repair_polygon_oversized_hole_is_repaired():
    # Real failure mode 2: a mis-associated / oversized interior ring
    # (from `_assign_rings_to_labels` grabbing a neighboring object's
    # contour) drives `exterior_area - hole_area` negative -- this is
    # exactly what later makes `measure_polygon`'s `math.sqrt` raise
    # `ValueError: math domain error` if it isn't repaired first.
    shell = [(0, 0), (10, 0), (10, 10), (0, 10)]
    oversized_hole = [(-5, -5), (15, -5), (15, 15), (-5, 15)]
    poly = Polygon(shell, [oversized_hole])
    assert not poly.is_valid
    assert poly.area < 0.0

    repaired = _repair_polygon(poly)

    assert repaired is not None
    assert repaired.is_valid
    assert repaired.area > 0.0


def test_repair_polygon_unrecoverable_degenerate_ring_returns_none():
    # A ring with no interior at all (collinear points) has nothing for
    # `make_valid` to recover a polygonal component from -- it should be
    # dropped (`None`), not raise, so the caller can flag the detection
    # `contour_ok=False` instead of crashing on it.
    collinear = Polygon([(0, 0), (1, 0), (2, 0)])
    assert _repair_polygon(collinear) is None


def test_object_free_noise_frame_detects_nothing():
    # A flat-fielded frame with NO opaque objects -- just sensor noise around
    # the background level of 1.0 -- must yield zero detections. Otsu always
    # splits the histogram, so without the object-presence gate it would split
    # the noise and trace hundreds of specks. Backlit grains are near-opaque,
    # so a real core sits far below background; pure noise never does.
    rng = np.random.default_rng(0)
    corrected = (1.0 + rng.normal(0.0, 0.03, size=(256, 256))).astype(np.float32)
    cfg = load_config(None)
    level = threshold_level(corrected, cfg)
    assert level < float(corrected.min())  # sentinel: selects no foreground
    _, dets = detect_objects(corrected, cfg)
    assert dets == []


def test_small_object_amid_noise_survives_gate():
    # The gate must be count-based, not percentile-based: a single small but
    # genuinely-opaque grain among heavy noise contributes enough opaque
    # pixels to survive, even though a percentile of the dark subset would be
    # noise-dominated and average it away.
    cfg = load_config(None)
    obj = {"kind": "ellipse", "cx": 128, "cy": 128, "a": 10, "b": 9,
           "angle": 0.0, "blur_sigma": 0.0}
    f = make_frame(size=(256, 256), noise_sigma=3.0, objects=[obj])
    corrected, _ = apply_flatfield(f.image, None, cfg)
    _, dets = detect_objects(corrected, cfg)
    assert len(dets) == 1
