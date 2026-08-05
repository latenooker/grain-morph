from __future__ import annotations

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
