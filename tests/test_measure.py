from __future__ import annotations

import math

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
    obj = {"kind": "ellipse", "cx": 128, "cy": 128, "a": r, "b": r, "angle": 0.0, "blur_sigma": 0.0}
    f = make_frame(size=(256, 256), objects=[obj])
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
    obj = {
        "kind": "ellipse", "cx": 128, "cy": 128, "a": 40, "b": 40, "angle": 0.0, "blur_sigma": 0.0
    }
    f = make_frame(size=(256, 256), objects=[obj])
    corrected, _ = apply_flatfield(f.image, None, cfg)
    labels, dets = detect_objects(corrected, cfg)
    poly_perim = measure_polygon(dets[0].polygon, 1.0)["perimeter_um"]
    raster_perim = raster_perimeter_px(labels == dets[0].label)
    truth = f.objects[0].polygon.length
    assert poly_perim < raster_perim
    assert abs(poly_perim - truth) < abs(raster_perim - truth)


def test_efd_and_wadell_smoke():
    from grain_morph.measure import measure_efd, measure_wadell

    circle = shapely.Point(0, 0).buffer(50, quad_segs=128)
    efd, cum90 = measure_efd(circle, order=15, resample_n=256)
    assert len(efd) == 15 * 4
    assert 1 <= cum90 <= 15
    w = measure_wadell(circle, smoothing=1.0)
    assert 0.0 < w["wadell_roundness"] <= 1.2
    assert 0.0 < w["wadell_sphericity"] <= 1.2
