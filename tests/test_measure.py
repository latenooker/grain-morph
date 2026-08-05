from __future__ import annotations

import math

import numpy as np
import pytest
import shapely
from skimage.draw import disk

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


def test_measure_wadell_uses_largest_component_not_first_label():
    # FIX ROUND 1, item 1: a stray fragment that appears *earlier* in
    # connected-component scan order than the true main blob must not be
    # picked instead of it.
    from grain_morph.measure import measure_wadell

    main = shapely.Point(0, 0).buffer(30, quad_segs=64)
    stray = shapely.Point(0, -45).buffer(1.2, quad_segs=16)
    # A sub-pixel-wide bridge keeps `combo` one valid simple Polygon (no
    # self-intersection) while its *rasterization* still fragments into
    # two disconnected components: the bridge's x-extent is chosen so it
    # never straddles an integer raster column, so no raster pixel along
    # its length is ever filled. The stray blob sits at smaller row
    # indices (more negative y) than the main blob, so it is labeled
    # first in `skimage.measure.label`'s (row-major) scan order — picking
    # connected-component list index 0 without an explicit largest-area
    # check would silently grab the ~5 px stray fragment instead of the
    # ~2800 px main blob.
    bridge = shapely.Polygon([(0.35, -29.0), (0.65, -29.0), (0.65, -44.5), (0.35, -44.5)])
    combo = shapely.union_all([main, stray, bridge])
    assert isinstance(combo, shapely.Polygon)  # one simple polygon, not a MultiPolygon

    combo_result = measure_wadell(combo, smoothing=1.0)
    clean_result = measure_wadell(main, smoothing=1.0)

    assert combo_result["wadell_roundness"] == pytest.approx(
        clean_result["wadell_roundness"], abs=0.05
    )
    assert combo_result["wadell_sphericity"] == pytest.approx(
        clean_result["wadell_sphericity"], abs=0.05
    )


def test_perimeter_crofton_px_smoke():
    # FIX ROUND 1, item 2
    from grain_morph.measure import perimeter_crofton_px

    r = 50
    mask = np.zeros((2 * r + 20, 2 * r + 20), dtype=bool)
    rr, cc = disk((r + 10, r + 10), r, shape=mask.shape)
    mask[rr, cc] = True

    crofton = perimeter_crofton_px(mask)
    truth = 2 * math.pi * r
    assert math.isfinite(crofton)
    assert abs(crofton - truth) / truth < 0.03
