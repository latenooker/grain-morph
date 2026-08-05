from __future__ import annotations

import numpy as np

from tests.synth import make_frame


def test_frame_shape_and_dtype():
    f = make_frame(
        size=(128, 128),
        objects=[
            {
                "kind": "ellipse",
                "cx": 64,
                "cy": 64,
                "a": 20,
                "b": 15,
                "angle": 0.0,
                "blur_sigma": 0.0,
            }
        ],
    )
    assert f.image.shape == (128, 128)
    assert f.image.dtype == np.uint8
    assert len(f.objects) == 1


def test_objects_are_darker_than_background():
    f = make_frame(
        size=(128, 128),
        objects=[
            {
                "kind": "ellipse",
                "cx": 64,
                "cy": 64,
                "a": 20,
                "b": 15,
                "angle": 0.0,
                "blur_sigma": 0.0,
            }
        ],
    )
    ys, xs = np.nonzero(f.image < 100)
    assert xs.size > 0
    assert abs(xs.mean() - 64) < 5 and abs(ys.mean() - 64) < 5


def test_gradient_makes_right_side_darker_background():
    f = make_frame(size=(128, 128), objects=[], gradient=0.4)
    left = f.image[:, :10].mean()
    right = f.image[:, -10:].mean()
    assert left - right > 20  # 40% gradient is visible


def test_border_object_flag_and_blur_flag():
    f = make_frame(
        size=(128, 128),
        objects=[
            {"kind": "rect", "cx": 4, "cy": 64, "a": 20, "b": 10, "angle": 0.0, "blur_sigma": 0.0},
            {
                "kind": "ellipse",
                "cx": 90,
                "cy": 40,
                "a": 12,
                "b": 12,
                "angle": 0.0,
                "blur_sigma": 5.0,
            },
        ],
    )
    assert any(o.touches_border for o in f.objects)
    assert any(o.blurred for o in f.objects)
