from __future__ import annotations

import numpy as np
import pytest

from grain_morph.config import load_config
from grain_morph.flatfield import apply_flatfield
from tests.synth import make_frame


def test_blank_division_flattens_background():
    cfg = load_config(None)
    cfg.flatfield.method = "blank"
    f = make_frame(size=(128, 128), objects=[], gradient=0.4)
    blank = f.image.copy()  # empty frame IS the illumination field
    corrected, method = apply_flatfield(f.image, blank, cfg)
    assert method == "blank"
    # background now uniform: std across the frame is tiny
    assert corrected.std() < 0.02
    assert abs(float(np.median(corrected)) - 1.0) < 0.02


def test_morphological_used_when_no_blank():
    cfg = load_config(None)
    cfg.flatfield.method = "auto"
    f = make_frame(
        size=(256, 256),
        objects=[
            {
                "kind": "ellipse",
                "cx": 128,
                "cy": 128,
                "a": 15,
                "b": 15,
                "angle": 0.0,
                "blur_sigma": 0.0,
            }
        ],
    )
    corrected, method = apply_flatfield(f.image, None, cfg)
    assert method == "morphological"
    # object still darker than corrected background
    assert corrected.min() < 0.6
    assert abs(float(np.median(corrected)) - 1.0) < 0.05


def test_forced_blank_method_without_blank_raises():
    cfg = load_config(None)
    cfg.flatfield.method = "blank"
    f = make_frame(size=(128, 128), objects=[], gradient=0.4)
    with pytest.raises(ValueError, match="requires a blank frame"):
        apply_flatfield(f.image, None, cfg)
