from __future__ import annotations

import numpy as np

from grain_morph.config import load_config
from grain_morph.detect import detect_objects
from grain_morph.flatfield import apply_flatfield
from grain_morph.qc import qc_flags, qc_metrics
from tests.synth import make_frame


def _measure_flags(objs, size=(256, 256)):
    cfg = load_config(None)
    f = make_frame(size=size, objects=objs)
    corrected, _ = apply_flatfield(f.image, None, cfg)
    labels, dets = detect_objects(corrected, cfg)
    out = []
    for d in dets:
        mask = labels == d.label
        m = qc_metrics(
            f.image.astype(float), corrected, d.polygon, mask, d.mask_bbox, f.image.shape, cfg
        )
        out.append((m, qc_flags(m, cfg)))
    return out, f


def test_border_object_flagged():
    # ACCEPTANCE CRITERION 4
    res, _ = _measure_flags(
        [{"kind": "rect", "cx": 3, "cy": 128, "a": 30, "b": 20, "angle": 0.0, "blur_sigma": 0.0}]
    )
    assert res[0][1]["flag_border"] is True


def test_defocus_recall_and_fp_rate():
    # ACCEPTANCE CRITERION 3 (aggregate over many objects)
    rng = np.random.default_rng(0)
    sharp_fp = 0
    blur_tp = 0
    n = 30
    for _i in range(n):
        cx = int(rng.integers(60, 196))
        cy = int(rng.integers(60, 196))
        sharp, _ = _measure_flags(
            [
                {
                    "kind": "ellipse",
                    "cx": cx,
                    "cy": cy,
                    "a": 20,
                    "b": 18,
                    "angle": 0.0,
                    "blur_sigma": 0.0,
                }
            ]
        )
        if sharp and sharp[0][1]["flag_defocus"]:
            sharp_fp += 1
        blur, _ = _measure_flags(
            [
                {
                    "kind": "ellipse",
                    "cx": cx,
                    "cy": cy,
                    "a": 20,
                    "b": 18,
                    "angle": 0.0,
                    "blur_sigma": 5.0,
                }
            ]
        )
        if blur and blur[0][1]["flag_defocus"]:
            blur_tp += 1
    assert blur_tp / n >= 0.95  # recall
    assert sharp_fp / n <= 0.02  # false-positive rate
