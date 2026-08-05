"""Flat-field correction for backlit silhouette frames.

Illumination is rarely perfectly uniform across a frame, so before
thresholding we divide out an estimate of the illumination field. Two field
estimators are supported: a measured blank (background-only) frame, and a
morphological background estimate (grey closing) that works even without a
blank, provided objects are small relative to the structuring element.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from grain_morph.config import Config


def _estimate_field_morphological(image: np.ndarray, kernel_px: int) -> np.ndarray:
    """Estimate the illumination field from the image itself via grey closing.

    Grey closing (dilation followed by erosion) fills in dark, compact
    features (objects) smaller than the structuring element while
    preserving the bright background level, making it a more correct
    background estimator than a uniform (mean) filter for dark-object-on
    -bright-background silhouettes. (Grey *opening* — erosion then
    dilation — instead removes small *bright* features and would leave
    dark objects untouched, so it is closing, not opening, that recovers
    the background here.)

    Args:
        image: Grayscale frame, any numeric dtype.
        kernel_px: Structuring-element size (pixels) passed to
            `scipy.ndimage.grey_closing` as `size`.

    Returns:
        Float32 array, same shape as `image`, holding the estimated
        illumination field.
    """
    field: np.ndarray = ndimage.grey_closing(image.astype(np.float32), size=kernel_px)
    return field


def _odd_guarded_kernel(kernel_px: int, image_shape: tuple[int, ...]) -> int:
    """Clamp a morphological kernel size to fit within a (small) test image.

    Args:
        kernel_px: Requested structuring-element size, px.
        image_shape: Shape of the image the kernel will be applied to.

    Returns:
        An odd, positive kernel size no larger than `min(image_shape) - 1`.
    """
    kernel = min(kernel_px, min(image_shape) - 1)
    kernel = max(kernel, 1)
    if kernel % 2 == 0:
        kernel -= 1
    return max(kernel, 1)


def apply_flatfield(
    image: np.ndarray, blank: np.ndarray | None, cfg: Config
) -> tuple[np.ndarray, str]:
    """Correct illumination non-uniformity by dividing out a field estimate.

    Selects the illumination field per `cfg.flatfield.method`: a measured
    blank frame (`"blank"`, or `"auto"` when `blank` is given), or a
    morphological estimate derived from `image` itself (`"morphological"`,
    or `"auto"` when no blank is available). The image is divided by the
    field and rescaled so the corrected background sits at ~1.0, leaving
    darker objects at values below 1.0.

    Args:
        image: Grayscale frame to correct, any numeric dtype.
        blank: Optional measured background-only frame, same shape as
            `image`.
        cfg: Resolved pipeline configuration; consumes `cfg.flatfield`.

    Returns:
        A `(corrected, method_used)` tuple: `corrected` is a float32 array
        normalized so the background is ~1.0, and `method_used` is
        `"blank"` or `"morphological"`.

    Raises:
        ValueError: If `cfg.flatfield.method == "blank"` but `blank` is
            `None`.
    """
    image_f32 = image.astype(np.float32)
    use_blank = cfg.flatfield.method == "blank" or (
        cfg.flatfield.method == "auto" and blank is not None
    )

    if use_blank:
        if blank is None:
            raise ValueError(
                "flatfield.method='blank' requires a blank frame, but none was "
                "provided/paired for this frame"
            )
        field = blank.astype(np.float32)
        method_used = "blank"
    else:
        kernel = _odd_guarded_kernel(cfg.flatfield.morph_kernel_px, image.shape)
        field = _estimate_field_morphological(image, kernel)
        method_used = "morphological"

    # Floor the divisor: the backlit background is bright, so a field pixel
    # can only be near-zero due to a degenerate blank/estimate, never a
    # legitimate background reading. This avoids a divide-by-near-zero blowup
    # without introducing a tunable threshold.
    field = np.maximum(field, 1.0)
    corrected = image_f32 / field
    corrected = np.clip(corrected, 0.0, None)
    corrected = corrected / float(np.median(corrected))
    return corrected.astype(np.float32), method_used
