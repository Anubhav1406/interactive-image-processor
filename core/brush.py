import cv2
import numpy as np


def blend_by_alpha(base, overlay, alpha):
    """Masked blend (alpha compositing) — the one formula behind every brush.

    Math
    ----
    For each pixel we linearly interpolate between the original and an
    "overlay" image, using the brush's coverage `alpha` as the mixing weight:

        out = alpha * overlay + (1 - alpha) * base            (alpha in [0, 1])

        alpha = 1  -> fully the overlay   (where you painted solidly)
        alpha = 0  -> fully the original  (where you didn't paint)
        0 < a < 1  -> a smooth blend      (soft, anti-aliased brush edges)

    The clever part is that changing only `overlay` gives you every brush:
        - paint      -> overlay = the coloured strokes
        - blur brush -> overlay = the blurred image
        - sharpen    -> overlay = the sharpened image
        - grayscale  -> overlay = the grayscale image
    So we compute the effect on the WHOLE image once, and the mask decides
    where it actually shows.

    Parameters
    ----------
    base, overlay : (H, W, 3) uint8 BGR images of the same size.
    alpha         : (H, W) float array in [0, 1] — the brush mask.
    """
    a = alpha[:, :, None]                       # (H, W, 1) to broadcast over BGR
    out = a * overlay.astype(np.float32) + (1 - a) * base.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def strokes_to_mask(strokes_rgba, size):
    """Turn the canvas' painted RGBA strokes into a full-resolution alpha mask.

    The drawable canvas returns an (h, w, 4) RGBA array at DISPLAY resolution;
    its 4th channel is opaque where the brush touched. We normalise that to
    [0, 1] and resize it up to the real image `size` = (width, height) so the
    mask lines up with the full-resolution photo.
    """
    alpha = strokes_rgba[:, :, 3].astype(np.float32) / 255.0
    return cv2.resize(alpha, size, interpolation=cv2.INTER_LINEAR)
