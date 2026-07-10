import cv2
import numpy as np


def apply_hist_eq(img):
    """Global histogram equalization — spreads tones to use the full range.

    Math
    ----
    Build the histogram h(i) = number of pixels with brightness i (0..255).
    Turn it into a cumulative distribution (CDF):

        cdf(i) = sum of h(0..i)

    Then remap every brightness through the normalised CDF:

        out = round( 255 * (cdf(in) - cdf_min) / (N - cdf_min) )

    Brightness levels where many pixels pile up get stretched apart (more
    contrast there), while sparse levels get squeezed, so a flat, gray,
    under-exposed photo gains contrast.

    We equalize only the LUMA (Y) channel of YCrCb so colours are preserved;
    equalizing B, G, R separately would shift the hue.
    """
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def apply_clahe(img, clip_limit, tile=8):
    """CLAHE — Contrast Limited Adaptive Histogram Equalization.

    Math
    ----
    Plain equalization uses one histogram for the whole image, which can
    blow out faces. CLAHE fixes two things:

      1. ADAPTIVE: split the image into a grid of `tile x tile` cells and
         equalize each cell using its OWN local CDF, then blend cell borders
         smoothly (bilinear) so no seams appear.
      2. CONTRAST LIMITED: before building each cell's CDF, any histogram bin
         taller than `clip_limit` is clipped and the excess is spread evenly
         across all bins. This caps how aggressively noise gets amplified.

    Again applied to the Y (luma) channel only, so colour is untouched.
    `clip_limit = 0` means "off" (return the image unchanged).
    """
    if clip_limit <= 0:
        return img
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
