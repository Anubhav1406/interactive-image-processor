import cv2
import numpy as np


def apply_gamma(img, gamma):
    """Gamma (power-law) correction — fixes exposure in the mid-tones.

    Math
    ----
    Each pixel is normalised to [0, 1], raised to a power, then scaled back:

        out = 255 * (in / 255) ** (1 / gamma)

    Because the curve is non-linear, it moves the mid-tones a lot while
    barely touching pure black (0) and pure white (255):

        gamma > 1  -> exponent < 1  -> curve bows up   -> image BRIGHTER
        gamma < 1  -> exponent > 1  -> curve bows down -> image DARKER
        gamma = 1  -> straight line -> no change

    We precompute the mapping for all 256 possible values into a lookup
    table (LUT) once, then let OpenCV apply it — O(1) per pixel.
    """
    if abs(gamma - 1.0) < 1e-3:
        return img
    inv = 1.0 / gamma
    table = (((np.arange(256) / 255.0) ** inv) * 255).clip(0, 255).astype(np.uint8)
    return cv2.LUT(img, table)


def apply_saturation(img, scale):
    """Scale colour saturation without touching brightness or hue.

    Math
    ----
    RGB mixes colour and brightness together, so we convert to HSV where
    Saturation ('how vivid') lives on its own channel:

        H (hue)        -> which colour            (unchanged)
        S (saturation) -> how vivid it is         (we multiply by `scale`)
        V (value)      -> how bright it is        (unchanged)

        S_new = clip(S * scale, 0, 1)

        scale = 0   -> greyscale (no colour left)
        scale = 1   -> unchanged
        scale > 1   -> more vivid

    We work in float [0, 1] so S is a clean 0..1 fraction.
    """
    if abs(scale - 1.0) < 1e-3:
        return img
    img_f = img.astype(np.float32) / 255.0
    hsv = cv2.cvtColor(img_f, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * scale, 0, 1)
    out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return (out * 255).clip(0, 255).astype(np.uint8)


def apply_white_balance(img):
    """Auto white balance using the 'gray-world' assumption.

    Math
    ----
    Assumption: averaged over a whole photo, the real world is neutral gray,
    so the mean of the Blue, Green and Red channels *should* be equal. If a
    photo looks too warm/cool, it's because one channel's mean drifted.

    We compute each channel mean, and the overall gray target, then rescale
    every channel so its mean lands back on that target:

        gray        = (mean_B + mean_G + mean_R) / 3
        gain_c      = gray / mean_c            for c in {B, G, R}
        out_c       = clip(in_c * gain_c, 0, 255)

    This is the cheapest colour-cast remover there is, and it works well for
    formal portraits shot against a plain background.
    """
    result = img.astype(np.float32)
    means = result.reshape(-1, 3).mean(axis=0)          # [mean_B, mean_G, mean_R]
    gray = means.mean()
    gains = gray / (means + 1e-6)                        # avoid divide-by-zero
    result *= gains                                      # broadcast per channel
    return np.clip(result, 0, 255).astype(np.uint8)
