import cv2
import numpy as np


def apply_skin_smooth(img, strength):
    """Gentle skin smoothing with a BILATERAL FILTER (edge-preserving blur).

    Math
    ----
    A plain Gaussian blur smooths skin but also destroys the sharp edges of
    eyes, hair and the jawline. A bilateral filter fixes this by weighting
    each neighbour by TWO things at once:

        weight = spatial_weight(distance)  *  range_weight(colour difference)

        spatial_weight ~ exp( -(distance^2)      / (2*sigmaSpace^2) )
        range_weight   ~ exp( -(colour_diff^2)   / (2*sigmaColor^2) )

    The range term is the key: a neighbour contributes only if its COLOUR is
    also similar. So flat skin (similar colours) gets averaged and smoothed,
    while a strong edge (big colour jump, e.g. eye vs skin) contributes ~0
    weight and stays crisp.

    `strength` (0..100) drives both sigmas — higher = smoother, more plastic.
    """
    if strength <= 0:
        return img
    sigma = 20 + strength                 # map slider -> filter strength
    d = 9                                 # neighbourhood diameter in pixels
    return cv2.bilateralFilter(img, d, sigmaColor=sigma, sigmaSpace=sigma)


def apply_heal(img, mask, radius=6):
    """Spot-heal / blemish removal by INPAINTING the painted region.

    Algorithm (Telea's Fast Marching Method, cv2.INPAINT_TELEA)
    ----------------------------------------------------------
    You paint a mask over the blemish. Inpainting then *reconstructs* those
    pixels from the good skin around them:

      1. Start at the BOUNDARY of the masked region (the pixels touching known
         skin) and march inward toward the centre, always filling the pixel
         closest to known territory next (a "fast marching" distance order).
      2. Each unknown pixel is set to a weighted average of its already-known
         neighbours. Neighbours that are closer, and that lie along the local
         image gradient (i.e. along edges rather than across them), get more
         weight — so texture and edges continue smoothly into the patch.

    The result: the blemish is replaced by a believable continuation of the
    surrounding skin, with no visible seam.

    Parameters
    ----------
    mask   : uint8, single channel — NONZERO where you painted (heal here).
    radius : how far around each pixel inpainting looks for good colour.
    """
    if mask is None or not mask.any():
        return img
    return cv2.inpaint(img, mask, radius, cv2.INPAINT_TELEA)
