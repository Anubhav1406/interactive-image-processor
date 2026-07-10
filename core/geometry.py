import cv2
import numpy as np


def apply_rotate(img, angle):
    """Rotate the image about its centre (for straightening a tilted photo).

    Math
    ----
    A 2-D rotation by angle t maps a point (x, y) to:

        x' =  cos(t) * x + sin(t) * y
        y' = -sin(t) * x + cos(t) * y

    To rotate about the image CENTRE (cx, cy) instead of the origin, OpenCV
    builds a 2x3 affine matrix M = getRotationMatrix2D(center, angle, scale):

        M = | cos  sin  tx |
            |-sin  cos  ty |

    where (tx, ty) shift the result back so the centre stays put. warpAffine
    then computes, for each OUTPUT pixel, where it came from in the input and
    interpolates the colour there.
    """
    if abs(angle) < 1e-3:
        return img
    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def apply_flip(img, horizontal=False, vertical=False):
    """Mirror the image. Horizontal flip = selfie 'un-mirror'."""
    if horizontal and vertical:
        return cv2.flip(img, -1)
    if horizontal:
        return cv2.flip(img, 1)
    if vertical:
        return cv2.flip(img, 0)
    return img


def crop_to_aspect(img, ratio_w, ratio_h):
    """Centre-crop the image to a target width:height ratio.

    Math
    ----
    Passport / placement portals demand a fixed aspect ratio (e.g. 35:45 for
    an Indian passport photo, or 1:1 for a LinkedIn/portal avatar). We keep
    the whole of one dimension and trim the other symmetrically:

        target = ratio_w / ratio_h
        actual = W / H

        if actual > target:   image too WIDE  -> crop the sides
            new_W = round(H * target);  x0 = (W - new_W) // 2
        else:                 image too TALL  -> crop top/bottom
            new_H = round(W / target);  y0 = (H - new_H) // 2

    Nothing is stretched, so faces keep their real proportions.
    """
    if ratio_w <= 0 or ratio_h <= 0:
        return img
    h, w = img.shape[:2]
    target = ratio_w / ratio_h
    actual = w / h
    if actual > target:                       # too wide -> trim sides
        new_w = int(round(h * target))
        x0 = (w - new_w) // 2
        return img[:, x0:x0 + new_w]
    else:                                     # too tall -> trim top/bottom
        new_h = int(round(w / target))
        y0 = (h - new_h) // 2
        return img[y0:y0 + new_h, :]


def crop_box(img, x, y, w, h):
    """Crop an axis-aligned rectangle given its top-left corner and size.

    Used by the interactive drag-to-crop tool: the drawable canvas gives us a
    rectangle in *display* coordinates, the UI scales it back to full-image
    coordinates, and this function does the actual slice — clamped to the image
    so an out-of-bounds or zero-area rectangle can never crash or return empty.
    """
    h_img, w_img = img.shape[:2]
    x0 = max(0, int(round(x)))
    y0 = max(0, int(round(y)))
    x1 = min(w_img, int(round(x + w)))
    y1 = min(h_img, int(round(y + h)))
    if x1 <= x0 or y1 <= y0:           # empty / inverted selection -> no-op
        return img
    return img[y0:y1, x0:x1]


def resize_to(img, width, height):
    """Resample to an exact pixel size (last step before export).

    Uses INTER_AREA when shrinking (averages source pixels -> no aliasing)
    and INTER_CUBIC when enlarging (smooth interpolation). Choosing the right
    interpolation is what separates a clean resize from a jagged one.
    """
    if width <= 0 or height <= 0:
        return img
    h, w = img.shape[:2]
    shrinking = width * height < w * h
    interp = cv2.INTER_AREA if shrinking else cv2.INTER_CUBIC
    return cv2.resize(img, (width, height), interpolation=interp)
