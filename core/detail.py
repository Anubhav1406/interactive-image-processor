import cv2
import numpy as np


def apply_gaussian_blur(img, ksize):
    """Gaussian blur — smooth noise by averaging each pixel with its neighbours.

    Math
    ----
    Blurring is a CONVOLUTION: slide a small weight matrix (the kernel) over
    the image and replace each pixel with the weighted sum of the pixels
    under it. A Gaussian kernel weights nearby pixels most, far ones least,
    following the bell curve:

        G(x, y) = (1 / (2*pi*sigma^2)) * exp( -(x^2 + y^2) / (2*sigma^2) )

    Bigger kernel (ksize) -> wider neighbourhood -> stronger blur.
    `ksize` must be ODD (it needs a centre pixel); we force that below.
    OpenCV derives sigma from the kernel size when we pass sigma = 0.
    """
    if ksize <= 1:
        return img
    if ksize % 2 == 0:                 # kernels must be odd-sized
        ksize += 1
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def apply_sharpen(img, amount):
    """Sharpen via UNSHARP MASKING.

    Math
    ----
    Sharpening cannot create real detail; it only exaggerates edges that are
    already present:

        blurred = GaussianBlur(img)            # the low-frequency / soft part
        detail  = img - blurred                # the high-frequency / edge part
        out     = img + amount * detail
                = (1 + amount) * img  -  amount * blurred

    `detail` is the part the blur removed (the edges); a multiple of it is
    added back on top of the original. `addWeighted` computes the last line
    in one pass.

        amount = 0 -> unchanged,   larger amount -> crisper (and noisier).
    """
    if amount <= 0:
        return img
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=2.0)
    return cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)
