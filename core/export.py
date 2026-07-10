import cv2
import numpy as np


def encode_image(img, fmt, target_kb=None):
    """Encode a BGR image to bytes, optionally hitting a file-size budget.

    Many upload portals reject files larger than a fixed size (e.g. 200 KB).
    For JPEG, quality can be traded for size, so the highest quality that
    still fits the budget is found by BINARY SEARCH over the quality parameter
    (0..100):

        lo, hi = 10, 95
        while lo <= hi:
            mid = (lo + hi) // 2
            encode at quality=mid
            if size <= budget:  keep it, try HIGHER quality (lo = mid + 1)
            else:               too big,  try LOWER  quality (hi = mid - 1)

    Binary search costs ~log2(85) ~= 7 encodes instead of trying all 100.
    PNG is lossless (no quality knob), so a size budget can't be honoured
    there and we just encode once.

    Returns (bytes, actual_kb).
    """
    fmt = fmt.lower()
    ext = f".{fmt}"

    if fmt in ("jpg", "jpeg") and target_kb:
        budget = target_kb * 1024
        lo, hi = 10, 95
        best = None
        while lo <= hi:
            mid = (lo + hi) // 2
            ok, buf = cv2.imencode(ext, img, [cv2.IMWRITE_JPEG_QUALITY, mid])
            if ok and buf.nbytes <= budget:
                best = buf                 # fits -> remember, push for better
                lo = mid + 1
            else:
                hi = mid - 1               # too big -> back off
        if best is None:                   # even lowest quality overflowed
            ok, best = cv2.imencode(ext, img, [cv2.IMWRITE_JPEG_QUALITY, 10])
        data = best.tobytes()
        return data, len(data) / 1024.0

    ok, buf = cv2.imencode(ext, img)
    data = buf.tobytes()
    return data, len(data) / 1024.0
