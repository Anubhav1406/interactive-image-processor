import cv2

def apply_brightness(img, b):
    return cv2.convertScaleAbs(img, alpha=1, beta=b)

def apply_contrast(img, a):
    return cv2.convertScaleAbs(img, alpha=a, beta=0)