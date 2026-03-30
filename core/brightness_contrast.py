import cv2
import numpy as np

def apply_brightness(img, b):
    img_inter = img.astype(np.float32) + b
    np.clip(img_inter, 0, 255, out=img_inter)
    return img_inter.astype(np.uint8)

def apply_contrast(img, a):
    img_inter = a*(img.astype(np.float32)-128)+128
    np.clip(img_inter, 0, 255, out=img_inter)
    return img_inter.astype(np.uint8)