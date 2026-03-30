import cv2
import numpy as np

def apply_grayscale(img):
    processed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.repeat(processed[:, :, np.newaxis], 3, axis=2)