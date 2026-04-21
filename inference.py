import numpy as np
import cv2

def create_mask(pred):
    return (pred > 0.5).astype(np.uint8) * 255

def overlay_image(img, mask):
    return cv2.addWeighted(img, 0.7, mask, 0.3, 0)

def tumor_stats(mask):
    tumor_pixels = np.sum(mask > 0)
    total_pixels = mask.size
    percent = (tumor_pixels / total_pixels) * 100

    ys, xs = np.where(mask > 0)
    bbox = None
    if len(xs) > 0:
        bbox = (min(xs), min(ys), max(xs), max(ys))

    return tumor_pixels, percent, bbox
