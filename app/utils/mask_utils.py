import numpy as np
import cv2


def create_mask_from_polygons(polygons, image_shape):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for polygon in polygons:
        if len(polygon) >= 3:
            polygon = polygon.reshape((-1, 1, 2)).astype(np.int32)
            cv2.fillPoly(mask, [polygon], 255)
    return mask


def dilate_mask(mask, kernel_size=3, iterations=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(mask, kernel, iterations=iterations)


def erode_mask(mask, kernel_size=3, iterations=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(mask, kernel, iterations=iterations)


def smooth_mask_edges(mask, blur_size=5):
    return cv2.GaussianBlur(mask, (blur_size, blur_size), 0)


def combine_masks(mask1, mask2):
    return cv2.bitwise_or(mask1, mask2)


def mask_to_bboxes(mask, min_area=100):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            bboxes.append([x, y, x + w, y + h])
    return bboxes


def create_inverse_mask(mask):
    return cv2.bitwise_not(mask)