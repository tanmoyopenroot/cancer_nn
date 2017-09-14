import cv2
import numpy as np

def translateX(image_array, value):
    rows, cols = image_array.shape
    value = np.random.uniform() * value
    M = np.float32([[1, 0, value], [0, 1, 0]])
    result_image = cv2.warpAffine(image_array, M, (rows, cols))
    return result_image

def translateY(image_array, value):
    rows, cols = image_array.shape
    value = np.random.uniform() * value
    M = np.float32([[1, 0, 0], [0, 1, value]])
    result_image = cv2.warpAffine(image_array, M, (rows, cols))
    return result_image

def horizontalFlip(image_array):
    return cv2.flip(image_array, 0)

def verticalFlip(image_array):
    return cv2.flip(image_array, 1)

def augment()