import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread("../data/aug/train/benign/ISIC_0000201.jpg_aug0.jpg")
# img = cv2.imread("../data/aug/train/benign/ISIC_0000011.jpg_aug0.jpg")
# img = cv2.imread("../data/aug/train/benign/ISIC_0000113.jpg_aug0.jpg")
# img = cv2.imread("../data/aug/train/benign/ISIC_0009344.jpg_aug12.jpg")

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


cv2.namedWindow('full', cv2.WINDOW_NORMAL)
cv2.imshow("full", hsv)

cv2.namedWindow('hsv', cv2.WINDOW_NORMAL)
cv2.imshow('hsv', hsv[:, :, 0])


cv2.namedWindow('sat', cv2.WINDOW_NORMAL)
cv2.imshow('sat', hsv[:, :, 1])


cv2.namedWindow('val', cv2.WINDOW_NORMAL)
cv2.imshow('val', hsv[:, :, 2])

cv2.waitKey(0)
cv2.destroyAllWindows()