import numpy as np
import cv2
from matplotlib import pyplot as plt

def getHSVImage(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    return hsv, hue, sat, val

img = cv2.imread("../data/aug/train/benign/ISIC_0000201.jpg_aug0.jpg")
# img = cv2.imread("../data/aug/train/benign/ISIC_0000011.jpg_aug0.jpg")
# img = cv2.imread("../data/aug/train/benign/ISIC_0000113.jpg_aug0.jpg")
# img = cv2.imread("../data/aug/train/benign/ISIC_0009344.jpg_aug12.jpg")
# img = cv2.imread('coins.png')

# hsv, hue, sat, val = getHSVImage(img)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.imshow('img', img)

cv2.namedWindow('thresh', cv2.WINDOW_NORMAL)
cv2.imshow('thresh', thresh)

cv2.namedWindow('opening', cv2.WINDOW_NORMAL)
cv2.imshow('opening', opening)

cv2.waitKey(0)
cv2.destroyAllWindows()