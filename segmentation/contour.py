import numpy as np
import cv2

img = cv2.imread("../data/aug/train/benign/ISIC_0000201.jpg_aug0.jpg")
imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,0)
im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img, contours, -1, (0,255,0), 3)

cv2.namedWindow('contour', cv2.WINDOW_NORMAL)
cv2.imshow('contour', img)

cv2.waitKey(0)
cv2.destroyAllWindows()