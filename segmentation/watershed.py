import numpy as np
import cv2
from matplotlib import pyplot as plt

# img = cv2.imread("../data/aug/train/benign/ISIC_0000201.jpg_aug0.jpg")
# img = cv2.imread("../data/aug/train/benign/ISIC_0000011.jpg_aug0.jpg")
# img = cv2.imread("../data/aug/train/benign/ISIC_0000113.jpg_aug0.jpg")
img = cv2.imread("../data/aug/train/benign/ISIC_0009344.jpg_aug12.jpg")

blur = cv2.GaussianBlur(img,(5,5),0)

gray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

cv2.imshow("Binary", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

# kernel = np.ones((5,5),np.uint8)
# erosion = cv2.erode(thresh,kernel,iterations = 1)

# cv2.imshow("Erosion", erosion)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

cv2.imshow("Opening", opening)
cv2.waitKey(0)
cv2.destroyAllWindows()


# # sure background area
# sure_bg = cv2.dilate(opening,kernel,iterations=3)

# # Finding sure foreground area
# dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
# ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

# # Finding unknown region
# sure_fg = np.uint8(sure_fg)
# unknown = cv2.subtract(sure_bg,sure_fg)

# # Marker labelling
# ret, markers = cv2.connectedComponents(sure_fg)

# # Add one to all labels so that sure background is not 0, but 1
# markers = markers+1

# # Now, mark the region of unknown with zero
# markers[unknown==255] = 0

# markers = cv2.watershed(img,markers)
# img[markers == -1] = [255,0,0]

# cv2.imshow("Marker", markers)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imshow("FINAL", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

