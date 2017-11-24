import numpy as np
import cv2
from matplotlib import pyplot as plt

def autoCannyEdgeDetection(img, sigma = 0.7):
	v = np.median(img)

	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(img, 245, 255)

	return edged

def getContourImage(org, img):
    im2, contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros(img.shape, np.uint8)

    # largest_cnt_area = cv2.contourArea(contours[0])
    # largest_cnt = 0
    # cnt_area = largest_cnt_area

    # for cnt in contours:
    #     cnt_area = cv2.contourArea(cnt)
    #     if cnt_area > largest_cnt_area:
    #         largest_cnt_area = cnt_area
    #         largest_cnt = cnt
        
    #     print cnt_area

    cv2.drawContours(mask, contours, -1, (255,255,255), 5)

    return mask

def getDilationImage(img):
    kernel = np.ones((5,5),np.uint8)
    dilation = cv2.dilate(img,kernel,iterations = 1)
    return dilation

def getOpeningImage(img):
    kernel = np.ones((20,20),np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return opening

def getClosingImage(img):
    kernel = np.ones((35,35),np.uint8)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return closing

def getHSVImage(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    return hsv, hue, sat, val

def getBlurImage(img):
     blur = cv2.GaussianBlur(img,(5,5),0)
     return blur

def getGrayImage(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def getFilterImage(img):
    kernel = np.ones((5, 5), np.float32) / 25
    filtered = cv2.filter2D(img, -1, kernel)
    return filtered

def getBinaryImage(img):
    gray = getGrayImage(img)
    thresh = cv2.threshold(gray, 145, 255, cv2.THRESH_BINARY)[1]
    return thresh

def main():
    img = cv2.imread("../data/aug/train/benign/ISIC_0000201.jpg_aug0.jpg")
    # img = cv2.imread("../data/aug/train/benign/ISIC_0000011.jpg_aug0.jpg")
    # img = cv2.imread("../data/aug/train/benign/ISIC_0    # cv2.namedWindow('dilation', cv2.WINDOW_NORMAL)
    # cv2.imshow("dilation", dilation_img)  000113.jpg_aug0.jpg")
    # img = cv2.imread("../data/aug/train/benign/ISIC_0009344.jpg_aug12.jpg")

    blured_img = getBlurImage(img)
    hsv, hue, sat, val = getHSVImage(blured_img)
    # binary_img_hsv = binaryIMG(blured_img)


    cv2.namedWindow('hsv', cv2.WINDOW_NORMAL)
    cv2.imshow("hsv", hsv)

    # cv2.namedWindow('hue', cv2.WINDOW_NORMAL)
    # cv2.imshow('hue', hsv[:, :, 0])

    # cv2.namedWindow('sat', cv2.WINDOW_NORMAL)
    # cv2.imshow('sat', hsv[:, :, 1])

    # cv2.namedWindow('val', cv2.WINDOW_NORMAL)
    # cv2.imshow('val', hsv[:, :, 2])

    hsv += 35

    cv2.namedWindow('hsv-inc', cv2.WINDOW_NORMAL)
    cv2.imshow("hsv-inc", hsv)


    gray = getBinaryImage(hsv)
    filtered_img = getFilterImage(gray)
    opening_img = getOpeningImage(filtered_img)
    closing_img = getClosingImage(opening_img)

    canny_edge_img = autoCannyEdgeDetection(closing_img)
    contour_img = getContourImage(hsv, canny_edge_img) 
    dilation_img = getDilationImage(contour_img)

    # hsv[:, :, 0] += 10

    # cv2.namedWindow('hue-inc', cv2.WINDOW_NORMAL)
    # cv2.imshow("hue-inc", hsv[:, :, 0])


    cv2.namedWindow('gray', cv2.WINDOW_NORMAL)
    cv2.imshow("gray", gray)    

    # cv2.namedWindow('filter', cv2.WINDOW_NORMAL)
    # cv2.imshow("filter", filtered_img)     

    # cv2.namedWindow('opening', cv2.WINDOW_NORMAL)
    # cv2.imshow("opening", opening_img)  

    cv2.namedWindow('closing', cv2.WINDOW_NORMAL)
    cv2.imshow("closing", closing_img)   

    cv2.namedWindow('canny-edge', cv2.WINDOW_NORMAL)
    cv2.imshow("canny-edge", canny_edge_img)

    cv2.namedWindow('contour', cv2.WINDOW_NORMAL)
    cv2.imshow("contour", contour_img)  

    cv2.namedWindow('dilation', cv2.WINDOW_NORMAL)
    cv2.imshow("dilation", dilation_img)    

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()