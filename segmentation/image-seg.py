import sys
sys.path.append('../')

import glob

import cv2
import numpy as np
from matplotlib import pyplot as plt

from config import train_data_melanoma_dir, train_data_benign_dir
from config import validation_data_melanoma_dir, validation_data_benign_dir

from config import train_seg_melanoma_dir, train_seg_benign_dir
from config import validation_seg_melanoma_dir, validation_seg_benign_dir

def autoCannyEdgeDetection(img, sigma = 0.7):
	v = np.median(img)

	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(img, 245, 255)

	return edged

def drawConvexHull(img, contours):
    cnt = contours[0]

    mask = np.zeros(img.shape, np.uint8)

    hull = cv2.convexHull(cnt,returnPoints = False)
    defects = cv2.convexityDefects(cnt,hull)

    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        cv2.line(mask,start,end,[255,255,255],5)
        cv2.circle(mask,far,5,[255,255,255],-1)

    (x,y),radius = cv2.minEnclosingCircle(cnt)
    center = (int(x),int(y))
    radius = int(radius)
    cv2.circle(mask,center,radius,(255,255,255),-1)


    return mask  


def drawLines(img, contours):
    cnt = contours[0]
    (x,y),radius = cv2.minEnclosingCircle(cnt)
    center = (int(x),int(y))
    # radius = int(radius)
    # cv2.circle(mask,center,radius,(255,255,255),-1)

    mask = np.zeros(img.shape, np.uint8)
    cnt_points = []

    for cnt in contours:
        for pts in cnt:
            cnt_points.append(pts[0])

    cnt_points = np.array(cnt_points)
    # np.random.shuffle(cnt_points)
    # print cnt_points.shape
    # print cnt_points[0:10,:]

    for i in range(len(cnt_points)):
        cv2.line(mask,center,(cnt_points[i,0], cnt_points[i,1]),(255,255,255),5)

    return mask


def getContourImage(img):
    im2, contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros(img.shape, np.uint8)
    cv2.drawContours(mask, contours, -1, (255,255,255), 5)

    return contours, mask

def getDilationImage(img):
    kernel = np.ones((50,50),np.uint8)
    dilation = cv2.dilate(img,kernel,iterations = 1)
    return dilation

def getOpeningImage(img):
    kernel = np.ones((35,35),np.uint8)
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

def generateMask(img):    
    blured_img = getBlurImage(img)
    hsv, hue, sat, val = getHSVImage(blured_img)
    # binary_img_hsv = binaryIMG(blured_img)

    # cv2.namedWindow('hsv', cv2.WINDOW_NORMAL)
    # cv2.imshow("hsv", hsv)

    hsv += 35

    # cv2.namedWindow('hsv-inc', cv2.WINDOW_NORMAL)
    # cv2.imshow("hsv-inc", hsv)

    gray = getBinaryImage(hsv)
    filtered_img = getFilterImage(gray)
    opening_img = getOpeningImage(filtered_img)
    closing_img = getClosingImage(opening_img)

    canny_edge_img = autoCannyEdgeDetection(closing_img)
    contours, contour_img = getContourImage(canny_edge_img) 
    binary_line_img = drawLines(contour_img, contours)
    # convex_img = drawConvexHull(hsv, contours) 
    dilation_img = getDilationImage(binary_line_img)

    # cv2.namedWindow('gray', cv2.WINDOW_NORMAL)
    # cv2.imshow("gray", gray)       
    # cv2.namedWindow('canny-edge', cv2.WINDOW_NORMAL)
    # cv2.imshow("canny-edge", canny_edge_img)
    # cv2.namedWindow('contour', cv2.WINDOW_NORMAL)
    # cv2.imshow("contour", contour_img)    
    # cv2.namedWindow('dilation', cv2.WINDOW_NORMAL)
    # cv2.imshow("dilation", dilation_img)    
    # cv2.namedWindow('convex', cv2.WINDOW_NORMAL)
    # cv2.imshow("convex", convex_img)  
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return dilation_img

def extractRegion(img, mask):
    processed_img = cv2.bitwise_and(img, img, mask = mask)
    return processed_img

def segmentImages(train_or_valid, image_dir, img_save_dir):
    if train_or_valid == "train":
        # Training
        print("Segmenting Training Data")
    else:
       
        # Validation
        print("Segmenting Validation Data")
    image_set = glob.glob(image_dir + "*.jpg")
    image_len = len(image_set)

    for index, img in enumerate(image_set):
        img_name = img.split("/")[-1] 
        x = cv2.imread(img, cv2.IMREAD_COLOR)
        # print x.shape
        print("Segmenting Image : {0} / {1} - {2}".format(index, image_len, img_name))
        mask = generateMask(x)
        segment_img = extractRegion(x, mask)
        cv2.imwrite(img_save_dir + img_name + "_seg" + ".jpg", segment_img)

def segmentData():
    segmentImages("train", train_data_melanoma_dir, train_seg_melanoma_dir)
    segmentImages("train", train_data_benign_dir, train_seg_benign_dir)
    segmentImages("validation", validation_data_melanoma_dir, validation_seg_melanoma_dir)
    segmentImages("validation", validation_data_benign_dir, validation_seg_benign_dir)

segmentData()