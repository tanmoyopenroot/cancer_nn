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

def centerCrop(img, pad):
    img_height, img_width = img.shape[:2]
    
    return img[pad : img_height - pad, pad : img_width - pad] 

def border(img, width):
    img_height, img_width = img.shape[:2]

    img[0 : img_height, 0 : width] = 0
    img[0 : img_height, img_width - width : img_width] = 0

    img[0 : width, 0 : img_width] = 0
    img[img_height - width : img_height, 0 : img_width] = 0

    return img

def getHSVImage(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    return hsv, hue, sat, val


def generateMask(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)    
    equ = cv2.equalizeHist(gray)
    # blur = cv2.GaussianBlur(equ,(5,5),0)
    ret, thresh = cv2.threshold(equ,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    kernel = np.ones((7,7),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 10)

    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations = 3)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg) 

    return opening

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
        x = border(x, 120)
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

def test():
    # img = "../data/aug/train/benign/ISIC_0000201.jpg_aug0.jpg"
    # img = "../data/aug/train/benign/ISIC_0000011.jpg_aug0.jpg"
    img = "../data/aug/train/benign/ISIC_0000113.jpg_aug0.jpg"
    # img = "../data/aug/train/benign/ISIC_0009344.jpg_aug12.jpg"

    # img = "../data/aug/train/melanoma/ISIC_0000031.jpg_aug1.jpg"
    # img = "../data/aug/train/melanoma/ISIC_0000036.jpg_aug4.jpg"

    x = cv2.imread(img, cv2.IMREAD_COLOR)
    print(x.shape)
    x = border(x, 120)
    print(x.shape)
    mask = generateMask(x)
    print(mask.shape)

    segment_img = extractRegion(x, mask)

    cv2.namedWindow('x', cv2.WINDOW_NORMAL)
    cv2.imshow('x',x)

    cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
    cv2.imshow('mask',mask)

    cv2.namedWindow('segment_img', cv2.WINDOW_NORMAL)
    cv2.imshow('segment_img',segment_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# test()

def testBorder():
    # img = "../data/aug/train/benign/ISIC_0000201.jpg_aug0.jpg"
    # img = "../data/aug/train/benign/ISIC_0000011.jpg_aug0.jpg"
    # img = "../data/aug/train/benign/ISIC_0000113.jpg_aug0.jpg"
    img = "../data/aug/train/benign/ISIC_0009344.jpg_aug12.jpg"

    # img = "../data/aug/train/melanoma/ISIC_0000031.jpg_aug1.jpg"
    # img = "../data/aug/train/melanoma/ISIC_0000036.jpg_aug4.jpg"

    x = cv2.imread(img, cv2.IMREAD_COLOR)
    print(x.shape)
    bordered_img = border(x, 120)
    print(bordered_img.shape)

    cv2.namedWindow('x', cv2.WINDOW_NORMAL)
    cv2.imshow('x',x)

    cv2.namedWindow('bordered_img', cv2.WINDOW_NORMAL)
    cv2.imshow('bordered_img',bordered_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# testBorder()