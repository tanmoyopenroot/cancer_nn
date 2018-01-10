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

def generateMask(img):
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

	kernel = np.ones((5,5),np.uint8)
	opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

	# cv2.namedWindow('img', cv2.WINDOW_NORMAL)
	# cv2.imshow('img', img)

	# cv2.namedWindow('thresh', cv2.WINDOW_NORMAL)
	# cv2.imshow('thresh', thresh)

	# cv2.namedWindow('opening', cv2.WINDOW_NORMAL)
	# cv2.imshow('opening', opening)

	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

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