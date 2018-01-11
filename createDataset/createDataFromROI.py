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

from config import ground_truth_data_dir

def extractRegion(img, mask):
    processed_img = cv2.bitwise_and(img, img, mask = mask)
    return processed_img

def joinWithROI(train_or_valid, image_dir, img_save_dir):
    if train_or_valid == "train":
        # Training
        print("Segmenting Training Data")
    else:
        # Validation
        print("Segmenting Validation Data")

    image_set = glob.glob(image_dir + "*.jpg")
    image_len = len(image_set)

    for index, img in enumerate(image_set):
        img_name_with_ext = img.split("/")[-1]
        img_name = (img_name_with_ext).split(".")[0]
        roi_img = ground_truth_data_dir + img_name + "_segmentation.png" 
        mask = cv2.imread(roi_img, cv2.IMREAD_GRAYSCALE)
        x = cv2.imread(img, cv2.IMREAD_COLOR)
        print("Segmenting Image : {0} / {1} - {2}".format(index, image_len, img_name_with_ext))
        segment_img = extractRegion(x, mask)
        cv2.imwrite(img_save_dir + img_name + "_seg" + ".jpg", segment_img)

def ROIData():
    joinWithROI("train", train_data_melanoma_dir, train_seg_melanoma_dir)
    joinWithROI("train", train_data_benign_dir, train_seg_benign_dir)
    joinWithROI("validation", validation_data_melanoma_dir, validation_seg_melanoma_dir)
    joinWithROI("validation", validation_data_benign_dir, validation_seg_benign_dir)

ROIData()