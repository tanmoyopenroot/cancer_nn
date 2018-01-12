import sys
sys.path.append('../')

import glob

import cv2
import numpy as np

from config import train_seg_melanoma_dir, train_seg_benign_dir
from config import validation_seg_melanoma_dir, validation_seg_benign_dir

from config import train_aug_melanoma_dir, train_aug_benign_dir
from config import validation_aug_melanoma_dir, validation_aug_benign_dir

from config import augment_values

def translateXY(image_array, wrg, hrg):
    rows, cols, ch = image_array.shape
    tx = np.random.uniform(-hrg, hrg) * rows
    ty = np.random.uniform(-wrg, wrg) * cols
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    result_image = cv2.warpAffine(image_array, M, (cols, rows))
    return result_image

def translateY(image_array, value):
    rows, cols, ch = image_array.shape
    translate_value = np.random.uniform(-value, value) 
    M = np.float32([[1, 0, 0], [0, 1, translate_value]])
    result_image = cv2.warpAffine(image_array, M, (rows, cols))
    return result_image

def rotate(image_array, value):
    rows, cols, ch = image_array.shape
    theata = np.random.uniform(-value, value)
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), theata, 1)    
    result_image = cv2.warpAffine(image_array, M, (cols, rows))
    return result_image

def horizontalFlip(image_array):
    return cv2.flip(image_array, 0)

def verticalFlip(image_array):
    return cv2.flip(image_array, 1)

def shearImage(image_array, shear_value):
    rows, cols, ch = image_array.shape
    pts1 = np.float32([[5, 5], [20, 5], [5, 20]])
    pt1 = 5 + shear_value * np.random.uniform() - shear_value / 2
    pt2 = 20 + shear_value * np.random.uniform() - shear_value / 2
    pts2 = np.float32([[pt1, 5], [pt2, pt1], [5, pt2]])
    shear_matrix = cv2.getAffineTransform(pts1, pts2)
    shear_image = cv2.warpAffine(image_array, shear_matrix, (cols, rows))
    return shear_image

def processImage(x, h_flip = True):
    global augment_values
    x_aug = translateXY(x, augment_values["width_shift_range"], augment_values["height_shift_range"])
    x_aug = rotate(x_aug, augment_values["rotation_range"])
    if h_flip:
        x_aug = horizontalFlip(x_aug)
    else:
        x_aug = verticalFlip(x_aug)
    x_aug = shearImage(x_aug, augment_values["shear_range"])
    return x_aug

def augment(x, aug_no, img_save_dir, img_name):
    for i in range(aug_no):
        if i % 2:
            h_flip = True
        else:
            h_flip = False
        augmented_img = processImage(x, h_flip)
        cv2.imwrite(img_save_dir + img_name + "_aug_" + str(i) + ".jpg", augmented_img)

def augmentImages(train_or_valid, image_dir, img_save_dir):
    if train_or_valid == "train":
        # Training
        print("Augment Training Data")
    else:
        # Validation
        print("Augment Validation Data")
    
    image_set = glob.glob(image_dir + "*.jpg")
    aug_no = 16
    image_len = len(image_set)

    for index, img in enumerate(image_set):
        img_name_with_ext = img.split("/")[-1]
        img_name = (img_name_with_ext).split(".")[0]
        x = cv2.imread(img, cv2.IMREAD_COLOR)
        # print x.shape
        print("Augmenting Image : {0} / {1} - {2}".format(index, image_len, img_name_with_ext))
        augment(x, aug_no, img_save_dir, img_name)


def agumentData():
    augmentImages("train", train_seg_melanoma_dir, train_aug_melanoma_dir)
    augmentImages("train", train_seg_benign_dir, train_aug_benign_dir)
    augmentImages("validation", validation_seg_melanoma_dir, validation_aug_melanoma_dir)
    augmentImages("validation", validation_seg_benign_dir, validation_aug_benign_dir)

agumentData()