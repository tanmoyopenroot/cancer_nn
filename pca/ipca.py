import os
import sys
sys.path.append('../')

import cv2
import numpy as np
from matplotlib import pyplot as plt

from sklearn.decomposition import IncrementalPCA

from config import train_aug_melanoma_dir, train_aug_benign_dir
from config import validation_aug_melanoma_dir, validation_aug_benign_dir

from config import trans_train_melanoma_dir, trans_train_benign_dir
from config import trans_validation_melanoma_dir, trans_validation_benign_dir

from config import batch_size, nb_train_samples, nb_validation_samples
from config import img_width, img_height

def getImgArr(path):
	x = cv2.imread(path, cv2.IMREAD_COLOR)
	x = cv2.resize(x, (img_width, img_height))
	x = x.astype(np.float32)
	x = x / 255.0

	return x

def transformData(path, num_components, save_dir):
	img_set = os.listdir(path)
	img_set.sort()
	img_set_len = len(img_set)
	X = np.zeros((img_set_len, img_height * img_width * 3))

	for i, img in enumerate(img_set):
		img_arr = getImgArr(path + "/" + img).flatten()
		X[i] = img_arr
	print("X - Shape : {0}".format(X.shape))

	ipca = IncrementalPCA(n_components = num_components, batch_size = batch_size)

	# for i in range(0, img_set_len // batch_size):
	ipca.partial_fit(X)


	X_ipca = ipca.fit_transform(X)
	print("Transformed X : {0}".format(X_ipca.shape))
	np.save(open("{0}ipca_data.npy".format(save_dir), "w"), X_ipca)


def processImages():
	aug_dir_arr = [
		validation_aug_melanoma_dir, validation_aug_benign_dir,
		train_aug_melanoma_dir, train_aug_benign_dir,
	]

	ipca_dir_arr = [
		trans_validation_melanoma_dir, trans_validation_benign_dir,
		trans_train_melanoma_dir, trans_train_benign_dir
	]

	sample_size_arr = [
		nb_validation_samples / 2, nb_validation_samples / 2,
		nb_train_samples / 2, nb_train_samples / 2
	]

	for i in range(len(aug_dir_arr)):
		transformData(aug_dir_arr[i], 512, ipca_dir_arr[i])

processImages()