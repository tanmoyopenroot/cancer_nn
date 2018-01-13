import sys
sys.path.append('../')

import os
import itertools

import cv2
import numpy as np
import matplotlib.pyplot as plt

from config import img_width, img_height

def getImgArr(path):
	x = cv2.imread(path, cv2.IMREAD_COLOR)
	x = cv2.resize(x, (img_width, img_height))
	x = x.astype(np.float32)
	x = x / 255.0

	return x


def imageGenerator(img_path, batch_size):
	img_set = os.listdir(img_path)
	img_set = img_set.sort()
	img_set_len = len(img_path)
	img_set_cycle = itertools.cycle(img_set)

	while True:
		X = []
		for _ in range(batch_size)
			img_name = img_set_cycle.next()
			X.append(getImgArr(img_path + "/" + img_name))

		yield np.array(X)
