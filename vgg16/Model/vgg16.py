import sys
sys.path.append('../../')

import numpy as np
import matplotlib.pyplot as plt

from keras import applications
from keras.utils import plot_model

from config import save_graph_path

def VGG16(include_top, weights, pooling):
    model = applications.VGG16(
        include_top=include_top,
        weights=weights,
        pooling=pooling
    )

    return model

if __name__ == '__main__':
	m = VGG16(include_top=False, weights="imagenet", pooling="avg")
	plot_model( m , show_shapes=True , to_file= save_graph_path + "vgg16_model.png")
