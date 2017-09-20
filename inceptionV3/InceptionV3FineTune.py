import numpy as np
import matplotlib.pyplot as plt

from keras import optimizers
from keras import applications
from keras import regularizers
from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator 

img_width, img_height = 224, 224

top_model_weights_path = "isic-vgg16-transfer-learning-07-l2-300e.h5"

train_data_dir = '/home/openroot/Tanmoy/Working Stuffs/myStuffs/havss-tf/ISIC-2017/data/train'
validation_data_dir = '/home/openroot/Tanmoy/Working Stuffs/myStuffs/havss-tf/ISIC-2017/data/validation'

train_aug_data_dir = '/home/openroot/Tanmoy/Working Stuffs/myStuffs/havss-tf/ISIC-2017/data/aug/train'
validation_aug_data_dir = "/home/openroot/Tanmoy/Working Stuffs/myStuffs/havss-tf/ISIC-2017/data/aug/validation"

nb_train_samples = 9216
nb_validation_samples = 2304

epochs = 100

batch_size = 16

# VGG16 Model
base_model = applications.inception_v3.InceptionV3(include_top = False, weights = "imagenet", input_shape = (224,224,3))

# Top Model
# top_model = Sequential()
# top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
# top_model.add(Dense(256, activation = "relu"))
# top_model.add(Dropout(0.7))
# top_model.add(Dense(1, activation = "sigmoid"))

# Add Weights
# top_model.load_weights(top_model_weights_path)

# model = Sequential()
# for layer in base_model.layers:
#     model.add(layer)

# model.add(top_model)

# Set The First 25 Layers To Non Trainlable (Up To Last Conv Block)
for layer in base_model.layers:
    print(layer)

print(len(base_model.layers))
