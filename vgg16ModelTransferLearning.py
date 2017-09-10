import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications

img_width, img_height = 224, 224

top_model_weight_path = "isic-vgg16-transfer-learning.h5"
train_data_dir = '/home/openroot/Tanmoy/Working Stuffs/myStuffs/havss-tf/ISIC-2017/data/train'
validation_data_dir = '/home/openroot/Tanmoy/Working Stuffs/myStuffs/havss-tf/ISIC-2017/data/validation'
nb_train_samples = 580
nb_validation_samples = 150

epochs = 100

batch_size = 16


def saveBottleneckTransferValues():
    # VGG16 Model
    model = applications.VGG16(include_top = False, weights = "imagenet")

    datagen = ImageDataGenerator(rescale = 1./255)

    # Training
    train_datagen = ImageDataGenerator(
        # featurewise_center = True,
        # featurewise_std_normalization = True,
        rescale = 1./255,
        # rotation_range = 40,
        # width_shift_range = 0.1,
        # height_shift_range = 0.1,
        # shear_range = 0.2,
        # zoom_range = 0.2,
        # horizontal_flip = True,
        # fill_mode = "nearest"
    )

    train_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size = (img_width, img_height),
        batch_size = batch_size,
        class_mode = None,
        shuffle = False
    )

    train_transfer_values = model.predict_generator(
        train_generator,
        nb_train_samples // batch_size
    )

    print("Train Transfer Values Shape : {0} ".format(train_transfer_values.shape))

    np.save(open("train-transfer-values.npy", "w"), train_transfer_values)


    # Validation
    validation_datagen = ImageDataGenerator(rescale=1./255)    

    validation_generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size = (img_width, img_height),
        batch_size = batch_size,
        class_mode = None,
        shuffle = False
    )

    validation_transfer_values = model.predict_generator(
        validation_generator,
        nb_validation_samples // batch_size 
    )

    print("Validation Transfer Value Shape : {0}".format(validation_transfer_values.shape))

    np.save(open("validation-tansfer-values.npy", "w"), validation_transfer_values)


def trainTopModel():
    pass

def main():
    saveBottleneckTransferValues()

if __name__ == '__main__':
    main()