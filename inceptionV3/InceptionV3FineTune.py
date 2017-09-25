import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential, Model
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import applications
from keras import regularizers
from keras import optimizers
from keras import backend as K

img_width, img_height = 299, 299

top_model_weights_path = "isic-inceptionV3-transfer-learning-1024.h5"
model_weights_path = "isic-inceptionV3-fine-tune.h5"

train_data_dir = '../data/train'
validation_data_dir = '../data/validation'

train_aug_data_dir = '../data/aug/train'
validation_aug_data_dir = "../data/aug/validation"

nb_train_samples = 9216
nb_validation_samples = 2304

epochs = 50

batch_size = 32

def plotTraining(history):
    # list all data in history
    print(history.history.keys())


    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def nonTrainableLayer():

    # Inception Model
    base_model = applications.inception_v3.InceptionV3(
        include_top = False, 
        weights = "imagenet"
    )

    # Print Layers
    for index, layer in enumerate(base_model.layers):
        print(index, layer.name)

    # Base Non Trainable Layers
    last_non_trainable_layer = "mixed8"
    non_trainable_layer_model = Model(
        inputs = base_model.input,
        outputs = base_model.get_layer(last_non_trainable_layer).output
    )

    # Training
    train_datagen = ImageDataGenerator(
        rescale = 1./255,
    )

    train_generator = train_datagen.flow_from_directory(
        train_aug_data_dir,
        target_size = (img_height, img_width),
        batch_size = batch_size,
        class_mode = None,
        shuffle = False
    )

    train_fine_tune = non_trainable_layer_model.predict_generator(
        train_generator,
        nb_train_samples // batch_size
    )

    print("Train Fine Tune Shape : {0} ".format(train_fine_tune.shape))

    np.save(open("train-fine-tune-incepV3.npy", "w"), train_fine_tune)


    # Validation
    validation_datagen = ImageDataGenerator(
        rescale=1./255
    )    

    validation_generator = validation_datagen.flow_from_directory(
        validation_aug_data_dir,
        target_size = (img_height, img_width),
        batch_size = batch_size,
        class_mode = None,
        shuffle = False
    )

    validation_fine_tune = non_trainable_layer_model.predict_generator(
        validation_generator,
        nb_validation_samples // batch_size 
    )

    print("Validation Fine Tune Shape : {0}".format(validation_fine_tune.shape))

    np.save(open("validation-fine-tune-incepV3.npy", "w"), validation_fine_tune)


def trainLayers():
    train_data = np.load(open("train-fine-tune-incepV3.npy"))
    train_labels = np.array( [0] *  (nb_train_samples / 2) + [1] * (nb_train_samples / 2))

    validation_data = np.load(open("validation-fine-tune-incepV3.npy"))
    validation_labels = np.array( [0] * (nb_validation_samples / 2) + [1] * (nb_validation_samples / 2))
    

    # Inception Model
    base_model = applications.inception_v3.InceptionV3(
        include_top = False, 
        weights = "imagenet"
    )

    # for layer in base_model.layers:
    #     layer.trainable = False
    
    # Print Layers
    # for index, layer in enumerate(base_model.layers):
        # print(index, layer.name)


    # Trainable Layers
    trainable_model = Sequential()
    trainable_model.add(Input(shape = train_data.shape[1:]))
    for layer in base_model.layers[249:]:
        trainable_model.add(layer)
        print("Added {0} to Trainable Model".format(layer.name))

    print("Add Trainable Layers")
    # Print Layers Of Trainble Model
    for layer in trainable_model.layers:
        print layer.name

def main():
    # nonTrainableLayer()
    trainLayers()

if __name__ == '__main__':
    main()