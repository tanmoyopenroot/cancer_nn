import sys
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
# from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import applications
from keras import regularizers
from keras import optimizers

img_width, img_height = 224, 224

top_model_weights_path = "isic-vgg16-transfer-learning-weights.h5"

from config import train_transfer_melanoma_dir, train_transfer_benign_dir
from config import validation_transfer_melanoma_dir, validation_transfer_benign_dir

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


def getDataGenObject(directory):

    datagen = ImageDataGenerator(
        rescale=1. / 255,
        # rotation_range = 40,
        # width_shift_range = 0.1,
        # height_shift_range = 0.1,
        # shear_range = 0.1,
        # zoom_range = 0.1,
        # horizontal_flip = True,
        # fill_mode = "nearest"
    )

    datagen_generator = datagen.flow_from_directory(
        directory,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False
    )

    return datagen_generator


def getTrainDataGenObject(path):

    return getDataGenObject(path)


def getValidationDataGenObject(path):

    return getDataGenObject(path)


def loadVGG16(include_top, weights, pooling):
    # VGG16 Model
    model = applications.VGG16(
        include_top=include_top,
        weights=weights,
        pooling=pooling
    )

    return model


def saveBottleneckTransferValues():
    model = loadVGG16(
        include_top=False,
        weights="imagenet",
        pooling="avg"
    )

    train_transfer_values = model.predict_generator(

        getTrainDataGenObject(train_aug_data_dir),
        nb_train_samples // batch_size,
        verbose=1

    )

    print("Train Transfer Values Shape : {0} ".format(
        train_transfer_values.shape))
    np.save(open("train-transfer-values.npy", "w"), train_transfer_values)

    validation_transfer_values = model.predict_generator(

        getValidationDataGenObject(validation_aug_data_dir),
        nb_validation_samples // batch_size,
        verbose=1

    )

    print("Validation Transfer Value Shape : {0}".format(
        validation_transfer_values.shape))
    np.save(open("validation-transfer-values.npy", "w"),
            validation_transfer_values)


def trainTopModel():
    train_data_melanoma = np.load(open( train_transfer_melanoma_dir + "transfer-values.npy"))
    train_data_benign = np.load(open( train_transfer_benign_dir + "transfer-values.npy"))
    train_data = np.concatenate((train_data_melanoma, train_data_benign), axis=0)
    print("Train Shape : {0}".format(train_data.shape))
    train_labels = np.array(
        [0] * (nb_train_samples / 2) + [1] * (nb_train_samples / 2))

    valid_data_melanoma = np.load(open( validation_transfer_melanoma_dir + "transfer-values.npy"))
    valid_data_benign = np.load(open( validation_transfer_melanoma_dir + "transfer-values.npy"))
    validation_data = np.concatenate((valid_data_melanoma, valid_data_benign), axis=0)
    print("Validation Shape : {0}".format(validation_data.shape))
    validation_labels = np.array(
        [0] * (nb_validation_samples / 2) + [1] * (nb_validation_samples / 2))

    model = Sequential()

    model.add(Dense(256, input_shape = train_data.shape[1:], activation="relu"))
    # model.add(Dropout(0.5))    
    # model.add(Dense(4096, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer="rmsprop",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # model.compile(loss='binary_crossentropy',
    #     optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
    #     metrics=['accuracy']
    # )

    '''
    view_transfer_value = TensorBoard(
        log_dir='../tensorboard/vgg16_transfer_values', 
        histogram_freq=0, 
        batch_size=batch_size, 
        write_graph=True, 
        write_grads=False, 
        write_images=False, 
        embeddings_freq=0, 
        embeddings_layer_names=None, 
        embeddings_metadata=None
    )

    checkpoint = ModelCheckpoint(
        "isic-vgg16-transfer-value-best-weight.h5",
        monitor = "val_acc",
        verbose = 1,
        save_best_only = True,
        mode = True
    )

    callbacks_list = [view_transfer_value, checkpoint]
    '''

    history = model.fit(train_data, train_labels,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(validation_data, validation_labels),
                        # callbacks = callbacks_list
                        )

    # model.save_weights(top_model_weights_path)

    # plot Training
    plotTraining(history)


def main():
    # saveBottleneckTransferValues()
    trainTopModel()


if __name__ == '__main__':
    main()
