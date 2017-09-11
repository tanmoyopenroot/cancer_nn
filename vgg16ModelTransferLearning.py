import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras import regularizers

img_width, img_height = 224, 224

top_model_weights_path = "isic-vgg16-transfer-learning.h5"
train_data_dir = '/home/openroot/Tanmoy/Working Stuffs/myStuffs/havss-tf/ISIC-2017/data/train'
validation_data_dir = '/home/openroot/Tanmoy/Working Stuffs/myStuffs/havss-tf/ISIC-2017/data/validation'
nb_train_samples = 576
nb_validation_samples = 144

epochs = 50

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
    train_data = np.load(open("train-transfer-values.npy"))
    train_labels = np.array( [0] *  (nb_train_samples / 2) + [1] * (nb_train_samples / 2))

    validation_data = np.load(open("validation-tansfer-values.npy"))
    validation_labels = np.array( [0] * (nb_validation_samples / 2) + [1] * (nb_validation_samples / 2))

    model = Sequential()
    model.add(Flatten(input_shape = train_data.shape[1:]))
    # model.add(Dense(512, activation = "relu"))
    # model.add(Dropout(0.7))
    model.add(Dense(256, activation = "relu"))
    model.add(Dropout(0.7))
    model.add(Dense(1, activation = "sigmoid"))

    model.compile(optimizer = "rmsprop", 
        loss = "binary_crossentropy", 
        metrics = ["accuracy"]
    )

    history = model.fit(train_data, train_labels, 
        epochs = epochs, 
        batch_size = batch_size, 
        validation_data = (validation_data, validation_labels)
    )

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

    model.save_weights(top_model_weights_path)

def main():
    # saveBottleneckTransferValues()
    trainTopModel()

if __name__ == '__main__':
    main()