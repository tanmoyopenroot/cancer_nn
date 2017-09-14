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

train_data_melanoma_dir = "/home/openroot/Tanmoy/Working Stuffs/myStuffs/havss-tf/ISIC-2017/data/train/melanoma"
train_transfer_value_melanoma_file = "train-transfer-value-melanoma.npy"
train_data_benign_dir = "/home/openroot/Tanmoy/Working Stuffs/myStuffs/havss-tf/ISIC-2017/data/train/benign/"
train_transfer_value_benign_file = "train-transfer-value-benign.npy"


validation_data_melanoma_dir = "/home/openroot/Tanmoy/Working Stuffs/myStuffs/havss-tf/ISIC-2017/data/validation/benign/"
validation_transfer_value_melanoma_file = "validation-transfer-value-melanoma.npy"
validation_data_benign_dir = "/home/openroot/Tanmoy/Working Stuffs/myStuffs/havss-tf/ISIC-2017/data/validation/melanoma/"
validation_transfer_value_benign_file = "validation-transfer-value-benign.npy"

nb_train_samples = 576
nb_validation_samples = 144

epochs = 50

batch_size = 16

# VGG16 Model
model = applications.VGG16(include_top = False, weights = "imagenet")

def saveBottleneckTransferValues(train_or_valid, data_dir, samples, transfer_file):

    if train_or_valid == "train":
        # Training
        print("Augment Training Data")
        datagen = ImageDataGenerator(
            rescale = 1./255,
            rotation_range = 40,
            width_shift_range = 0.1,
            height_shift_range = 0.1,
            shear_range = 0.2,
            zoom_range = 0.2,
            horizontal_flip = True,
            fill_mode = "nearest"
        )
    
    else:
        # Validation
        print("Augment Validation Data")
        datagen = ImageDataGenerator(rescale = 1./255)
        

    generator = datagen.flow_from_directory(
        data_dir,
        target_size = (img_width, img_height),
        batch_size = batch_size,
        class_mode = None,
        shuffle = False
    )

    print("Class Dictionary")
    class_dictionary = generator.class_indices
    print class_dictionary

    print("Classes")
    classes = generator.classes
    print class_dictionary

    print("Loop Through Generator")
    for i in generator:
        idx = (generator.batch_index - 1) * generator.batch_size
        print(generator.filenames[idx : idx + generator.batch_size])

    transfer_values = model.predict_generator(
        generator,
        samples // batch_size
    )

    if train_or_valid == "train":
        print("Train Transfer Values Shape : {0} ".format(transfer_values.shape))
    else:
        print("Validation Transfer Values Shape : {0} ".format(transfer_values.shape))

    np.save(open(transfer_file, "w"), transfer_values)


def trainTopModel():
    train_melanoma_data = np.load(open(train_transfer_value_melanoma_file))
    train_benign_data = np.load(open(train_transfer_value_benign_file))

    train_labels = np.array( [0] *  (nb_train_samples / 2) + [1] * (nb_train_samples / 2))

    validation_melanoma_data = np.load(open(validation_transfer_value_melanoma_file))
    validation_benign_data = np.load(open(validation_transfer_value_benign_file))
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
    print("Creating Training Transfer Values For Melanoma")
    # Train Melanoma
    saveBottleneckTransferValues(
        train_or_valid = "train", 
        data_dir = train_data_dir,
        samples = nb_train_samples,
        transfer_file = train_transfer_value_melanoma_file    
    )

    # print("Creating Training Transfer Values For Benign")
    # # Train Benign
    # saveBottleneckTransferValues(
    #     train_or_valid = "train", 
    #     data_dir = train_data_benign_dir,
    #     samples = nb_train_samples // 2,
    #     transfer_file = train_transfer_value_benign_file    
    # )

    # print("Creating Validation Transfer Values For Melanoma")
    # # Validation Melanoma
    # saveBottleneckTransferValues(
    #     train_or_valid = "validation", 
    #     data_dir = validation_data_melanoma_dir,
    #     samples = nb_validation_samples // 2,
    #     transfer_file = validation_transfer_value_melanoma_file    
    # )

    # print("Creating Validation Transfer Values For Benign")
    # # Train Benign
    # saveBottleneckTransferValues(
    #     train_or_valid = "validation", 
    #     data_dir = validation_data_benign_dir,
    #     samples = nb_validation_samples // 2,
    #     transfer_file = validation_transfer_value_benign_file    
    # )

    # trainTopModel()

if __name__ == '__main__':
    main()