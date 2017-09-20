import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
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
    
    # Inception Model
    base_model = applications.inception_v3.InceptionV3(
        include_top = False, 
        weights = "imagenet"
    )
    
    # Trainable Layers
    from_trainable_layer = "mixed8"
    to_trainable_layer = ""
    trainable_layer_model = Model(
        inputs = base_model.get_layer(from_trainable_layer).input,
        outputs = base_model.get_layer(to_trainable_layer).output
    )  

    # Top Model
    top_model = Sequential()
    top_model.add(Dense(1024, input_shape = train_data.shape[1:], activation = "relu"))
    # top_model.add(Dropout(0.7))
    top_model.add(Dense(1, activation = "sigmoid"))

    # Add Weights
    top_model.load_weights(top_model_weights_path)

    # model = Sequential()
    # for layer in base_model.layers:
    #     model.add(layer)

    trainable_layer_model.add(top_model)  

    trainable_layer_model.compile(loss='binary_crossentropy',
        optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
        metrics=['accuracy']
    )

    view_transfer_value = TensorBoard(
        log_dir='../tensorboard/inception_transfer_values', 
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
        "isic-inceptionV3-transfer-value-best-weight.h5",
        monitor = "val_acc",
        verbose = 1,
        save_best_only = True,
        mode = True
    )

    callbacks_list = [view_transfer_value, checkpoint]

    history = trainable_layer_model.fit(train_data, train_labels, 
        epochs = epochs, 
        batch_size = batch_size, 
        validation_data = (validation_data, validation_labels)
        # callbacks = callbacks_list
    )

    trainable_layer_model.save_weights(model_weights_path)

    # plot Training
    plotTraining(history)

def main():
    nonTrainableLayer()

if __name__ == '__main__':
    main()