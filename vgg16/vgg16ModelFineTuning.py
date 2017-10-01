import numpy as np
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

from keras import backend as K
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.models import Model
from keras.utils.data_utils import get_file
from keras import optimizers
from keras import applications
from keras import regularizers
from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, Dense


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


def VGG16ConvBlockFive( pretrained_weights ):

    input_img = Input( shape = ( 14, 14, 512 ) )

    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')( input_img )
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    model = Model( input_img, x )

    if pretrained_weights :
        print "finetuned conv_block_5 weights loading"

        model.load_weights( 'fine-tune-vgg16.h5', by_name = True )

    return model

def FCC( pretrained_weights ):
    
    model = Sequential()
    model.add(Flatten(input_shape = (28, 28, 256) ))

    model.add(Dense(
        64,
        activation = "relu"
    ))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation = "sigmoid"))

    if pretrained_weights:
        print "pretrained FCC weights loading"
        model.load_weights( 'FCC-init-random-weights-on-finetuned-data.h5' )

    return model


def vgg16FromScratch():

    model = applications.VGG16(include_top = False, weights = None, input_shape = (224, 224, 3))

    return model

def initModel( fine_tune = True ):


    weights_path = 'fine-tune-vgg16.h5'

    #load model
    if fine_tune :
        #load data
        print ( "loading intermediate transfer values" )
        train_data = np.load( 'train_transfer_intermediate_values.npy' )
        train_labels = np.array( [0] *  (nb_train_samples / 2) + [1] * (nb_train_samples / 2))
        validation_data = np.load(open("validation_transfer_intermediate_values.npy"))
        validation_labels = np.array( [0] * (nb_validation_samples / 2) + [1] * (nb_validation_samples / 2))

        print ("loading conv block 5")
        model = VGG16ConvBlockFive( False )
        

    else:
        weights_path = 'scratch-vgg16.h5'
        model = vgg16FromScratch()



    print ("loading FCC")
    top_model = FCC( True )

    print( "combining")
    model = Model( input = model.input, output = top_model( model.output ) )



    model.compile(loss='binary_crossentropy',
        optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
        #optimizer = optimizers.RMSprop( lr = 1e-2 ),
        #optimizer = optimizers.Adam( lr = 1e-2 ),
        metrics=['accuracy']
    )

    if fine_tune:
        history = model.fit(train_data, train_labels, 
            epochs = epochs, 
            batch_size = batch_size, 
            validation_data = (validation_data, validation_labels),
            shuffle = True
           #callbacks = callbacks_list
        )
    else:
        history = model.fit_generator(
            getTrainDataGenObject( class_mode = 'binary' ),
            epochs = epochs,
            steps_per_epoch = nb_train_samples // batch_size,
            validation_data = getValidationDataGenObject( class_mode = 'binary' ),
            validation_steps = nb_validation_samples // batch_size,
        )




    model.save_weights(weights_path)

    # plot Training
    plotTraining(history)
    

def testFCC():
    
    model = FCC( True )
    validation_data = np.load(open("vgg16-validation-conv-block-5-fine-tune-feature-maps.npy"))
    # validation_data = np.load(open("validation-transfer-values-vgg16.npy"))
    validation_labels = np.array( [0] * (nb_validation_samples / 2) + [1] * (nb_validation_samples / 2))

    pred = model.predict(

        validation_data,
        batch_size = 16,
        verbose = 1
    )

    print "confusion matrix"

    print confusion_matrix( np.round( pred ) , validation_labels, labels = [0, 1])

    print roc_auc_score(validation_labels, pred)

def trainFCC():

    model = FCC( False )    

    train_data = np.load( "train_transfer_block3_pool_values.npy", mmap_mode = 'r' )
    # train_data = np.load( 'train-transfer-values-vgg16.npy' )
    train_labels = np.array( [0] *  (9216 / 2) + [1] * (9216 / 2))

    validation_data = np.load( "validation_transfer_block3_pool_values.npy", mmap_mode = 'r')
    # validation_data = np.load(open("validation-transfer-values-vgg16.npy"))
    validation_labels = np.array( [0] * (2304 / 2) + [1] * (2304/ 2))

    model.compile(loss='binary_crossentropy',
        #optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
        #optimizer = optimizers.RMSprop( lr = 1e-2 ),
        optimizer = optimizers.Adam( lr = 1e-3 ),
        metrics=['accuracy']
    )


    history = model.fit(train_data, train_labels, 
        epochs = 30, 
        batch_size = 64, 
        validation_data = (validation_data, validation_labels),
        shuffle = True
       #callbacks = callbacks_list
    )

    #model.save_weights( 'FCC-init-random-weights-on-finetuned-data.h5')
    #plotTraining(history)

def main():
    # initModel( fine_tune = False )

    trainFCC()
    # testFCC()
    

    
if __name__ == '__main__':
    main()
