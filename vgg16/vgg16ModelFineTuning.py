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

import loadData
from FCC import FCC


def VGG16ConvBlockFour():

    input_vector = Input( shape = ( 28, 28, 256 ) )
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(input_vector)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    model = Model( input_vector, x )

    return model



def VGG16ConvBlockFive( pretrained_weights ):

    input_vector = Input( shape = ( 14, 14, 512 ) )

    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')( input_vector )
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    model = Model( input_vector, x )

    if pretrained_weights :
        print "finetuned conv_block_5 weights loading"

        model.load_weights( 'FCC-init-random-weights-on-finetuned-data.h5', by_name = True )

    return model

def fineTune( block = 'block_4' ):

    print "loading conv layers"
    model = VGG16ConvBlockFive( True )
    print "loading conv block 4"
    conv4_model = VGG16ConvBlockFour() 

    model = Model( input = conv4_model.input, output = model( conv4_model.output ) )

    print "loding FCC"
    top_model = FCC().loadFCC( (7, 7, 512) )
    top_model.load_weights( 'FCC-init-random-weights-on-finetuned-data.h5' )

    print( "combining")
    model = Model( input = model.input, output = top_model( model.output ) )

    model.compile(loss='binary_crossentropy',
        optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
        #optimizer = optimizers.RMSprop( lr = 1e-2 ),
        #optimizer = optimizers.Adam( lr = 1e-2 ),
        metrics=['accuracy']
    )

    train_data_path = "train_transfer_block3_pool_values.npy"
    vald_data_path = "validation_transfer_block3_pool_values.npy"

    train_data, vald_data = loadData.data( train_data_path, vald_data_path, 'r')
    train_labels, vald_labels = loadData.labels()


    history = model.fit(
        train_data, train_labels, 
        epochs = 25, 
        batch_size = 32, 
        validation_data = (vald_data, vald_labels),
        shuffle = True
       #callbacks = callbacks_list
    )
    model.save_weights(weights_path)




def main():

    fineTune()
    

    
if __name__ == '__main__':
    main()
