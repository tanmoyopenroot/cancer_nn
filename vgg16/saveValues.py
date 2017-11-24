import numpy as np
import json
from keras.preprocessing.image import ImageDataGenerator 
from keras.models import Model
from keras import applications  
 
data = json.load( open('params.json') )

def getDataGenObject ( directory, class_mode ):

    datagen = ImageDataGenerator(
        rescale = 1./255,
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
        target_size = ( data["img_height"], data["img_width"] ),
        batch_size = data["batch_size"],
        class_mode = class_mode,
        shuffle = False
    )

    return datagen_generator

def getTrainDataGenObject( path = data["train_aug_data_dir"], class_mode = None ):

    return getDataGenObject( path, class_mode )

def getValidationDataGenObject( path = data["validation_aug_data_dir"], class_mode = None ):

    return getDataGenObject( path, class_mode )

def intermedaiteValues( layer_name = "block4_pool" ):


    print (" Calculating transfer values of layer {0}".format(layer_name) )

    model = applications.VGG16( include_top = False, weights = "imagenet")

    intermediate_model = Model( 
        inputs = model.input,
        outputs = model.get_layer(layer_name).output
    )

    train_transfer_values = intermediate_model.predict_generator(

        getTrainDataGenObject(),
        data["nb_train_samples"] // data["batch_size"],
        verbose = 1

    )

    print ( "Train transfer Values shape {0}".format(train_transfer_values.shape) )

    filename = "train_transfer_" + layer_name + "_values.npy"
    np.save( open(filename, "w+"), train_transfer_values )

    validation_transfer_values = intermediate_model.predict_generator(

        getValidationDataGenObject(),
        data["nb_validation_samples"] // data["batch_size"],
        verbose = 1
    )

    print ( "Validation transfer Values shape {0}".format(validation_transfer_values.shape) )

    filename = "validation_transfer_" + layer_name + "_values.npy"
    np.save( open(filename, "w+"), validation_transfer_values )

def fineTuned(): 
    model = VGG16ConvBlockFive( True )

    print ( "loading intermediate train transfer values" )
    data = np.load( 'train_transfer_intermediate_values.npy' )

    print( "Train Trainsfer Intermediate Values shape {0}".format(
        data.shape
    ))

    train_values = model.predict(
        data,
        verbose = 1
    )

    print( "Train Transfer Fine Tuned Values shape {0}".format( 
        train_values.shape
    ))

    np.save( open('vgg16-train-conv-block-5-fine-tune-feature-maps.npy', "w"), train_values )

    data = np.load(open("validation_transfer_intermediate_values.npy"))

    print( "Validation Trainsfer Intermediate Values shape {0}".format(
        data.shape
    ))

    validation_values = model.predict(
        data,
        verbose = 1
    )

    print( "Validation Transfer Fine Tuned Values shape {0}".format( 
        validation_values.shape
    ))


    np.save( open('vgg16-validation-conv-block-5-fine-tune-feature-maps.npy', "w"), validation_values )

