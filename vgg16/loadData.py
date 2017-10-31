import numpy as np
import json


def trainData( train_dir, mmap ):

	return np.load( train_dir, mmap_mode = mmap ) 

def valdData( vald_dir, mmap ):

	return np.load( vald_dir, mmap_mode = mmap)

def data( train_dir, vald_dir, mmap ):

	t = trainData( train_dir, mmap )
	v = valdData( vald_dir, mmap )

	return t, v

def labels():
	data = json.load( open('params.json') )
	train_labels = np.array( [0] *  ( data["nb_train_samples"] / 2) + [1] * (data["nb_train_samples"]/ 2))
	vald_labels = np.array( [0] *  (data["nb_validation_samples"] / 2) + [1] * (data["nb_validation_samples"] / 2))
	return train_labels, vald_labels

def createDataLabelTuple( train_dir, vald_dir ):

	train_data, vald_data = data(train_dir, vald_dir)
	train_labels, vald_labels = labels()

	return zip(train_data, train_labels), zip(vald_data, vald_labels)


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


# train_data_path = "train_transfer_block3_pool_values.npy"
# vald_data_path = "validation_transfer_block3_pool_values.npy"

# #t,v = data(train_data_path, vald_data_path, 'r')
# s =  np.load( train_data_path, mmap_mode = 'r' )

