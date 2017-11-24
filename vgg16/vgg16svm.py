import time
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
# from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import applications
from keras import regularizers
from keras import optimizers

from sklearn import svm
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.linear_model.logistic import LogisticRegression

img_width, img_height = 224, 224

top_model_weights_path = "isic-vgg16-svm-weights.h5"

train_data_dir = '../data/train'
validation_data_dir = '../data/validation'

train_aug_data_dir = '../data/aug/train'
validation_aug_data_dir = "../data/aug/validation"

nb_train_samples = 9216
nb_validation_samples = 2304

epochs = 50

batch_size = 32

def plotInputData(X, Y, title, data_len):
    
    time_start = time.time()   
    X = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300).fit_transform(X)
    print("After Reduction Data Shape : {0}".format(X.shape))    
    print 't-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start) 

    # Main scatter plot and plot annotation
    f, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(X[:data_len / 2, 0] * 10, X[:data_len / 2, 1] * 10, marker = 'o', color = 'green', s=30, alpha=0.5)
    ax.scatter(X[data_len / 2:, 0] * 10, X[data_len / 2:, 1] * 10, marker = '^', color = 'blue', s=30, alpha=0.5)
    plt.legend(["Melanoma", "Benign"], loc='upper right') 
    plt.title(title)
    plt.ylabel('Y')
    plt.xlabel('X')

    # plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.title('SVC Data Plot')
    plt.show()

def plotConfusionMatrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    print('Confusion matrix, without normalization')
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

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


def loadVGG16(include_top, weights):
    # VGG16 Model
    model = applications.VGG16(
        include_top=include_top,
        weights=weights
    )

    return model


def extractFeatures():
    model = loadVGG16(
        include_top=True,
        weights="imagenet"
    )

    train_fcc_values = model.predict_generator(

        getTrainDataGenObject(train_aug_data_dir),
        nb_train_samples // batch_size,
        verbose=1

    )

    print("Train FC8 Output Shape : {0} ".format(train_fcc_values.shape))
    np.save(open("train-fc8-output.npy", "w"), train_fcc_values)

    validation_fcc_values = model.predict_generator(

        getValidationDataGenObject(validation_aug_data_dir),
        nb_validation_samples // batch_size,
        verbose=1

    )

    print("Validation FC8 Output Shape : {0}".format(validation_fcc_values.shape))
    np.save(open("validation-fc8-output.npy", "w"), validation_fcc_values)


def topModel(optimizer='adam', init='glorot_uniform'):
    # Create Model
    model = Sequential()
    model.add(Flatten(input_shape = (7, 7, 512) ))
    model.add(Dense(256, kernel_initializer=init, activation="relu"))
    model.add(Dropout(0.7))    
    # model.add(Dense(4096, activation="relu"))
    model.add(Dense(1, kernel_initializer=init, activation="sigmoid"))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer,  metrics=['accuracy'])

    # model.compile(
    #     loss='binary_crossentropy',
    #     optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
    #     metrics=['accuracy']
    # )

    return model

def topModelSVM():
    train_data = np.load(open("train-fc8-output.npy"))
    print("Training Data Shape : {0}".format(train_data.shape))
    train_labels = np.array([0] * (nb_train_samples / 2) + [1] * (nb_train_samples / 2))

    # train_shape = train_data.shape
    # train_data = np.reshape(train_data, (-1, train_shape[1] * train_shape[2] * train_shape[3]))
    # print("Shape : {0}".format(train_data.shape))

    validation_data = np.load(open("validation-fc8-output.npy"))
    print("Validation Data Shape : {0}".format(validation_data.shape))
    validation_labels = np.array([0] * (nb_validation_samples / 2) + [1] * (nb_validation_samples / 2))


    # validation_shape = validation_data.shape
    # validation_data = np.reshape(validation_data, (-1, validation_shape[1] * validation_shape[2] * validation_shape[3]))
    # print("Shape : {0}".format(validation_data.shape))


    # the impact on the results
    # classifier = svm.SVC(kernel='rbf', C = 700, gamma=100)
    # validation_labels_pred = classifier.fit(train_data, train_labels).predict(validation_data)

    # print validation_data[0] * 10
    # print("Checking")
    # print("Validation Labels")
    # print(validation_labels[0:10])
    # print("Pred Labels")
    # print(validation_labels_pred[0:10])

    # # Calculate Accuracy
    # validation_acc = accuracy_score(validation_labels, validation_labels_pred)
    # print("Validation Accuracy : {0}".format(validation_acc))

    # # Calculating AUC
    # fpr, tpr, thresholds = roc_curve(validation_data, validation_labels_pred, pos_label=2)
    # classifier_auc = auc(fpr, tpr)
    # print("Model AUC : {0}".format(classifier_auc))

    # Compute confusion matrix
    # cnf_matrix = confusion_matrix(validation_labels, validation_labels_pred)
    # np.set_printoptions(precision=2)
    # print("Confusion Matrix")
    # print(cnf_matrix)

    # Plot non-normalized confusion matrix
    # plt.figure()
    # plotConfusionMatrix(cnf_matrix, classes=['0', '1'],
    #                     title='Confusion matrix, without normalization')

    # plt.show()

    # Plot Decision Boundary
    plotInputData(
        X = train_data,
        Y = train_labels,
        title = "Train Data",
        data_len = nb_train_samples
    )

    plotInputData(
        X = validation_data,
        Y = validation_labels,
        title = "Validation Data",
        data_len = nb_validation_samples
    )

def main():
    # extractFeatures()
    topModelSVM()


if __name__ == '__main__':
    main()
