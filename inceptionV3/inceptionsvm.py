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
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc

img_width, img_height = 299, 299

top_model_weights_path = "isic-inceptionV3-svm-weights.h5"

train_data_dir = '../data/train'
validation_data_dir = '../data/validation'

train_aug_data_dir = '../data/aug/train'
validation_aug_data_dir = "../data/aug/validation"

nb_train_samples = 9216
nb_validation_samples = 2304

epochs = 50

batch_size = 32

def plotDecisionBoundary():
    svc = svm.SVC(kernel='linear', C=1,gamma=0).fit(X, y)
    
    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = (x_max / x_min)/100
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
    np.arange(y_min, y_max, h))
    
    plt.subplot(1, 1, 1)
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.title('SVC with linear kernel')
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


def loadInceptionV3(include_top, weights):
    # Load InceptionV3 Model
    model = applications.InceptionV3(
        include_top=include_top,
        weights=weights
    )

    return model


def extractFeatures():
    model = loadInceptionV3(
        include_top=True,
        weights="imagenet"
    )

    train_fcc_values = model.predict_generator(

        getTrainDataGenObject(train_aug_data_dir),
        nb_train_samples // batch_size,
        verbose=1

    )

    print("Train InceptionV3 FCC Output Shape : {0} ".format(train_fcc_values.shape))
    np.save(open("train-fcc-output.npy", "w"), train_fcc_values)

    validation_fcc_values = model.predict_generator(

        getValidationDataGenObject(validation_aug_data_dir),
        nb_validation_samples // batch_size,
        verbose=1

    )

    print("Validation InceptionV3 FCC Output Shape : {0}".format(validation_fcc_values.shape))
    np.save(open("validation-fcc-output.npy", "w"), validation_fcc_values)


def topModel(optimizer='adam', init='glorot_uniform'):
    # Create Model
    model = Sequential()
    model.add(Flatten(input_shape = (7, 7, 512) ))
    model.add(Dense(1024, kernel_initializer=init, activation="relu"))
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
    train_data = np.load(open("train-transfer-values-vgg16.npy"))
    print("Training Data Shape : {0}".format(train_data.shape))
    train_labels = np.array([0] * (nb_train_samples / 2) + [1] * (nb_train_samples / 2))

    validation_data = np.load(open("validation-transfer-values-vgg16.npy"))
    print("Validation Data Shape : {0}".format(validation_data.shape))
    validation_labels = np.array([0] * (nb_validation_samples / 2) + [1] * (nb_validation_samples / 2))


    # the impact on the results
    classifier = svm.SVC(kernel='rbf', C=50, gamma="auto")
    validation_labels_pred = classifier.fit(train_data, train_labels).predict(validation_data)

    # Calculate Accuracy
    validation_acc = accuracy_score(validation_labels, validation_labels_pred)
    print("Validation Accuracy : {0}".format(validation_acc))

    # # Calculating AUC
    # fpr, tpr, thresholds = roc_curve(validation_data, validation_labels_pred, pos_label=2)
    # classifier_auc = auc(fpr, tpr)
    # print("Model AUC : {0}".format(classifier_auc))

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(validation_labels, validation_labels_pred)
    np.set_printoptions(precision=2)
    print("Confusion Matrix")
    print(cnf_matrix)

    # Plot non-normalized confusion matrix
    # plt.figure()
    # plotConfusionMatrix(cnf_matrix, classes=['0', '1'],
    #                     title='Confusion matrix, without normalization')

    # plt.show()

    # Plot Decision Boundary
    # plotDecisionBoundary()

def main():
    extractFeatures()
    # topModelSVM()


if __name__ == '__main__':
    main()
