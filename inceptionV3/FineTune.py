import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential, Model
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dropout, Flatten, Dense, GlobalAveragePooling2D
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
base_model = InceptionV3(weights='imagenet', include_top=False)

for i, layer in enumerate(base_model.layers):
       print(i, layer.name)

# Top Model
top_model = Sequential()
top_model.add(Dense(1024, input_shape = base_model.output.shape[1:], activation = "relu"))
# model.add(Dense(1024, activation = "relu"))
top_model.add(Dense(1, activation = "sigmoid"))

# Add Weights
top_model.load_weights(top_model_weights_path)

model = Sequential()
for layer in base_model.layers:
    model.add(layer)

model.add(top_model)

model.compile(loss='binary_crossentropy',
    optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
    metrics=['accuracy']
)

for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True


# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale = 1./255,
)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
    train_aug_data_dir,
    target_size = (img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
) 

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
    validation_aug_data_dir,
    target_size = (img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

history = model.fit_generator(
    train_generator,
    steps_per_epoch = nb_train_samples // batch_size,
    epochs = epochs,
    validation_data = validation_generator,
    validation_steps = nb_validation_samples // batch_size
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
