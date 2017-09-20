import numpy as np
import matplotlib.pyplot as plt

from keras import optimizers
from keras import applications
from keras import regularizers
from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator 

img_width, img_height = 224, 224

top_model_weights_path = "isic-vgg16-transfer-learning-07-l2-300e.h5"

train_data_dir = '/home/openroot/Tanmoy/Working Stuffs/myStuffs/havss-tf/ISIC-2017/data/train'
validation_data_dir = '/home/openroot/Tanmoy/Working Stuffs/myStuffs/havss-tf/ISIC-2017/data/validation'

train_aug_data_dir = '/home/openroot/Tanmoy/Working Stuffs/myStuffs/havss-tf/ISIC-2017/data/aug/train'
validation_aug_data_dir = "/home/openroot/Tanmoy/Working Stuffs/myStuffs/havss-tf/ISIC-2017/data/aug/validation"

nb_train_samples = 9216
nb_validation_samples = 2304

epochs = 100

batch_size = 16

# VGG16 Model
base_model = applications.VGG16(include_top = False, weights = "imagenet", input_shape = (224,224,3))

# Top Model
top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(256, activation = "relu"))
top_model.add(Dropout(0.7))
top_model.add(Dense(1, activation = "sigmoid"))

# Add Weights
top_model.load_weights(top_model_weights_path)

model = Sequential()
for layer in base_model.layers:
    model.add(layer)

model.add(top_model)

# Set The First 25 Layers To Non Trainlable (Up To Last Conv Block)
for layer in model.layers[:25]:
    print(layer)
    layer.tainable = False

model.compile(
    loss = "binary_crossentropy",
    optimizer = optimizers.SGD(lr = 1e-4, momentum = 0.9),
    metrics = ["accuracy"]
)

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    # rotation_range = 40,
    # width_shift_range = 0.1,
    # height_shift_range = 0.1,
    # shear_range = 0.2,
    # zoom_range = 0.2,
    # horizontal_flip = True,
    # fill_mode = "nearest"
)

# this is the augmentation configuration we will use for testing:
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

model.save_weights('InceptionV3ModelFineTuning.h5')