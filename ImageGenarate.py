import glob
# import h5py
import numpy as np
# from PIL import Image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

train_data_melanoma_dir = "/home/openroot/Tanmoy/Working Stuffs/myStuffs/havss-tf/ISIC-2017/data/train/melanoma/"
train_aug_melanoma_dir = "/home/openroot/Tanmoy/Working Stuffs/myStuffs/havss-tf/ISIC-2017/data/aug/train/melanoma/"
train_melanoma_file = "train-melanoma.npy"
train_data_benign_dir = "/home/openroot/Tanmoy/Working Stuffs/myStuffs/havss-tf/ISIC-2017/data/train/benign/"
train_aug_benign_dir = "/home/openroot/Tanmoy/Working Stuffs/myStuffs/havss-tf/ISIC-2017/data/aug/train/benign/"
train_benign_file = "train-benign.npy"


validation_data_melanoma_dir = "/home/openroot/Tanmoy/Working Stuffs/myStuffs/havss-tf/ISIC-2017/data/validation/melanoma/"
validation_aug_melanoma_dir = "/home/openroot/Tanmoy/Working Stuffs/myStuffs/havss-tf/ISIC-2017/data/aug/validation/melanoma/"
validation_melanoma_file = "validation-melanoma.npy"
validation_data_benign_dir = "/home/openroot/Tanmoy/Working Stuffs/myStuffs/havss-tf/ISIC-2017/data/validation/benign/"
validation_aug_benign_dir = "/home/openroot/Tanmoy/Working Stuffs/myStuffs/havss-tf/ISIC-2017/data/aug/validation/benign/"
validation_benign_file = "validation-benign.npy"

def augment(datagen, image_set, save_to_dir, save_prefix, save_file):
    
    # img = load_img(image_set[0])
    # x = img_to_array(img)
    # augmented_data = np.array(x)
    # x = x.reshape((1,) + x.shape)
    # augmented_data = np.empty( shape=x.shape )
    # print augmented_data.shape
    image_len = len(image_set)

    # x = np.array([np.array(Image.open(img)) for img in image_set])

    for index, img in enumerate(image_set):
        img_name = img.split("/")[-1] 
        img = load_img(img)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)

        print("Augmenting Image : {0} / {1} - {2}".format(index, image_len, img_name))
        # print x.shape

        # augmented_data = np.append(augmented_data, x, axis=0)

        i = 0
        for X_batch in datagen.flow(x, batch_size = 1, save_to_dir = save_to_dir, save_prefix = save_prefix, save_format = "jpg"):
            i += 1
            # augmented_data.append(X_batch)
            # print X_batch.shape
            # augmented_data = np.concatenate((augmented_data, X_batch))
            # augmented_data = np.append(augmented_data, X_batch, axis=0)
            if i > 16:
                break
        
    # print x.shape

    # np.save(open(save_file, "w"), augmented_data)

def augmentImages(train_or_valid, image_dir, image_save_dir, save_file):
    if train_or_valid == "train":
        # Training
        save_prefix = "train"
        print("Augment Training Data")
        datagen = ImageDataGenerator(
            # rescale = 1./255,
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
        save_prefix = "validation"
        print("Augment Validation Data")
        datagen = ImageDataGenerator(
            # rescale = 1./255,
            rotation_range = 35,
            width_shift_range = 0.1,
            height_shift_range = 0.1,
            shear_range = 0.2,
            zoom_range = 0.2,
            horizontal_flip = True,
            fill_mode = "nearest"
        )

    image_set = glob.glob(image_dir + "*.jpg")
    # print image_set[0:10]

    augment(datagen, image_set, image_save_dir, save_prefix, save_file)

def main():
    augmentImages("train", train_data_benign_dir, train_aug_benign_dir, train_benign_file)
    augmentImages("train", train_data_melanoma_dir, train_aug_melanoma_dir, train_melanoma_file)
    augmentImages("validation", validation_data_benign_dir, validation_aug_benign_dir, validation_benign_file)
    augmentImages("validation", validation_data_melanoma_dir, validation_aug_melanoma_dir, validation_melanoma_file)

if __name__ == '__main__':
    main()
