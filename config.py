# Dataset
path = "../data/"

# Main Data
image_folder_path = path + "ISIC-2017/ISIC-2017/"
main_isic_path = path

csv_file_path = path
csv_file_name = "ISIC-2017-label.csv"

image_extension = ".jpg"
resized_image_size = (100, 70)
pixel_depth = 255.0

# Save Graph Path
save_graph_path = path + "graph/"

# Train Files
train_data_melanoma_dir =  path + "train/melanoma/"
train_data_benign_dir =  path +  "train/benign/"

# Validation Files
validation_data_melanoma_dir =  path +  "validation/melanoma/"
validation_data_benign_dir =  path +  "validation/benign/"

# Validation Files
test_data_melanoma_dir =  path +  "test/melanoma/"
test_data_benign_dir =  path +  "test/benign/"

# Ground Truth
ground_truth_data_dir = path + "ISIC-2017-GroundTruth/"

# Train Augment Files
train_aug_melanoma_dir =  path +  "aug/train/melanoma/"
train_aug_benign_dir =  path +  "aug/train/benign/"

# Validation Augment Files
validation_aug_melanoma_dir =  path +  "aug/validation/melanoma/"
validation_aug_benign_dir =  path +  "aug/validation/benign/"

# Train Segmented Files
train_seg_melanoma_dir = path + "seg/train/melanoma/"
train_seg_benign_dir = path + "seg/train/benign/"

# Validation Segmented Files
validation_seg_melanoma_dir = path + "seg/validation/melanoma/"
validation_seg_benign_dir = path + "seg/validation/benign/"

# Train Transfer Files
train_transfer_melanoma_dir = path + "transfer_values/train/melanoma/"
train_transfer_benign_dir = path + "transfer_values/train/benign/"

# Validation Transfer Files
validation_transfer_melanoma_dir = path + "transfer_values/validation/melanoma/"
validation_transfer_benign_dir = path + "transfer_values/validation/benign/"

# Train Numpy Files
train_melanoma_file =  path +  "train/train-melanoma.npy"
train_benign_file =  path +  "train/train-benign.npy"

# Validation Numpy Files
validation_melanoma_file =  path +  "validation/validation-melanoma.npy"
validation_benign_file =  path +  "validation/validation-benign.npy"

# Traansformed Train Numpy Files
trans_train_melanoma_dir =  path +  "transformed_data/train/melanoma/"
trans_train_benign_dir =  path +  "transformed_data/train/benign/"

# Traansformed Validation Numpy Files
trans_validation_melanoma_dir =  path +  "transformed_data/validation/melanoma/"
trans_validation_benign_dir =  path +  "transformed_data/validation/benign/"

# Augment Values
augment_values = {
    "rotation_range" : 40,
    "width_shift_range" : 0.1,
    "height_shift_range" : 0.1,
    "shear_range" : 2,
    "zoom_range" : 0.2,
    "horizontal_flip" : True,
    "vertical_flip" : False,
    "rescale" : 1./255,
}

# Batch Size
batch_size = 32

# vgg16 required image height and width
img_width, img_height = 224, 224

# Sample Size
nb_train_samples = 9216
nb_validation_samples = 2304