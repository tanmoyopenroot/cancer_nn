import csv
import shutil
import numpy as np

from config import image_folder_path, main_isic_path
from config import csv_file_path, csv_file_name

from config import image_extension

from config import train_data_melanoma_dir, train_data_benign_dir
from config import validation_data_melanoma_dir, validation_data_benign_dir
from config import test_data_melanoma_dir, test_data_benign_dir

from createDir import createDirectory

def randomize(dataset):
    permutation = np.random.permutation(len(dataset))
    shuffled_dataset = dataset[permutation]
    return shuffled_dataset

def readCSV(path, file):
    csv_file = path + file
    cancer_image_name_dataset = []
    non_cancer_image_name_dataset = []  
    count_cancer, count_non_cancer = 0, 0
    try:
        with open(csv_file, "rb") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if int(row["melanoma"]):
                    cancer_image_name_dataset = np.append(cancer_image_name_dataset, row["image_id"])
                    count_cancer += 1
                else:
                    non_cancer_image_name_dataset = np.append(non_cancer_image_name_dataset, row["image_id"])
                    count_non_cancer += 1 
                   
        cancer_image_name_dataset = randomize(cancer_image_name_dataset)
        non_cancer_image_name_dataset = randomize(non_cancer_image_name_dataset)

        non_cancer_image_name_dataset = non_cancer_image_name_dataset[0:count_cancer]

        print("Cancer Images : {0}, Non Cancer Images : {1}".format(len(cancer_image_name_dataset), len(non_cancer_image_name_dataset))) 

    except Exception as e:
        print("Uable To Read The CSV File")

    return cancer_image_name_dataset, non_cancer_image_name_dataset

def seperateData(cancer_image_name_dataset, non_cancer_image_name_dataset):
    validation_len = 72
    train_len = 288
    test_len = 8

    for index, image in enumerate(cancer_image_name_dataset):
        if index < train_len:
            shutil.copy2(image_folder_path + image + image_extension, train_data_melanoma_dir)
        elif index < (train_len + validation_len):
            shutil.copy2(image_folder_path + image + image_extension, validation_data_melanoma_dir)
        else:
            shutil.copy2(image_folder_path + image + image_extension, test_data_melanoma_dir)

    for index, image in enumerate(non_cancer_image_name_dataset):
        if index < train_len:
            shutil.copy2(image_folder_path + image + image_extension, train_data_benign_dir)
        elif index < (train_len + validation_len):
            shutil.copy2(image_folder_path + image + image_extension, validation_data_benign_dir)
        else:
            shutil.copy2(image_folder_path + image + image_extension, test_data_benign_dir)
            

def createDataset():
    createDirectory()

    print("Reading CSV FILE")
    cancer_image_name_dataset, non_cancer_image_name_dataset = readCSV(csv_file_path, csv_file_name)

    print("List Of Melanoma Images")
    print cancer_image_name_dataset[0:10]

    print("List Of Benign Images")
    print non_cancer_image_name_dataset[0:10]

    print("Creating Training, Validation And Testing")
    seperateData(cancer_image_name_dataset, non_cancer_image_name_dataset)
    