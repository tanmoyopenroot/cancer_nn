import csv
import shutil
import numpy as np

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

def seperateData(from_path, to_path, image_extension, cancer_image_name_dataset, non_cancer_image_name_dataset):
    validation_len = 72
    train_len = 288
    test_len = 8

    train_melanoma_path = to_path + "train/melanoma/"
    train_benign_path = to_path + "train/benign/"

    validation_melanoma_path = to_path + "validation/melanoma/"
    validation_benign_path = to_path + "validation/benign/"

    test_melanoma_path = to_path + "test/melanoma/"
    test_benign_path = to_path + "test/benign/"

    for index, image in enumerate(cancer_image_name_dataset):
        if index < train_len:
            shutil.copy2(from_path + image + image_extension, train_melanoma_path)
        elif index < (train_len + validation_len):
            shutil.copy2(from_path + image + image_extension, validation_melanoma_path)
        else:
            shutil.copy2(from_path + image + image_extension, test_melanoma_path)

    for index, image in enumerate(non_cancer_image_name_dataset):
        if index < train_len:
            shutil.copy2(from_path + image + image_extension, train_benign_path)
        elif index < (train_len + validation_len):
            shutil.copy2(from_path + image + image_extension, validation_benign_path)
        else:
            shutil.copy2(from_path + image + image_extension, test_benign_path)
            

def main():
    image_folder_path = "/home/openroot/Tanmoy/Working Stuffs/myStuffs/ISIC-2017/ISIC-2017/"
    main_isic_path = "/home/openroot/Tanmoy/Working Stuffs/myStuffs/havss-tf/ISIC-2017/data/"

    csv_file_path = "/home/openroot/Tanmoy/Working Stuffs/myStuffs/havss-tf/ISIC-2017/data/"
    csv_file_name = "ISIC-2017-label.csv"

    image_extension = ".jpg"
    resized_image_size = (100, 70)
    pixel_depth = 255

    print("Reading CSV FILE")
    cancer_image_name_dataset, non_cancer_image_name_dataset = readCSV(csv_file_path, csv_file_name)

    print("List Of Melanoma Images")
    print cancer_image_name_dataset[0:10]

    print("List Of Benign Images")
    print non_cancer_image_name_dataset[0:10]

    print("Creating Training, Validation And Testing")
    seperateData(image_folder_path, main_isic_path, image_extension, cancer_image_name_dataset, non_cancer_image_name_dataset)

if __name__ == '__main__':
    main()
    
