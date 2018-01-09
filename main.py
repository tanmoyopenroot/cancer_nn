import argparse
import sys

from config import train_data_melanoma_dir, train_data_benign_dir
from config import validation_data_melanoma_dir, validation_data_benign_dir
from config import test_data_melanoma_dir, test_data_benign_dir

from config import image_extension

parser = argparse.ArgumentParser()
parser.add_argument(
	"mode", 
	help = "Create Dataset - create, Train CNN - train, Test CNN - test", 
	nargs = "+",
	choices = ["create", "train", "test"]
)

args = parser.parse_args()

print("Skin Cancer Config")

print("Training Melanoma Path : {0}".format(train_data_melanoma_dir))
print("Training Benign Path : {0}".format(train_data_melanoma_dir))

print("Validation Melanoma Path : {0}".format(validation_data_melanoma_dir))
print("Validation Benign Path : {0}".format(validation_data_benign_dir))

print("Testing Melanoma Path : {0}".format(test_data_melanoma_dir))
print("Testing Benign Path : {0}".format(test_data_benign_dir))

print("Image Extension : {0}".format(image_extension))