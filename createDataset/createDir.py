import os

from config import train_data_melanoma_dir, train_data_benign_dir
from config import validation_data_melanoma_dir, validation_data_benign_dir
from config import test_data_melanoma_dir, test_data_benign_dir

from config import train_aug_melanoma_dir, train_aug_benign_dir
from config import validation_aug_melanoma_dir, validation_aug_benign_dir
from config import train_seg_melanoma_dir, train_seg_benign_dir
from config import validation_seg_melanoma_dir, validation_seg_benign_dir


def createDirectory():
    if not os.path.exists(train_data_melanoma_dir):
        try:
            os.makedirs(train_data_melanoma_dir)
        except Exception as e:
            raise e

    if not os.path.exists(train_data_benign_dir):
        try:
            os.makedirs(train_data_benign_dir)
        except Exception as e:
            raise e


    if not os.path.exists(validation_data_melanoma_dir):
        try:
            os.makedirs(validation_data_melanoma_dir)
        except Exception as e:
            raise e

    if not os.path.exists(validation_data_benign_dir):
        try:
            os.makedirs(validation_data_benign_dir)
        except Exception as e:
            raise e


    if not os.path.exists(test_data_melanoma_dir):
        try:
            os.makedirs(test_data_melanoma_dir)
        except Exception as e:
            raise e

    if not os.path.exists(test_data_benign_dir):
        try:
            os.makedirs(test_data_benign_dir)
        except Exception as e:
            raise e
                        

    if not os.path.exists(train_aug_melanoma_dir):
        try:
            os.makedirs(train_aug_melanoma_dir)
        except Exception as e:
            raise e

    if not os.path.exists(train_aug_benign_dir):
        try:
            os.makedirs(train_aug_benign_dir)
        except Exception as e:
            raise e


    if not os.path.exists(validation_aug_melanoma_dir):
        try:
            os.makedirs(validation_aug_melanoma_dir)
        except Exception as e:
            raise e

    if not os.path.exists(validation_aug_benign_dir):
        try:
            os.makedirs(validation_aug_benign_dir)
        except Exception as e:
            raise e


    if not os.path.exists(train_seg_melanoma_dir):
        try:
            os.makedirs(train_seg_melanoma_dir)
        except Exception as e:
            raise e

    if not os.path.exists(train_seg_benign_dir):
        try:
            os.makedirs(train_seg_benign_dir)
        except Exception as e:
            raise e

    if not os.path.exists(validation_seg_melanoma_dir):
        try:
            os.makedirs(validation_seg_melanoma_dir)
        except Exception as e:
            raise e

    if not os.path.exists(validation_seg_benign_dir):
        try:
            os.makedirs(validation_seg_benign_dir)
        except Exception as e:
            raise e           