from __future__ import absolute_import, division, print_function

import os
import fnmatch
import numpy as np
from scipy.misc import imread, imresize, imsave

PROJECT_NAME = "Cervical Cancer Screening"
PROJECT_FOLDER_PATH = os.path.join(os.path.expanduser("~"), "Documents/datasets", PROJECT_NAME)
ORIGINAL_DATASET_FOLDER_PATH = os.path.join(PROJECT_FOLDER_PATH, "original")
PROCESSED_DATASET_FOLDER_PATH = os.path.join(PROJECT_FOLDER_PATH, "processed")
PROCESSED_IMAGE_HEIGHT, PROCESSED_IMAGE_WIDTH = 300, 224

def get_certain_files_recursively_within_folder(root_folder_path, file_name_rule):
    for folder_path, _, file_name_list in os.walk(root_folder_path):
        for file_name in fnmatch.filter(file_name_list, file_name_rule):
            yield os.path.join(folder_path, file_name)

def perform_preprocessing(original_image_file_path, processed_image_file_path):
    try:
        original_image = imread(original_image_file_path)
        original_image_height, original_image_width = original_image.shape[:2]

        if (PROCESSED_IMAGE_HEIGHT > PROCESSED_IMAGE_WIDTH) != (original_image_height > original_image_width):
            original_image = np.swapaxes(original_image, 0, 1)

        processed_image_parent_folder_path = os.path.dirname(processed_image_file_path)
        if not os.path.isdir(processed_image_parent_folder_path):
            os.makedirs(processed_image_parent_folder_path)

        imsave(processed_image_file_path, imresize(original_image, (PROCESSED_IMAGE_HEIGHT, PROCESSED_IMAGE_WIDTH)))
        assert os.path.isfile(processed_image_file_path)
    except Exception as exception:
        print("[WARNING]: exception for %s: %s" % (original_image_file_path[len(ORIGINAL_DATASET_FOLDER_PATH):], exception))

def run():
    print("[INFO]: resizing and rotating images ...")

    print("[INFO]: original folder: %s" % ORIGINAL_DATASET_FOLDER_PATH)
    print("[INFO]: processed folder: %s" % PROCESSED_DATASET_FOLDER_PATH)

    original_image_file_path_list = list(get_certain_files_recursively_within_folder(ORIGINAL_DATASET_FOLDER_PATH, "*.jpg"))
    for original_image_file_index, original_image_file_path in enumerate(original_image_file_path_list, start=1):
        print("[INFO]: working on image %s/%s ..." % (original_image_file_index, len(original_image_file_path_list)))

        processed_image_file_path = PROCESSED_DATASET_FOLDER_PATH + original_image_file_path[len(ORIGINAL_DATASET_FOLDER_PATH):]
        if not os.path.isfile(processed_image_file_path):
            perform_preprocessing(original_image_file_path, processed_image_file_path)

    print("[INFO]: edited all images, exit!")

if __name__ == "__main__":
    run()
