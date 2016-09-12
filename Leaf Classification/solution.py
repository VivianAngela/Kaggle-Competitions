import os
import numpy as np
import pandas as pd

from skimage.io import imread
from skimage.transform import resize
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

# Image Processing
IMAGE_SIZE = 64

# Data Set
DATASET_PATH = "./input"

def preprocess_image(image_path):
    try:
        # Read the image as a binary image
        image = imread(image_path)

        # Omit totally black rows and columns
        image = image[np.sum(image, axis=1) != 0][:, np.sum(image, axis=0) != 0]

        # Further processing
        row_num, column_num = image.shape
        higher_num = max(image.shape)
        row_start_index = int((higher_num - row_num) / 2)
        column_start_index = int((higher_num - column_num) / 2)
        final_image = np.zeros((higher_num, higher_num))
        final_image[row_start_index:row_start_index + row_num,
                    column_start_index:column_start_index + column_num] = image
        final_image = resize(final_image, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.bool).astype(np.float32)
        return final_image
    except:
        return None

def run():
    # Read file content of the training data set
    training_file_content = pd.read_csv(os.path.join(DATASET_PATH, "train.csv"))
    id_with_species = training_file_content[["id", "species"]].as_matrix()
    id_array, species_array = id_with_species[:, 0], id_with_species[:, 1]

    # Encode labels
    label_encoder = LabelEncoder()
    encoded_species_array = label_encoder.fit_transform(species_array)

    # Cross validation
    cross_validation_iterator = StratifiedShuffleSplit(encoded_species_array, n_iter=1, test_size=0.2, random_state=0)
    for train_index_array, test_index_array in cross_validation_iterator:
        break

    print("All done!")

if __name__ == "__main__":
    run()
