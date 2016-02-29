import numpy as np
import os
import pandas as pd
import shutil

# The path of the folder where the scripts are saved
SCRIPTS_FOLDER_PATH = os.path.dirname(os.path.realpath(__file__))

# Variables related to the data sets
DATA_PATH = os.path.join(SCRIPTS_FOLDER_PATH, "input")
TRAINING_DATASET_NAME = "train"
TESTING_DATASET_NAME = "test"
TESTING_FILE_NAME = "pairs.csv"
BBOX_EXTENSION = "_bbox.csv"

# The size of facial images
FACIAL_IMAGE_SIZE = 300

# The path of OpenFace
OPENFACE_PATH = "/opt/openface"

# Variables related to VGG Face
VGG_FACE_PATH = "/opt/vggface"
MODEL_DEFINITION_FILE_NAME = "VGG_FACE_deploy.prototxt"
TRAINED_MODEL_FILE_NAME = "VGG_FACE.caffemodel"
VGG_FACE_IMAGE_SIZE = 224

# Variables related to congealingcomplex
CONGEALINGCOMPLEX_PATH = "/opt/congealingcomplex"

# Variables related to Keras and scikit-learn models
KERAS_MODEL_EXTENSION = ".hdf5"
SCIKIT_LEARN_EXTENSION = ".pkl"

# The path of the folder where the submission files are saved
SUBMISSIONS_FOLDER_PATH = os.path.join(SCRIPTS_FOLDER_PATH, "Submissions")

# The file name of the GroundTruth. This file is saved at SUBMISSIONS_FOLDER_PATH.
GROUNDTRUTH_FILE_NAME = "GroundTruth.csv"

def read_from_file(file_path):
    file_content = pd.read_csv(file_path, delimiter=",", engine="c", header=None, \
                               na_filter=False, dtype=np.float64, low_memory=False).as_matrix()
    return np.squeeze(file_content)

def write_to_file(file_path, file_content):
    pd.Series(file_content).to_csv(file_path, header=False, index=False)

def get_working_directory(description):
    """Get the path of working directory.
    
    :param description: the folder name of the working directory
    :type description: string
    :return: the path of working directory
    :rtype: string
    """

    return os.path.join("/tmp", description)

def reset_working_directory(description):
    """Reset the working directory.
    
    :param description: the folder name of the working directory
    :type description: string
    :return: the working directory will be reset
    :rtype: None
    """

    print("Resetting working directory: {}".format(description))

    # Remove the old directory and create a new one
    working_directory = get_working_directory(description)
    shutil.rmtree(working_directory, ignore_errors=True)
    os.makedirs(working_directory)
