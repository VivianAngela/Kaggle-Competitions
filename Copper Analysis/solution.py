from scipy import stats
from skimage import img_as_ubyte
from skimage.feature import local_binary_pattern
from skimage.io import imread
import glob
import keras_NN
import numpy as np
import os
import pandas as pd
import time

# Define the global variables related to the dataset
DATASET_PATH = "./input"
TRAINING_FOLDER_NAME = "train"
TESTING_FOLDER_NAME = "test"
TRAINING_FILE_NAME = "train.csv"
TESTING_FILE_NAME = "test.csv"
IMAGE_EXTENSION = ".jpg"
FEATURE_EXTENSION = "_LBP.csv"

def load_image_path_list():
    training_image_path_rule = os.path.join(DATASET_PATH, TRAINING_FOLDER_NAME, "*" + IMAGE_EXTENSION)
    testing_image_path_rule = os.path.join(DATASET_PATH, TESTING_FOLDER_NAME, "*" + IMAGE_EXTENSION)

    training_image_path_list = glob.glob(training_image_path_rule)
    testing_image_path_list = glob.glob(testing_image_path_rule)

    return (training_image_path_list, testing_image_path_list)

def retrieve_LBP_feature_histogram(image_path):
    try:
        # Read feature directly from file
        image_feature_path = image_path + FEATURE_EXTENSION
        if os.path.isfile(image_feature_path):
            LBP_feature_histogram = np.genfromtxt(image_feature_path, delimiter=",")
            return LBP_feature_histogram

        # Define LBP parameters
        radius = 5
        n_points = 8
        bins_num = pow(2, n_points)
        LBP_value_range = (0, pow(2, n_points) - 1)

        # Retrieve feature
        assert os.path.isfile(image_path)
        image_content_in_gray = imread(image_path, as_grey=True)
        image_content_in_gray = img_as_ubyte(image_content_in_gray)
        LBP_feature = local_binary_pattern(image_content_in_gray, n_points, radius)
        LBP_feature_histogram, _ = np.histogram(LBP_feature, bins=bins_num, range=LBP_value_range, density=True)

        # Save feature to file
        assert LBP_feature_histogram is not None
        np.savetxt(image_feature_path, LBP_feature_histogram, delimiter=",")
        return LBP_feature_histogram
    except:
        print("Unable to retrieve LBP feature histogram in %s." % (os.path.basename(image_path)))
        return None

def load_features(image_path_list):
    feature_dict = {}

    for image_path in image_path_list:
        LBP_feature_histogram = retrieve_LBP_feature_histogram(image_path)
        feature_dict[os.path.basename(image_path)] = LBP_feature_histogram

    return feature_dict

def load_csv_files():
    training_file_path = os.path.join(DATASET_PATH, TRAINING_FILE_NAME)
    testing_file_path = os.path.join(DATASET_PATH, TESTING_FILE_NAME)

    training_file_content = pd.read_csv(training_file_path, skiprows=0).as_matrix()
    training_names = training_file_content[:, 0]
    training_labels = training_file_content[:, 1]
    training_labels = training_labels.astype(np.uint32)

    testing_file_content = pd.read_csv(testing_file_path, skiprows=0).as_matrix()
    testing_names = testing_file_content[:, 0]

    return (training_names, training_labels, testing_names)

def get_attributes(feature_dict, names):
    feature_list = []
    for name in names:
        feature_list.append(feature_dict[name])
    return np.array(feature_list)

def run():
    # Load image paths in the dataset
    training_image_path_list, testing_image_path_list = load_image_path_list()

    # Load features
    training_image_feature_dict = load_features(training_image_path_list)
    testing_image_feature_dict = load_features(testing_image_path_list)

    # Load training labels
    training_names, training_labels, testing_names = load_csv_files()

    # Convert data to suitable form for training/testing phase
    X_train = get_attributes(training_image_feature_dict, training_names)
    Y_train = training_labels
    X_test = get_attributes(testing_image_feature_dict, testing_names)

    # Generate prediction list
    prediction_list = []
    for trial_index in range(11):
        print("Working on trial NO.{:d}".format(trial_index + 1))
        current_prediction = keras_NN.generate_prediction(X_train, Y_train, X_test)
        prediction_list.append(current_prediction)

    # Generate ensemble prediction
    ensemble_prediction, _ = stats.mode(prediction_list)
    ensemble_prediction = np.squeeze(ensemble_prediction)

    # Create submission file
    submission_file_name = "Aurora_" + str(int(time.time())) + ".csv"
    file_content = pd.DataFrame({"Id": testing_names, "Prediction": ensemble_prediction})
    file_content.to_csv(submission_file_name, index=False, header=True)

    print("All done!")

if __name__ == "__main__":
    run()
