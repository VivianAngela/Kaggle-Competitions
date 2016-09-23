import os
import glob
import numpy as np
import pandas as pd

from skimage import img_as_ubyte
from skimage.feature import local_binary_pattern
from skimage.io import imread
from keras.callbacks import ModelCheckpoint
from keras.layers.advanced_activations import PReLU
from keras.layers.core import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.preprocessing import LabelBinarizer

# Data Set
DATASET_FOLDER_PATH = "./"
INPUT_FOLDER_PATH = os.path.join(DATASET_FOLDER_PATH, "input")
IMAGE_FOLDER_PATH = os.path.join(INPUT_FOLDER_PATH, "images")
TRAIN_FILE_PATH = os.path.join(INPUT_FOLDER_PATH, "train.csv")
TEST_FILE_PATH = os.path.join(INPUT_FOLDER_PATH, "test.csv")
SUBMISSION_FOLDER_PATH = os.path.join(DATASET_FOLDER_PATH, "submission")
ID_COLUMN_NAME = "id"
LABEL_COLUMN_NAME = "species"
IMAGE_EXTENSION = ".jpg"
FEATURE_EXTENSION = "_LBP.csv"
FEATURE_NAME_PREFIX = "LBP"

# Model Structure
BLOCK_NUM = 3
DENSE_DIM = 512
DROPOUT_RATIO = 0.8

# Training Procedure
CROSS_VALIDATION_NUM = 20
MAXIMUM_EPOCH_NUM = 5000
TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 2048

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
        print("Unable to retrieve LBP feature histogram in {}.".format(os.path.basename(image_path)))
        return None

def add_LBP_feature_histogram(file_content):
    LBP_feature_histogram_list = []
    id_array = file_content[ID_COLUMN_NAME].as_matrix()
    for id_value in id_array:
        image_path = os.path.join(IMAGE_FOLDER_PATH, "{}{}".format(id_value, IMAGE_EXTENSION))
        LBP_feature_histogram = retrieve_LBP_feature_histogram(image_path)
        LBP_feature_histogram_list.append(LBP_feature_histogram)

    LBP_feature_histogram_dim = len(LBP_feature_histogram_list[0])
    LBP_feature_histogram_array = np.array(LBP_feature_histogram_list)
    for entry_index in range(LBP_feature_histogram_dim):
        feature_name = FEATURE_NAME_PREFIX + str(entry_index + 1)
        file_content[feature_name] = LBP_feature_histogram_array[:, entry_index]

def init_model(feature_dim, label_num):
    model = Sequential()

    for block_index in range(BLOCK_NUM):
        if block_index == 0:
            model.add(Dense(DENSE_DIM, input_dim=feature_dim))
        else:
            model.add(Dense(DENSE_DIM))

        model.add(PReLU())
        model.add(BatchNormalization())
        model.add(Dropout(DROPOUT_RATIO))

    model.add(Dense(label_num, activation="softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="adadelta", metrics=["accuracy"])

    return model

def ensemble_predictions():
    def _ensemble_predictions(ensemble_func, ensemble_submission_file_name):
        ensemble_proba = ensemble_func(proba_array, axis=0)
        ensemble_proba = ensemble_proba / np.sum(ensemble_proba, axis=1)[:, np.newaxis]
        ensemble_submission_file_content.loc[:, proba_columns] = ensemble_proba
        ensemble_submission_file_content.to_csv(os.path.join(SUBMISSION_FOLDER_PATH, ensemble_submission_file_name), index=False)

    # Read predictions
    submission_file_path_list = glob.glob(os.path.join(SUBMISSION_FOLDER_PATH, "submission_*.csv"))
    submission_file_content_list = [pd.read_csv(submission_file_path) for submission_file_path in submission_file_path_list]
    ensemble_submission_file_content = submission_file_content_list[0]

    # Concatenate predictions
    proba_columns = list(set(ensemble_submission_file_content) - {ID_COLUMN_NAME})
    proba_list = [np.expand_dims(submission_file_content.as_matrix(proba_columns), axis=0)
                  for submission_file_content in submission_file_content_list]
    proba_array = np.vstack(proba_list)

    # Ensemble predictions
    for ensemble_func, ensemble_submission_file_name in \
        zip([np.max, np.min, np.mean, np.median], ["max.csv", "min.csv", "mean.csv", "median.csv"]):
        _ensemble_predictions(ensemble_func, ensemble_submission_file_name)

def run():
    # Read file content
    train_file_content = pd.read_csv(TRAIN_FILE_PATH)
    test_file_content = pd.read_csv(TEST_FILE_PATH)

    # Add LBP feature histogram
    add_LBP_feature_histogram(train_file_content)
    add_LBP_feature_histogram(test_file_content)

    # Perform scaling
    feature_column_list = list(train_file_content.drop([ID_COLUMN_NAME, LABEL_COLUMN_NAME], axis=1))
    for feature_keyword in ["shape", "texture", "margin", FEATURE_NAME_PREFIX]:
        selected_feature_column_list = [feature_column for feature_column in feature_column_list
                                        if feature_keyword in feature_column]

        max_value = np.max(train_file_content[selected_feature_column_list].as_matrix())
        min_value = np.min(train_file_content[selected_feature_column_list].as_matrix())

        train_file_content[selected_feature_column_list] = (train_file_content[selected_feature_column_list] - min_value) / (max_value - min_value)
        test_file_content[selected_feature_column_list] = (test_file_content[selected_feature_column_list] - min_value) / (max_value - min_value)

    # Split data
    train_species_array = train_file_content[LABEL_COLUMN_NAME].as_matrix()
    train_X = train_file_content.drop([ID_COLUMN_NAME, LABEL_COLUMN_NAME], axis=1).as_matrix()
    test_id_array = test_file_content[ID_COLUMN_NAME].as_matrix()
    test_X = test_file_content.drop([ID_COLUMN_NAME], axis=1).as_matrix()

    # Encode labels
    label_binarizer = LabelBinarizer()
    train_Y = label_binarizer.fit_transform(train_species_array)

    # Initiate model
    model = init_model(train_X.shape[1], len(label_binarizer.classes_))
    vanilla_weights = model.get_weights()

    # Cross validation
    cross_validation_iterator = StratifiedShuffleSplit(train_species_array,
                                n_iter=CROSS_VALIDATION_NUM, test_size=0.2, random_state=0)
    for cross_validation_index, (train_index, valid_index) in enumerate(cross_validation_iterator, start=1):
        print("Working on {}/{} ...".format(cross_validation_index, CROSS_VALIDATION_NUM))

        optimal_weights_path = "/tmp/Optimal_Weights_{}.h5".format(cross_validation_index)
        submission_file_path = os.path.join(SUBMISSION_FOLDER_PATH, "submission_{}.csv".format(cross_validation_index))

        if os.path.isfile(submission_file_path):
            continue

        if not os.path.isfile(optimal_weights_path):
            # Load the vanilla weights
            model.set_weights(vanilla_weights)

            # Perform the training procedure
            modelcheckpoint_callback = ModelCheckpoint(optimal_weights_path, monitor="val_loss", save_best_only=True)
            model.fit(train_X[train_index], train_Y[train_index],
                      batch_size=TRAIN_BATCH_SIZE, nb_epoch=MAXIMUM_EPOCH_NUM,
                      validation_data=(train_X[valid_index], train_Y[valid_index]),
                      callbacks=[modelcheckpoint_callback], verbose=2)

        # Load the optimal weights
        model.load_weights(optimal_weights_path)

        # Perform the testing procedure
        test_probabilities = model.predict_proba(test_X, batch_size=TEST_BATCH_SIZE, verbose=2)

        # Save submission to disk
        if not os.path.isdir(SUBMISSION_FOLDER_PATH):
            os.makedirs(SUBMISSION_FOLDER_PATH)
        submission_file_content = pd.DataFrame(test_probabilities, columns=label_binarizer.classes_)
        submission_file_content[ID_COLUMN_NAME] = test_id_array
        submission_file_content.to_csv(submission_file_path, index=False)

    # Perform ensembling
    ensemble_predictions()

    print("All done!")

if __name__ == "__main__":
    run()
