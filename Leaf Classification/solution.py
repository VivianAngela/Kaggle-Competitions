import os
import numpy as np
import pandas as pd

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
TRAIN_FILE_PATH = os.path.join(INPUT_FOLDER_PATH, "train.csv")
TEST_FILE_PATH = os.path.join(INPUT_FOLDER_PATH, "test.csv")
SUBMISSION_FOLDER_PATH = os.path.join(DATASET_FOLDER_PATH, "submission")
ID_COLUMN_NAME = "id"
LABEL_COLUMN_NAME = "species"

# Model Structure
BLOCK_NUM = 3
DENSE_DIM = 512
DROPOUT_RATIO = 0.8

# Training Procedure
CROSS_VALIDATION_NUM = 20
MAXIMUM_EPOCH_NUM = 5000
TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 2048

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

def run():
    # Read file content
    train_file_content = pd.read_csv(TRAIN_FILE_PATH)
    test_file_content = pd.read_csv(TEST_FILE_PATH)

    # Perform scaling
    feature_column_list = list(train_file_content.drop([ID_COLUMN_NAME, LABEL_COLUMN_NAME], axis=1))
    for feature_keyword in ["shape", "texture", "margin"]:
        selected_feature_column_list = [feature_column for feature_column in feature_column_list
                                        if feature_keyword in feature_column]

        max_value = np.max(train_file_content[selected_feature_column_list].as_matrix())
        min_value = np.min(train_file_content[selected_feature_column_list].as_matrix())

        train_file_content[selected_feature_column_list] = (train_file_content[selected_feature_column_list] - min_value) / (max_value - min_value)
        test_file_content[selected_feature_column_list] = (test_file_content[selected_feature_column_list] - min_value) / (max_value - min_value)

    # Split data
    train_id_with_species = train_file_content[[ID_COLUMN_NAME, LABEL_COLUMN_NAME]].as_matrix()
    train_id_array, train_species_array = np.transpose(train_id_with_species)
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

    print("All done!")

if __name__ == "__main__":
    run()
