import numpy as np
import pandas as pd

TRAINING_FILE_PATH = "./input/train.csv"
TESTING_FILE_PATH = "./input/test.csv"
ID_COLUMN_NAME = "ID"
LABEL_COLUMN_NAME = "label"
LABEL_COLUMN_NAME_IN_SUBMISSION = LABEL_COLUMN_NAME

def load_data():
    # Read file content
    training_file_content = pd.read_csv(TRAINING_FILE_PATH)
    testing_file_content = pd.read_csv(TESTING_FILE_PATH)

    # Separate the training and testing data set
    X_train = training_file_content.drop([LABEL_COLUMN_NAME], axis=1).as_matrix().astype("float32")
    Y_train = training_file_content[LABEL_COLUMN_NAME].as_matrix()
    X_test = testing_file_content.as_matrix().astype("float32")
    submission_file_content = pd.DataFrame({ID_COLUMN_NAME:np.arange(testing_file_content.shape[0]),
                                            LABEL_COLUMN_NAME_IN_SUBMISSION:np.zeros(testing_file_content.shape[0])})

    return (X_train, Y_train, X_test, submission_file_content)
