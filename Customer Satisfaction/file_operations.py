from itertools import combinations
import numpy as np
import pandas as pd

TRAINING_FILE_PATH = "./input/train.csv"
TESTING_FILE_PATH = "./input/test.csv"
ID_COLUMN_NAME = "ID"
LABEL_COLUMN_NAME = "TARGET"
LABEL_COLUMN_NAME_IN_SUBMISSION = LABEL_COLUMN_NAME

def load_data():
    # Read file content
    training_file_content = pd.read_csv(TRAINING_FILE_PATH)
    testing_file_content = pd.read_csv(TESTING_FILE_PATH)
    combined_file_content = pd.concat([training_file_content, testing_file_content])

    # Remove constant columns
    invalid_column_name_list = []
    for current_column_name in combined_file_content.columns.values:
        current_column_array = combined_file_content[current_column_name].as_matrix()
        if np.unique(current_column_array).size == 1:
            invalid_column_name_list.append(current_column_name)
    combined_file_content.drop(invalid_column_name_list, axis=1, inplace=True)
    print("{:d} constant columns removed.".format(len(invalid_column_name_list)))

    # Remove duplicated columns
    invalid_column_name_list = []
    for current_column_name_1, current_column_name_2 in combinations(combined_file_content.columns.values, 2):
        if current_column_name_1 in invalid_column_name_list or current_column_name_2 in invalid_column_name_list:
            continue

        current_column_array_1 = combined_file_content[current_column_name_1].as_matrix()
        current_column_array_2 = combined_file_content[current_column_name_2].as_matrix()
        if np.array_equal(current_column_array_1, current_column_array_2):
            invalid_column_name_list.append(current_column_name_2)
    combined_file_content.drop(invalid_column_name_list, axis=1, inplace=True)
    print("{:d} duplicated columns removed.".format(len(invalid_column_name_list)))

    # Feature engineering
    combined_file_content["zero_num"] = [np.sum(current_row.drop([ID_COLUMN_NAME, LABEL_COLUMN_NAME]) == 0)
                                         for _, current_row in combined_file_content.iterrows()]

    # Separate the data set
    X = combined_file_content.drop([ID_COLUMN_NAME, LABEL_COLUMN_NAME], axis=1).as_matrix()
    Y = combined_file_content[LABEL_COLUMN_NAME].as_matrix()
    ID = combined_file_content[ID_COLUMN_NAME].as_matrix()
    test_data_mask = pd.isnull(Y)
    X_train = X[np.logical_not(test_data_mask)]
    Y_train = Y[np.logical_not(test_data_mask)]
    X_test = X[test_data_mask]
    ID_test = ID[test_data_mask]

    return X_train, Y_train, X_test, ID_test

def write_submission(ID_test, prediction, submission_file_path):
    submission_file_content = pd.DataFrame({ID_COLUMN_NAME:ID_test, LABEL_COLUMN_NAME_IN_SUBMISSION:prediction})
    submission_file_content.to_csv(submission_file_path, index=False)
