from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import pandas as pd

TRAINING_FILE_PATH = "./input/train.csv"
TESTING_FILE_PATH = "./input/test.csv"
ID_COLUMN_NAME = "PassengerId"
LABEL_COLUMN_NAME = "Survived"
LABEL_COLUMN_NAME_IN_SUBMISSION = LABEL_COLUMN_NAME

def perform_categorization(column_vector):
    encoder = LabelEncoder()
    return encoder.fit_transform(column_vector).astype(np.int)

def load_data():
    # Read file content
    training_file_content = pd.read_csv(TRAINING_FILE_PATH)
    testing_file_content = pd.read_csv(TESTING_FILE_PATH)
    combined_file_content = pd.concat([training_file_content, testing_file_content])

    # Feature engineering
    combined_file_content.drop(["Ticket", "Name"], axis=1, inplace=True)
    valid_elements_mask = np.logical_not(pd.isnull(combined_file_content["Cabin"].as_matrix()))
    combined_file_content.loc[valid_elements_mask, "Cabin"] = [item[0] for item in combined_file_content["Cabin"].as_matrix()[valid_elements_mask]]

    # Manipulate file content
    X = combined_file_content.drop([ID_COLUMN_NAME, LABEL_COLUMN_NAME], axis=1).as_matrix()
    categorical_features_mask_list = []
    for column_vector in X.T:
        valid_elements_mask = np.logical_not(pd.isnull(column_vector))
        if np.can_cast(type(column_vector[valid_elements_mask][0]), np.float):
            categorical_features_mask_list.append(False)
            min_value = np.min(column_vector[valid_elements_mask])
            column_vector[np.logical_not(valid_elements_mask)] = min_value - 1
        else:
            categorical_features_mask_list.append(True)
            column_vector[np.logical_not(valid_elements_mask)] = "Missing"
            column_vector[:] = perform_categorization(column_vector)
    encoder = OneHotEncoder(categorical_features=categorical_features_mask_list)
    X = encoder.fit_transform(X).toarray()

    # Separate the data set
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
