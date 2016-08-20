from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import pandas as pd

TRAINING_FILE_PATH = "./input/train.csv"
TESTING_FILE_PATH = "./input/test.csv"
ID_COLUMN_NAME = "ID"
LABEL_COLUMN_NAME = "target"
LABEL_COLUMN_NAME_IN_SUBMISSION = "PredictedProb"

def perform_categorization(column_vector):
    encoder = LabelEncoder()
    return encoder.fit_transform(column_vector).astype(np.int)

def convert_hexavigesimal_value(original_string):
    if pd.isnull(original_string):
        return np.nan
    value_list = [ord(item) - ord("A") + 1 for item in original_string]
    weight_list = [26 ** item for item in range(len(original_string) - 1, 0 - 1, -1)]
    return np.dot(value_list, weight_list)

def load_data():
    # Read file content
    training_file_content = pd.read_csv(TRAINING_FILE_PATH)
    testing_file_content = pd.read_csv(TESTING_FILE_PATH)
    combined_file_content = pd.concat([training_file_content, testing_file_content])

    # Drop the ID and Label columns
    Y = combined_file_content[LABEL_COLUMN_NAME].as_matrix()
    combined_file_content.drop([ID_COLUMN_NAME, LABEL_COLUMN_NAME], axis=1, inplace=True)

    # Feature engineering
    combined_file_content["null_num"] = np.sum(pd.isnull(combined_file_content), axis=1)
    combined_file_content["v22"] = combined_file_content["v22"].apply(convert_hexavigesimal_value)
    combined_file_content.drop(["v8", "v25", "v36", "v37", "v46", "v51", "v53", "v54", "v63", "v73",
                                "v81", "v82", "v89", "v92", "v95", "v105", "v107", "v108", "v109", "v116",
                                "v117", "v118", "v119", "v123", "v124", "v128"], axis=1, inplace=True)

    # Manipulate file content
    X = combined_file_content.as_matrix()
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

    # Separate the training and testing data set
    valid_elements_mask = np.logical_not(pd.isnull(Y))
    X_train = X[valid_elements_mask]
    Y_train = Y[valid_elements_mask]
    X_test = X[np.logical_not(valid_elements_mask)]
    submission_file_content = pd.DataFrame({ID_COLUMN_NAME:testing_file_content[ID_COLUMN_NAME],
                                            LABEL_COLUMN_NAME_IN_SUBMISSION:np.zeros(testing_file_content.shape[0])})

    return X_train, Y_train, X_test, submission_file_content
