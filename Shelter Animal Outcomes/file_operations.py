from dateutil.parser import parse
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import pandas as pd

TRAINING_FILE_PATH = "./input/train.csv"
TESTING_FILE_PATH = "./input/test.csv"
ID_COLUMN_NAME = "ID"
LABEL_COLUMN_NAME = "OutcomeType"
LABEL_COLUMN_NAME_LIST_IN_SUBMISSION = None

def perform_categorization(column_vector):
    encoder = LabelEncoder()
    return encoder.fit_transform(column_vector).astype(np.int)

def get_age_in_months(input):
    try:
        input = str(input)
        number = int(input.split()[0])
        if "year" in input:
            return number * 12
        if "month" in input:
            return number
        if "week" in input:
            return number / 4
        if "day" in input:
            return number / 30
    except:
        return np.nan

def load_data():
    # Read file content
    training_file_content = pd.read_csv(TRAINING_FILE_PATH)
    testing_file_content = pd.read_csv(TESTING_FILE_PATH)
    combined_file_content = pd.concat([training_file_content, testing_file_content])

    # Feature engineering
    combined_file_content.drop(["AnimalID", "OutcomeSubtype"], axis=1, inplace=True)

    combined_file_content["has_no_name"] = pd.isnull(combined_file_content["Name"])
    combined_file_content["Name"].fillna("", inplace=True)
    combined_file_content["name_len"] = combined_file_content.Name.apply(len)
    combined_file_content.drop(["Name"], axis=1, inplace=True)

    datetime_list = [parse(current_record) for current_record in combined_file_content['DateTime']]
    combined_file_content["DateTime_year"] = [current_datetime.year for current_datetime in datetime_list]
    combined_file_content["DateTime_month"] = [current_datetime.month for current_datetime in datetime_list]
    combined_file_content["DateTime_day"] = [current_datetime.day for current_datetime in datetime_list]
    combined_file_content["DateTime_hour"] = [current_datetime.hour for current_datetime in datetime_list]
    combined_file_content.drop(["DateTime"], axis=1, inplace=True)

    combined_file_content["SexuponOutcome"].fillna("Unknown", inplace=True)
    combined_file_content["is_female"] = ["Female" in current_record for current_record in combined_file_content['SexuponOutcome']]
    combined_file_content["is_male"] = ["Male" in current_record for current_record in combined_file_content['SexuponOutcome']]
    combined_file_content["is_intact"] = ["Intact" in current_record for current_record in combined_file_content['SexuponOutcome']]

    combined_file_content["AgeuponOutcome"].fillna("-1 years", inplace=True)
    combined_file_content['age_in_months'] = combined_file_content.AgeuponOutcome.apply(get_age_in_months)

    combined_file_content["mixed_breed"] = ["Mix" in current_record for current_record in combined_file_content['Breed']]
    combined_file_content.drop(["Breed"], axis=1, inplace=True)

    combined_file_content["mixed_color"] = ["/" in current_record for current_record in combined_file_content['Color']]
    combined_file_content.drop(["Color"], axis=1, inplace=True)

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

    # Convert labels to numerical values
    global LABEL_COLUMN_NAME_LIST_IN_SUBMISSION
    encoder = LabelEncoder()
    Y_train = encoder.fit_transform(Y_train)
    LABEL_COLUMN_NAME_LIST_IN_SUBMISSION = encoder.classes_

    return X_train, Y_train, X_test, ID_test

def write_submission(ID_test, prediction, submission_file_path):
    submission_file_content = pd.DataFrame(data=prediction, columns=LABEL_COLUMN_NAME_LIST_IN_SUBMISSION)
    submission_file_content[ID_COLUMN_NAME] = ID_test
    submission_file_content.to_csv(submission_file_path, index=False)
