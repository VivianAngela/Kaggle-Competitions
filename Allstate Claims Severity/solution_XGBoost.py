import os
import glob
import numpy as np
import pandas as pd

from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import ShuffleSplit

# Data Set
DATASET_FOLDER_PATH = "./"
INPUT_FOLDER_PATH = os.path.join(DATASET_FOLDER_PATH, "input")
TRAIN_FILE_PATH = os.path.join(INPUT_FOLDER_PATH, "train.csv")
TEST_FILE_PATH = os.path.join(INPUT_FOLDER_PATH, "test.csv")
SUBMISSION_FOLDER_PATH = os.path.join(DATASET_FOLDER_PATH, "submission")
ID_COLUMN_NAME = "id"
LABEL_COLUMN_NAME = "loss"

# Training Procedure
CROSS_VALIDATION_NUM = 10
N_ESTIMATORS = 1000000
EARLY_STOPPING_ROUNDS = 200

def load_data():
    # Read file content
    train_file_content = pd.read_csv(TRAIN_FILE_PATH)
    test_file_content = pd.read_csv(TEST_FILE_PATH)
    combined_file_content = pd.concat([train_file_content, test_file_content])
    del(train_file_content, test_file_content)
    train_data_mask = combined_file_content[LABEL_COLUMN_NAME].notnull().as_matrix()
    test_data_mask = combined_file_content[LABEL_COLUMN_NAME].isnull().as_matrix()

    # Seperate the feature columns
    feature_column_list = list(combined_file_content.drop([ID_COLUMN_NAME, LABEL_COLUMN_NAME], axis=1))
    categorical_feature_column_list = [feature_column for feature_column in feature_column_list if feature_column.startswith("cat")]

    # Process categorical features: remove obsolete unique values and factorize the values
    for categorical_feature_column in categorical_feature_column_list:
        unique_train_data_array = combined_file_content[categorical_feature_column][train_data_mask].unique()
        unique_test_data_array = combined_file_content[categorical_feature_column][test_data_mask].unique()
        unique_data_array_to_discard = np.setdiff1d(np.union1d(unique_train_data_array, unique_test_data_array),
                                                    np.intersect1d(unique_train_data_array, unique_test_data_array))
        if len(unique_data_array_to_discard) > 0:
            discard_function = lambda input_value: np.nan if input_value in unique_data_array_to_discard else input_value
            combined_file_content[categorical_feature_column] = combined_file_content[categorical_feature_column].apply(discard_function)
        combined_file_content[categorical_feature_column], _ = pd.factorize(combined_file_content[categorical_feature_column])
        combined_file_content[categorical_feature_column] -= np.min(combined_file_content[categorical_feature_column])

    # Separate the training and testing data set
    X_array = combined_file_content.drop([ID_COLUMN_NAME, LABEL_COLUMN_NAME], axis=1).as_matrix()
    Y_array = combined_file_content[LABEL_COLUMN_NAME].as_matrix()
    ID_array = combined_file_content[ID_COLUMN_NAME].as_matrix()
    X_train = X_array[train_data_mask]
    Y_train = Y_array[train_data_mask]
    X_test = X_array[test_data_mask]
    ID_test = ID_array[test_data_mask]
    submission_file_content = pd.DataFrame({ID_COLUMN_NAME:ID_test, LABEL_COLUMN_NAME:np.zeros(ID_test.shape[0])})

    return X_train, Y_train, X_test, submission_file_content

def ensemble_predictions():
    def _ensemble_predictions(ensemble_func, ensemble_submission_file_name):
        ensemble_proba = ensemble_func(proba_array, axis=0)
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
    # Load data set
    X_train, Y_train, X_test, submission_file_content = load_data()
    Y_train = np.log(Y_train + 200)

    # Cross validation
    cross_validation_iterator = ShuffleSplit(n_splits=CROSS_VALIDATION_NUM, test_size=0.1, random_state=0)
    for cross_validation_index, (train_index, valid_index) in enumerate(cross_validation_iterator.split(X_train), start=1):
        print("Working on {}/{} ...".format(cross_validation_index, CROSS_VALIDATION_NUM))

        submission_file_path = os.path.join(SUBMISSION_FOLDER_PATH, "submission_{}.csv".format(cross_validation_index))

        if os.path.isfile(submission_file_path):
            continue

        model = XGBRegressor(
            learning_rate=0.01,
            max_depth=12,
            n_estimators=N_ESTIMATORS,
            silent=False,
            objective="reg:linear",
            gamma=1,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.5,
            reg_alpha=1,
            seed=cross_validation_index,
            nthread=-1)

        model.fit(X_train[train_index], Y_train[train_index], eval_set=[(X_train[valid_index], Y_train[valid_index])],
            eval_metric=lambda y_predicted, y_true:("actual_mae", mean_absolute_error(np.exp(y_true.get_label()), np.exp(y_predicted))),
            early_stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=True)

        # Perform the testing procedure
        Y_test = model.predict(X_test)

        # Save submission to disk
        if not os.path.isdir(SUBMISSION_FOLDER_PATH):
            os.makedirs(SUBMISSION_FOLDER_PATH)
        submission_file_content[LABEL_COLUMN_NAME] = np.exp(Y_test) - 200
        submission_file_content.to_csv(submission_file_path, index=False)

    # Perform ensembling
    ensemble_predictions()

    print("All done!")

if __name__ == "__main__":
    run()
