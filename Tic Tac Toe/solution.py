import os
import glob
import numpy as np
import pandas as pd
import lightgbm as lgb
from itertools import product
from sklearn.model_selection import StratifiedKFold

# Dataset
PROJECT_NAME = "Tic Tac Toe"
PROJECT_FOLDER_PATH = os.path.join(os.path.expanduser("~"), "Documents/Dataset", PROJECT_NAME)
TRAIN_FEATURE_FILE_PATH = os.path.join(PROJECT_FOLDER_PATH, "train.csv")
TRAIN_LABEL_FILE_PATH = os.path.join(PROJECT_FOLDER_PATH, "y_train.csv")
TEST_FEATURE_FILE_PATH = os.path.join(PROJECT_FOLDER_PATH, "test.csv")

# Output
OUTPUT_FOLDER_PATH = os.path.join(PROJECT_FOLDER_PATH, "{}_output".format(os.path.basename(__file__).split(".")[0]))
SUBMISSION_FOLDER_PATH = os.path.join(OUTPUT_FOLDER_PATH, "Submission")

# Training and Testing procedure
SPLIT_NUM = 10
RANDOM_STATE = None
NUM_BOOST_ROUND = 200
EARLY_STOPPING_ROUNDS = 50

def load_dataset():
    print("Loading feature files ...")
    feature_array_list = []
    for feature_file_path in [TRAIN_FEATURE_FILE_PATH, TEST_FEATURE_FILE_PATH]:
        feature_file_content = pd.read_csv(feature_file_path)
        feature_array = feature_file_content.as_matrix(["TLS", "TMS", "TRS", "MLS", "MMS", "MRS", "BLS", "BMS", "BRS"])
        feature_array[feature_array == "x"] = 1
        feature_array[feature_array == "o"] = -1
        feature_array[feature_array == "b"] = 0
        feature_array_list.append(feature_array)
    train_feature_array, test_feature_array = feature_array_list

    print("Loading label file ...")
    train_label_file_content = pd.read_csv(TRAIN_LABEL_FILE_PATH)
    train_label_array = train_label_file_content.as_matrix(["Category"])
    train_label_array[train_label_array == "positive"] = 1
    train_label_array[train_label_array == "negative"] = 0
    train_label_array = train_label_array.flatten()

    return train_feature_array, train_label_array, test_feature_array

def get_augmented_data(feature_array, label_array):
    feature_array_list = []
    label_array_list = []

    vanilla_index_matrix = np.arange(9).reshape(3, 3)
    for rotate_time, flip_func in product(np.arange(4), [None, np.fliplr, np.flipud]):
        index_matrix = np.copy(vanilla_index_matrix)
        index_matrix = np.rot90(index_matrix, k=rotate_time)
        if flip_func is not None:
            index_matrix = flip_func(index_matrix)
        feature_array_list.append(feature_array[:, index_matrix.flatten()])
        label_array_list.append(label_array)

    return np.array(feature_array_list).reshape(-1, feature_array.shape[-1]), np.array(label_array_list).flatten()

def ensemble_predictions(submission_folder_path, proba_column_name):
    # Read predictions
    submission_file_path_list = glob.glob(os.path.join(submission_folder_path, "submission_*.csv"))
    submission_file_content_list = [pd.read_csv(submission_file_path) for submission_file_path in submission_file_path_list]
    ensemble_submission_file_content = submission_file_content_list[0]
    print("There are {} submissions in total.".format(len(submission_file_path_list)))

    # Concatenate predictions
    proba_array = np.array([submission_file_content[proba_column_name].as_matrix() for submission_file_content in submission_file_content_list])

    # Ensemble predictions
    for ensemble_func, ensemble_submission_file_name in zip([np.max, np.min, np.mean, np.median], ["max.csv", "min.csv", "mean.csv", "median.csv"]):
        ensemble_submission_file_path = os.path.join(submission_folder_path, os.pardir, ensemble_submission_file_name)
        ensemble_submission_file_content[proba_column_name] = ensemble_func(proba_array, axis=0)
        ensemble_submission_file_content["Category"] = "negative"
        ensemble_submission_file_content.ix[ensemble_submission_file_content[proba_column_name] > 0.5, "Category"] = "positive"
        ensemble_submission_file_content[["ID", "Category"]].to_csv(ensemble_submission_file_path, index=False)

def run():
    print("Creating folders ...")
    os.makedirs(SUBMISSION_FOLDER_PATH, exist_ok=True)

    print("Loading dataset ...")
    train_feature_array, train_label_array, test_feature_array = load_dataset()

    cv_object = StratifiedKFold(n_splits=SPLIT_NUM, random_state=RANDOM_STATE)
    best_params = {"subsample": 0.9, "colsample_bytree": 0.9, "objective": "binary", "metric": "binary_logloss"}  # Use empirical parameters
    for split_index, (train_index_array, valid_index_array) in enumerate(cv_object.split(np.zeros((len(train_label_array), 1)), train_label_array), start=1):
        print("Working on splitting fold {} ...".format(split_index))

        submission_file_path = os.path.join(SUBMISSION_FOLDER_PATH, "submission_{}.csv".format(split_index))
        if os.path.isfile(submission_file_path):
            print("The submission file already exists.")
            continue

        print("Dividing the vanilla training dataset to actual training/validation dataset ...")
        actual_train_feature_array, actual_train_label_array = train_feature_array[train_index_array], train_label_array[train_index_array]
        actual_valid_feature_array, actual_valid_label_array = train_feature_array[valid_index_array], train_label_array[valid_index_array]

        print("Performing data augmentation ...")
        actual_train_feature_array, actual_train_label_array = get_augmented_data(actual_train_feature_array, actual_train_label_array)
        actual_valid_feature_array, actual_valid_label_array = get_augmented_data(actual_valid_feature_array, actual_valid_label_array)
        actual_train_data = lgb.Dataset(actual_train_feature_array, label=actual_train_label_array)
        actual_valid_data = lgb.Dataset(actual_valid_feature_array, label=actual_valid_label_array, reference=actual_train_data)

        print("Performing the training procedure ...")
        model = lgb.train(params=best_params, train_set=actual_train_data, valid_sets=[actual_valid_data],
                        num_boost_round=NUM_BOOST_ROUND, early_stopping_rounds=EARLY_STOPPING_ROUNDS)

        print("Performing the testing procedure ...")
        prediction_array = model.predict(test_feature_array, num_iteration=model.best_iteration)
        submission_file_content = pd.DataFrame({"ID": np.arange(len(prediction_array)), "Probability": prediction_array})
        submission_file_content.to_csv(submission_file_path, index=False)

    print("Performing ensembling ...")
    ensemble_predictions(submission_folder_path=SUBMISSION_FOLDER_PATH, proba_column_name="Probability")

    print("All done!")

if __name__ == "__main__":
    run()
