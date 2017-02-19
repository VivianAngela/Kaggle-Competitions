import os
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

# Dataset
DATASET_FOLDER_PATH = os.path.join(os.path.expanduser("~"), "Documents/Dataset/Gene Expression Prediction")
MODEL_FOLDER_PATH = os.path.join(DATASET_FOLDER_PATH, "model")
SUBMISSION_FOLDER_PATH = os.path.join(DATASET_FOLDER_PATH, "submission")

# The training/testing process
PARAMS = {"objective":"binary", "metric":"auc"}
NUM_BOOST_ROUND = 1000000
EARLY_STOPPING_ROUNDS = 100

def load_dataset():
    # Read csv files
    x_train = pd.read_csv(os.path.join(DATASET_FOLDER_PATH, "train/x_train.csv")).as_matrix()
    x_test = pd.read_csv(os.path.join(DATASET_FOLDER_PATH, "test/x_test.csv")).as_matrix()
    y_train = pd.read_csv(os.path.join(DATASET_FOLDER_PATH, "train/y_train.csv")).as_matrix()

    # Remove the first column
    x_train = x_train[:, 1:]
    x_test = x_test[:, 1:]
    y_train = y_train[:, 1:]

    # Every 100 rows correspond to one gene. Extract all 100-row-blocks into a list using np.split.
    num_genes_train = x_train.shape[0] / 100
    num_genes_test = x_test.shape[0] / 100
    x_train = np.split(x_train, num_genes_train)
    x_test = np.split(x_test, num_genes_test)

    # Reshape by raveling each 100x5 array into a 500-length vector
    x_train = [g.ravel() for g in x_train]
    x_test = [g.ravel() for g in x_test]

    # Convert data from list to array
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_train = np.ravel(y_train)

    return x_train, y_train, x_test

def run():
    print("Creating folder ...")
    os.makedirs(MODEL_FOLDER_PATH, exist_ok=True)
    os.makedirs(SUBMISSION_FOLDER_PATH, exist_ok=True)

    print("Loading dataset ...")
    x_train, y_train, x_test = load_dataset()

    prediction_array_list = []
    cv_object = StratifiedShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    for cv_index, (train_index, valid_index) in enumerate(cv_object.split(x_train, y_train), start=1):
        submission_file_path = os.path.join(SUBMISSION_FOLDER_PATH, "submission_{}.csv".format(cv_index))
        if not os.path.isfile(submission_file_path):
            model_file_path = os.path.join(MODEL_FOLDER_PATH, "model_{}.txt".format(cv_index))
            if not os.path.isfile(model_file_path):
                train_data = lgb.Dataset(x_train[train_index], label=y_train[train_index])
                validation_data = lgb.Dataset(x_train[valid_index], label=y_train[valid_index], reference=train_data)
                model = lgb.train(params=PARAMS, train_set=train_data, num_boost_round=NUM_BOOST_ROUND, valid_sets=[validation_data], early_stopping_rounds=EARLY_STOPPING_ROUNDS)
                model.save_model(model_file_path, num_iteration=model.best_iteration)

            assert os.path.isfile(model_file_path)
            print("Loading weights from {}".format(os.path.basename(model_file_path)))
            model = lgb.Booster(model_file=model_file_path)

            # Generate prediction
            prediction_array = np.expand_dims(model.predict(x_test), axis=-1)
            prediction_array_list.append(prediction_array)

            # Save prediction to disk
            submission_file_content = pd.DataFrame(np.hstack((np.expand_dims(np.arange(len(prediction_array)), axis=-1) + 1, prediction_array)), columns=["GeneId", "Prediction"])
            submission_file_content.GeneId = submission_file_content.GeneId.astype(int)
            submission_file_content.to_csv(submission_file_path, index=False)
            print("Submission saved at {}".format(submission_file_path))
        else:
            # Load prediction
            prediction_array = np.expand_dims(pd.read_csv(submission_file_path).as_matrix()[:, 1], axis=-1)
            prediction_array_list.append(prediction_array)

    # Ensemble predictions
    for ensemble_func, ensemble_func_name in zip([np.max, np.min, np.mean, np.median], ["max", "min", "mean", "median"]):
        submission_file_path = os.path.join(SUBMISSION_FOLDER_PATH, "submission_{}.csv".format(ensemble_func_name))
        if not os.path.isfile(submission_file_path):
            prediction_array = ensemble_func(prediction_array_list, axis=0)

            # Save prediction to disk
            submission_file_content = pd.DataFrame(np.hstack((np.expand_dims(np.arange(len(prediction_array)), axis=-1) + 1, prediction_array)), columns=["GeneId", "Prediction"])
            submission_file_content.GeneId = submission_file_content.GeneId.astype(int)
            submission_file_content.to_csv(submission_file_path, index=False)
            print("Submission saved at {}".format(submission_file_path))

    print("All done!")

if __name__ == "__main__":
    run()
