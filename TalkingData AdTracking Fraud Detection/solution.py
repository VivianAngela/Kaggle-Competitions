import os
import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split

# Dataset
PROJECT_NAME = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
PROJECT_FOLDER_PATH = os.path.join(os.path.expanduser("~"), "Documents/Dataset", PROJECT_NAME)
VANILLA_FOLDER_PATH = os.path.join(PROJECT_FOLDER_PATH, "vanilla")
TRAIN_FILE_PATH = os.path.join(VANILLA_FOLDER_PATH, "train.csv")
TEST_FILE_PATH = os.path.join(VANILLA_FOLDER_PATH, "test.csv")
SAMPLE_NUM = 100000

# Submission
TEAM_NAME = "Aurora"
SUBMISSION_FOLDER_PATH = os.path.join(PROJECT_FOLDER_PATH, "submission")
os.makedirs(SUBMISSION_FOLDER_PATH, exist_ok=True)

# Training and Testing procedure
NUM_BOOST_ROUND = 1000000
EARLY_STOPPING_ROUNDS = 100

def load_data(nrows=SAMPLE_NUM):
    parse_dates = ["click_time"]
    dtype = {"ip": "uint32", "app": "uint16", "device": "uint16", "os": "uint16", "channel": "uint16", "is_attributed": "bool", "click_id": "uint32"}

    print("Loading training data ...")
    train_df = pd.read_csv(TRAIN_FILE_PATH, parse_dates=parse_dates, dtype=dtype, nrows=nrows)
    train_df.drop("attributed_time", axis=1, inplace=True)
    train_num = len(train_df)

    print("Loading testing data ...")
    test_df = pd.read_csv(TEST_FILE_PATH, parse_dates=parse_dates, dtype=dtype, nrows=nrows)
    submission_df = pd.DataFrame(test_df["click_id"])
    test_df.drop("click_id", axis=1, inplace=True)

    print("Merging training data and testing data ...")
    merged_df = pd.concat([train_df, test_df], copy=False)

    print("Extracting date time info ...")
    merged_df["month"] = merged_df["click_time"].dt.month.astype("uint8")
    merged_df["day"] = merged_df["click_time"].dt.day.astype("uint8")
    merged_df["hour"] = merged_df["click_time"].dt.hour.astype("uint8")
    merged_df.drop("click_time", axis=1, inplace=True)

    print("Splitting data ...")
    train_indexes, valid_indexes = train_test_split(np.arange(train_num), test_size=0.1, random_state=0)
    train_df, valid_df = merged_df.iloc[train_indexes], merged_df.iloc[valid_indexes]
    test_df = merged_df.iloc[train_num:].drop("is_attributed", axis=1)

    return train_df, valid_df, test_df, submission_df

def run():
    print("Loading data ...")
    train_df, valid_df, test_df, submission_df = load_data()

    print("Generating LightGBM datasets ...")
    target_name = "is_attributed"
    categorical_feature = ["app", "channel", "device", "os", "month", "day", "hour"]
    train_dataset = lgb.Dataset(train_df.drop(target_name, axis=1), train_df[target_name], categorical_feature=categorical_feature)
    valid_dataset = lgb.Dataset(valid_df.drop(target_name, axis=1), valid_df[target_name], categorical_feature=categorical_feature, reference=train_dataset)

    print("Performing the training procedure ...")
    best_params = {"subsample": 0.9, "colsample_bytree": 0.9, "objective": "binary", "metric": "auc"}  # Use empirical parameters
    model = lgb.train(params=best_params, train_set=train_dataset, valid_sets=[valid_dataset],
                    num_boost_round=NUM_BOOST_ROUND, early_stopping_rounds=EARLY_STOPPING_ROUNDS)

    print("Performing the testing procedure ...")
    prediction_array = model.predict(test_df, num_iteration=model.best_iteration)
    submission_df["is_attributed"] = prediction_array
    submission_file_path = os.path.join(SUBMISSION_FOLDER_PATH, "{} {}.csv".format(TEAM_NAME, str(datetime.datetime.now()).split(".")[0]))
    print("Saving submission to {} ...".format(submission_file_path))
    submission_df.to_csv(submission_file_path, index=False)

    print("All done!")

if __name__ == "__main__":
    run()
