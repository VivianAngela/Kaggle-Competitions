import os
import time
import pandas as pd

# Dataset
PROJECT_NAME = "Text Normalization Challenge"
PROJECT_FOLDER_PATH = os.path.join(os.path.expanduser("~"), "Documents/Dataset", PROJECT_NAME)
VANILLA_DATASET_FOLDER_PATH = os.path.join(PROJECT_FOLDER_PATH, "vanilla")
TRAIN_FILE_PATH = os.path.join(VANILLA_DATASET_FOLDER_PATH, "en_train.csv")
TEST_FILE_PATH = os.path.join(VANILLA_DATASET_FOLDER_PATH, "en_test.csv")

# Output
SUBMISSION_FOLDER_PATH = os.path.join(PROJECT_FOLDER_PATH, "submission")
SUBMISSION_FILE_PATH = os.path.join(SUBMISSION_FOLDER_PATH, "{}.csv".format(time.strftime("%c")))

def run():
    print("Creating folders ...")
    os.makedirs(SUBMISSION_FOLDER_PATH, exist_ok=True)

    print("Loading {} ...".format(TRAIN_FILE_PATH))
    before_to_after_dict = {}
    before_to_count_dict = {}
    file_content = pd.read_csv(TRAIN_FILE_PATH, encoding="utf-8")[["before", "after"]]
    grouped_file_content = file_content.groupby(["before", "after"])
    count_array = grouped_file_content.apply(lambda group: len(group)).as_matrix()  # pylint: disable=unnecessary-lambda
    for (before, after), count in zip(grouped_file_content.groups.keys(), count_array):
        if before in before_to_count_dict and before_to_count_dict[before] >= count:
            continue
        before_to_after_dict[before] = after
        before_to_count_dict[before] = count

    print("Loading {} ...".format(TEST_FILE_PATH))
    file_content = pd.read_csv(TEST_FILE_PATH, encoding="utf-8")
    file_content["id"] = file_content.apply(lambda row: "{}_{}".format(row["sentence_id"], row["token_id"]), axis=1)
    file_content["after"] = file_content.apply(lambda row: before_to_after_dict.get(row["before"], row["before"]), axis=1)
    file_content[["id", "after"]].to_csv(SUBMISSION_FILE_PATH, index=False, encoding="utf-8")

    print("All done!")

if __name__ == "__main__":
    run()
