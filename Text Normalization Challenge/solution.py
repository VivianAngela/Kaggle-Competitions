import os
import time
import operator
import pandas as pd

# Dataset
PROJECT_NAME = "Text Normalization Challenge"
PROJECT_FOLDER_PATH = os.path.join(os.path.expanduser("~"), "Documents/Dataset", PROJECT_NAME)
VANILLA_DATASET_FOLDER_PATH = os.path.join(PROJECT_FOLDER_PATH, "vanilla")
TRAIN_FILE_PATH = os.path.join(VANILLA_DATASET_FOLDER_PATH, "en_train.csv")
TEST_FILE_PATH = os.path.join(VANILLA_DATASET_FOLDER_PATH, "en_test.csv")

# Output
SUBMISSION_FOLDER_PATH = os.path.join(PROJECT_FOLDER_PATH, "submission")

def run():
    print("Creating folders ...")
    os.makedirs(SUBMISSION_FOLDER_PATH, exist_ok=True)

    print("Loading {} ...".format(TRAIN_FILE_PATH))
    summary_dict = {}
    file_content = pd.read_csv(TRAIN_FILE_PATH, usecols=["before", "after"], encoding="utf-8")
    for before, after in file_content.itertuples(index=False):
        if before not in summary_dict:
            summary_dict[before] = {}
        if after not in summary_dict[before]:
            summary_dict[before][after] = 0
        summary_dict[before][after] += 1

    print("Generating lookup dict ...")
    lookup_dict = {}
    for before in summary_dict.keys():
        after = max(summary_dict[before].items(), key=operator.itemgetter(1))[0]
        if before != after:
            lookup_dict[before] = after

    print("Loading {} ...".format(TEST_FILE_PATH))
    entry_list = []
    file_content = pd.read_csv(TEST_FILE_PATH, usecols=["sentence_id", "token_id", "before"], encoding="utf-8")
    for sentence_id, token_id, before in file_content.itertuples(index=False):
        merged_id = "{}_{}".format(sentence_id, token_id)
        after = lookup_dict.get(before, before)
        entry_list.append((merged_id, after))

    print("Writing submission ...")
    submission_file_content = pd.DataFrame(entry_list, columns=["id", "after"])
    submission_file_path = os.path.join(SUBMISSION_FOLDER_PATH, "{}.csv".format(time.strftime("%c")))
    submission_file_content.to_csv(submission_file_path, index=False, encoding="utf-8")

    print("All done!")

if __name__ == "__main__":
    run()
