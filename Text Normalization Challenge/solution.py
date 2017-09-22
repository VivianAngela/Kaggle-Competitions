import os
import csv
import glob
import time
import operator
import pandas as pd

# Dataset
PROJECT_NAME = "Text Normalization Challenge"
PROJECT_FOLDER_PATH = os.path.join(os.path.expanduser("~"), "Documents/Dataset", PROJECT_NAME)
VANILLA_DATASET_FOLDER_PATH = os.path.join(PROJECT_FOLDER_PATH, "vanilla")
TRAIN_FILE_PATH = os.path.join(VANILLA_DATASET_FOLDER_PATH, "en_train.csv")
TEST_FILE_PATH = os.path.join(VANILLA_DATASET_FOLDER_PATH, "en_test.csv")
LOOKUP_FILE_PATH = os.path.join(VANILLA_DATASET_FOLDER_PATH, "lookup.csv")
ADDITIONAL_DATASET_FOLDER_PATH = os.path.join(PROJECT_FOLDER_PATH, "additional")
ADDITIONAL_FILE_PATH_LIST = glob.glob(os.path.join(ADDITIONAL_DATASET_FOLDER_PATH, "output-*-of-*"))

# Output
SUBMISSION_FOLDER_PATH = os.path.join(PROJECT_FOLDER_PATH, "submission")

def clean_text_file(old_file_path):
    new_file_path = os.path.join(os.path.abspath(os.path.join(old_file_path, "..")), "cleaned_" + os.path.basename(old_file_path))
    if not os.path.isfile(new_file_path):
        command = "grep -v ^\"<eos>\" \"{}\" > \"{}\"".format(old_file_path, new_file_path)
        os.system(command)
        assert os.path.isfile(new_file_path)
    return new_file_path

def load_text_file(file_path, sep=",", header="infer", usecols=None, quoting=0, chunksize=1e4, encoding="utf-8"):
    file_content = pd.read_csv(file_path, sep=sep, header=header, usecols=usecols, quoting=quoting, chunksize=chunksize, encoding=encoding)
    for chunk in file_content:
        for data in chunk.itertuples(index=False):
            yield data

def run():
    print("Creating folders ...")
    os.makedirs(SUBMISSION_FOLDER_PATH, exist_ok=True)

    if os.path.isfile(LOOKUP_FILE_PATH):
        print("Loading lookup file {} ...".format(LOOKUP_FILE_PATH))
        lookup_dict = {before: after for before, after in load_text_file(LOOKUP_FILE_PATH, sep="\t")}
    else:
        print("Loading {} ...".format(TRAIN_FILE_PATH))
        summary_dict = {}
        for before, after in load_text_file(TRAIN_FILE_PATH, usecols=["before", "after"]):
            if before not in summary_dict:
                summary_dict[before] = {}
            if after not in summary_dict[before]:
                summary_dict[before][after] = 0
            summary_dict[before][after] += 1
        vanilla_summary_dict = summary_dict.copy()

        for file_path in ADDITIONAL_FILE_PATH_LIST:
            print("Loading {} ...".format(file_path))
            for before, after in load_text_file(clean_text_file(file_path), sep="\t", header=None, usecols=[1, 2], quoting=csv.QUOTE_NONE):
                if after == "<self>" or after == "sil":
                    after = before
                if before not in summary_dict:
                    summary_dict[before] = {}
                if after not in summary_dict[before]:
                    summary_dict[before][after] = 0
                summary_dict[before][after] += 1
        summary_dict.update(vanilla_summary_dict)

        print("Generating lookup dict ...")
        lookup_dict = {}
        for before, after_dict in summary_dict.items():
            after = max(after_dict.items(), key=operator.itemgetter(1))[0]
            lookup_dict[before] = after

        print("Saving lookup file {} ...".format(LOOKUP_FILE_PATH))
        lookup_file_content = pd.DataFrame(list(lookup_dict.items()), columns=["before", "after"])
        lookup_file_content.to_csv(LOOKUP_FILE_PATH, sep="\t", index=False, encoding="utf-8")

    print("Loading {} ...".format(TEST_FILE_PATH))
    entry_list = []
    for sentence_id, token_id, before in load_text_file(TEST_FILE_PATH, usecols=["sentence_id", "token_id", "before"]):
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
