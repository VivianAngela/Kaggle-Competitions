from __future__ import absolute_import, division, print_function

import os
import numpy as np
import pandas as pd
from collections import Counter
from joblib import Parallel, delayed
from nltk.corpus import stopwords

# Dataset
PROJECT_NAME = "Quora Question Pairs"
PROJECT_FOLDER_PATH = os.path.join(os.path.expanduser("~"), "Documents/Dataset", PROJECT_NAME)
TRAIN_FILE_PATH = os.path.join(PROJECT_FOLDER_PATH, "train.csv")
TEST_FILE_PATH = os.path.join(PROJECT_FOLDER_PATH, "test.csv")
DATASET_FILE_PATH = os.path.join(PROJECT_FOLDER_PATH, "dataset.csv")

def get_word_to_weight_dict(question_list):
    def get_weight(count, eps=10000, min_count=2):
        """
            If a word appears only once, we ignore it completely (likely a typo)
            Epsilon defines a smoothing constant, which makes the effect of extremely rare words smaller
        """
        if count < min_count:
            return 0
        else:
            return 1 / (count + eps)
    word_list = (" ".join(pd.Series(question_list).astype(str))).lower().split()
    counter_object = Counter(word_list)
    word_to_weight_dict = {word: get_weight(count) for word, count in counter_object.items()}
    return word_to_weight_dict

print("Loading text files ...")
TRAIN_FILE_CONTENT = pd.read_csv(TRAIN_FILE_PATH, encoding="utf-8")
TEST_FILE_CONTENT = pd.read_csv(TEST_FILE_PATH, encoding="utf-8")

print("Getting handmade features ...")
STOPWORD_SET = set(stopwords.words("english"))
WORD_TO_WEIGHT_DICT = get_word_to_weight_dict(TRAIN_FILE_CONTENT["question1"].tolist() + TRAIN_FILE_CONTENT["question2"].tolist())

def get_handmade_feature(question1, question2, is_duplicate):
    # Convert to string
    question1 = str(question1)
    question2 = str(question2)

    # Initiate a dictionary
    entry = {}
    entry["question1"] = question1
    entry["question2"] = question2
    entry["is_duplicate"] = is_duplicate

    # Calculate whether two questions are identical
    entry["question_identical"] = int(question1 == question2)

    # Calculate difference between lengths of questions
    entry["question1_len"] = len(question1)
    entry["question2_len"] = len(question2)
    entry["question_len_diff"] = abs(entry["question1_len"] - entry["question2_len"])

    # Calculate difference between lengths of questions without spaces
    entry["question1_char_num"] = len(question1.replace(" ", ""))
    entry["question2_char_num"] = len(question2.replace(" ", ""))
    entry["question_char_num_diff"] = abs(entry["question1_char_num"] - entry["question2_char_num"])

    # Calculate difference between num of words
    entry["question1_word_num"] = len(question1.split())
    entry["question2_word_num"] = len(question2.split())
    entry["question_word_num_diff"] = abs(entry["question1_word_num"] - entry["question2_word_num"])

    # Calculate average word length
    if entry["question1_word_num"] != 0 and entry["question2_word_num"] != 0:
        entry["question1_average_word_length"] = entry["question1_char_num"] / entry["question1_word_num"]
        entry["question2_average_word_length"] = entry["question2_char_num"] / entry["question2_word_num"]
        entry["question_average_word_length_diff"] = abs(entry["question1_average_word_length"] - entry["question2_average_word_length"])

    # Calculate whether each question has certain interrogative
    for interrogative in ["how", "what", "when", "where", "which", "who", "why"]:
        entry["question1_" + interrogative] = int(interrogative in question1.lower())
        entry["question2_" + interrogative] = int(interrogative in question2.lower())
        entry["question_" + interrogative] = int(entry["question1_" + interrogative] and entry["question2_" + interrogative])

    # Get unique word list/set
    question1_word_list = question1.lower().split()
    question1_word_set = set(question1_word_list)
    question2_word_list = question2.lower().split()
    question2_word_set = set(question2_word_list)

    # Calculate the ratio of same words at the same positions
    if max(len(question1_word_list), len(question2_word_list)) != 0:
        entry["same_word_ratio"] = sum([question1_word == question2_word for question1_word, question2_word in zip(question1_word_list, question2_word_list)]) / max(len(question1_word_list), len(question2_word_list))

    # Calculate the ratio of number of stopword
    question1_stopword_set = question1_word_set.intersection(STOPWORD_SET)
    question2_stopword_set = question2_word_set.intersection(STOPWORD_SET)
    question1_non_stopword_set = question1_word_set.difference(STOPWORD_SET)
    question2_non_stopword_set = question2_word_set.difference(STOPWORD_SET)
    if len(question1_stopword_set) + len(question1_non_stopword_set) != 0 and len(question2_stopword_set) + len(question2_non_stopword_set) != 0:
        entry["question1_stopword_ratio"] = len(question1_stopword_set) / (len(question1_stopword_set) + len(question1_non_stopword_set))
        entry["question2_stopword_ratio"] = len(question2_stopword_set) / (len(question2_stopword_set) + len(question2_non_stopword_set))
        entry["question_stopword_ratio_diff"] = abs(entry["question1_stopword_ratio"] - entry["question2_stopword_ratio"])

    # Calculate the neighbour word pairs
    question1_neighbour_word_set = set([word_tuple for word_tuple in zip(question1_word_list, question1_word_list[1:])])
    question2_neighbour_word_set = set([word_tuple for word_tuple in zip(question2_word_list, question2_word_list[1:])])
    if len(question1_neighbour_word_set) + len(question2_neighbour_word_set) != 0:
        common_neighbour_word_set = question1_neighbour_word_set.intersection(question2_neighbour_word_set)
        entry["neighbour_word_ratio"] = len(common_neighbour_word_set) / (len(question1_neighbour_word_set) + len(question2_neighbour_word_set))

    # Calculate features of word count/weight
    common_non_stopword_set = question1_non_stopword_set.intersection(question2_non_stopword_set)
    common_non_stopword_weight_list = [WORD_TO_WEIGHT_DICT.get(word, 0) for word in common_non_stopword_set]
    question1_non_stopword_weight_list = [WORD_TO_WEIGHT_DICT.get(word, 0) for word in question1_non_stopword_set]
    question2_non_stopword_weight_list = [WORD_TO_WEIGHT_DICT.get(word, 0) for word in question2_non_stopword_set]
    total_non_stopword_weight_list = question1_non_stopword_weight_list + question2_non_stopword_weight_list
    non_stopword_weight_ratio_denominator = np.sum(total_non_stopword_weight_list)
    non_stopword_num_ratio_denominator = len(question1_non_stopword_set) + len(question2_non_stopword_set) - len(common_non_stopword_set)
    cosine_denominator = np.sqrt(np.dot(question1_non_stopword_weight_list, question1_non_stopword_weight_list)) * np.sqrt(np.dot(question2_non_stopword_weight_list, question2_non_stopword_weight_list))
    entry["common_non_stopword_num"] = len(common_non_stopword_set)
    if non_stopword_weight_ratio_denominator != 0:
        entry["non_stopword_weight_ratio"] = np.sum(common_non_stopword_weight_list) / non_stopword_weight_ratio_denominator
        entry["sqrt_non_stopword_weight_ratio"] = np.sqrt(entry["non_stopword_weight_ratio"])
    if non_stopword_num_ratio_denominator != 0:
        entry["non_stopword_num_ratio"] = len(common_non_stopword_set) / non_stopword_num_ratio_denominator
    if cosine_denominator != 0:
        entry["cosine"] = np.dot(common_non_stopword_weight_list, common_non_stopword_weight_list) / cosine_denominator

    return entry

def get_magic_feature(file_content):
    def get_id_to_frequency_dict(pandas_series_list):
        id_list = []
        for pandas_series in pandas_series_list:
            id_list += pandas_series.tolist()

        id_value_array, id_count_array = np.unique(id_list, return_counts=True)
        id_frequency_array = id_count_array / np.max(id_count_array)
        return dict(zip(id_value_array, id_frequency_array))

    print("Getting one ID for each unique question ...")
    all_question_list = file_content["question1"].tolist() + file_content["question2"].tolist()
    all_unique_question_list = list(set(all_question_list))
    question_to_id_dict = pd.Series(range(len(all_unique_question_list)), index=all_unique_question_list).to_dict()

    print("Converting to question ID ...")
    file_content["qid1"] = file_content["question1"].map(question_to_id_dict)
    file_content["qid2"] = file_content["question2"].map(question_to_id_dict)

    print("Calculating frequencies ...")
    id_to_frequency_dict = get_id_to_frequency_dict([file_content["qid1"], file_content["qid2"]])
    file_content["question1_frequency"] = file_content["qid1"].map(lambda qid: id_to_frequency_dict.get(qid, 0))
    file_content["question2_frequency"] = file_content["qid2"].map(lambda qid: id_to_frequency_dict.get(qid, 0))
    file_content["question_frequency_diff"] = file_content["question1_frequency"] - file_content["question2_frequency"]

    return file_content

def load_dataset():
    if os.path.isfile(DATASET_FILE_PATH):
        print("Loading dataset from disk ...")
        merged_file_content = pd.read_csv(DATASET_FILE_PATH, encoding="utf-8")
    else:
        print("Merging train and test file content ...")
        merged_file_content = pd.concat([TRAIN_FILE_CONTENT, TEST_FILE_CONTENT])

        print("Getting handmade feature ...")
        merged_file_content = pd.DataFrame(Parallel(n_jobs=-2)(delayed(get_handmade_feature)(question1, question2, is_duplicate) for question1, question2, is_duplicate in merged_file_content[["question1", "question2", 'is_duplicate']].as_matrix()))

        print("Getting magic features ...")
        merged_file_content = get_magic_feature(merged_file_content)

        print("Removing irrelevant columns ...")
        merged_file_content.drop(["qid1", "qid2", "question1", "question2"], axis=1, inplace=True)
        merged_file_content.fillna(-1, axis=1, inplace=True)

        print("Saving dataset to disk ...")
        merged_file_content.to_csv(DATASET_FILE_PATH, index=False)

    return merged_file_content

def run():
    print("Loading dataset ...")
    merged_file_content = load_dataset()

    print("All done!")

if __name__ == "__main__":
    run()
