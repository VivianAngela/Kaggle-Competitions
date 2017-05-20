from __future__ import absolute_import, division, print_function

import os
import glob
import numpy as np
import pandas as pd
import lightgbm as lgb
from collections import Counter
from joblib import Parallel, delayed
from nltk.corpus import stopwords
from sklearn.model_selection import StratifiedKFold

# Dataset
PROJECT_NAME = "Quora Question Pairs"
PROJECT_FOLDER_PATH = os.path.join(os.path.expanduser("~"), "Documents/Dataset", PROJECT_NAME)
TRAIN_FILE_PATH = os.path.join(PROJECT_FOLDER_PATH, "train.csv")
TEST_FILE_PATH = os.path.join(PROJECT_FOLDER_PATH, "test.csv")
DATASET_FILE_PATH = os.path.join(PROJECT_FOLDER_PATH, "shallow_learning_dataset.npz")

# Output
OUTPUT_FOLDER_PATH = os.path.join(PROJECT_FOLDER_PATH, "{}_output".format(os.path.basename(__file__).split(".")[0]))
OPTIMAL_WEIGHTS_FOLDER_PATH = os.path.join(OUTPUT_FOLDER_PATH, "Optimal Weights")
SUBMISSION_FOLDER_PATH = os.path.join(OUTPUT_FOLDER_PATH, "Submission")

# Training and Testing procedure
SPLIT_NUM = 10
RANDOM_STATE = 666666
NUM_BOOST_ROUND = 1000000
EARLY_STOPPING_ROUNDS = 100
TARGET_MEAN_PREDICTION = 0.175  # https://www.kaggle.com/davidthaler/how-many-1-s-are-in-the-public-lb

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

def get_magic_feature_v1(file_content):
    # https://www.kaggle.com/jturkewitz/magic-features-0-03-gain
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
    file_content["question_frequency_diff"] = abs(file_content["question1_frequency"] - file_content["question2_frequency"])

    return file_content

def load_dataset():
    if os.path.isfile(DATASET_FILE_PATH):
        print("Loading dataset from disk ...")
        dataset_file_content = np.load(DATASET_FILE_PATH)
        train_question1_feature_array = dataset_file_content["train_question1_feature_array"]
        train_question2_feature_array = dataset_file_content["train_question2_feature_array"]
        train_common_feature_array = dataset_file_content["train_common_feature_array"]
        train_label_array = dataset_file_content["train_label_array"]
        test_question1_feature_array = dataset_file_content["test_question1_feature_array"]
        test_question2_feature_array = dataset_file_content["test_question2_feature_array"]
        test_common_feature_array = dataset_file_content["test_common_feature_array"]
    else:
        print("Merging train and test file content ...")
        merged_file_content = pd.concat([TRAIN_FILE_CONTENT, TEST_FILE_CONTENT])

        print("Getting handmade feature ...")
        merged_file_content = pd.DataFrame(Parallel(n_jobs=-2)(delayed(get_handmade_feature)(question1, question2, is_duplicate) for question1, question2, is_duplicate in merged_file_content[["question1", "question2", "is_duplicate"]].as_matrix()))

        print("Getting magic features ...")
        merged_file_content = get_magic_feature_v1(merged_file_content)

        print("Removing irrelevant columns ...")
        merged_file_content.drop(["qid1", "qid2", "question1", "question2"], axis=1, inplace=True)
        merged_file_content.fillna(-1, axis=1, inplace=True)

        print("Separating feature columns ...")
        column_name_list = list(merged_file_content)
        question1_feature_column_name_list = sorted([column_name for column_name in column_name_list if column_name.startswith("question1_")])
        question2_feature_column_name_list = sorted([column_name for column_name in column_name_list if column_name.startswith("question2_")])
        common_feature_column_name_list = sorted(set(column_name_list) - set(question1_feature_column_name_list + question2_feature_column_name_list + ["is_duplicate"]))
        is_train_mask_array = merged_file_content["is_duplicate"] != -1
        train_question1_feature_array = merged_file_content[is_train_mask_array][question1_feature_column_name_list].as_matrix().astype(np.float32)
        train_question2_feature_array = merged_file_content[is_train_mask_array][question2_feature_column_name_list].as_matrix().astype(np.float32)
        train_common_feature_array = merged_file_content[is_train_mask_array][common_feature_column_name_list].as_matrix().astype(np.float32)
        train_label_array = merged_file_content[is_train_mask_array]["is_duplicate"].as_matrix().astype(np.bool)
        test_question1_feature_array = merged_file_content[np.logical_not(is_train_mask_array)][question1_feature_column_name_list].as_matrix().astype(np.float32)
        test_question2_feature_array = merged_file_content[np.logical_not(is_train_mask_array)][question2_feature_column_name_list].as_matrix().astype(np.float32)
        test_common_feature_array = merged_file_content[np.logical_not(is_train_mask_array)][common_feature_column_name_list].as_matrix().astype(np.float32)

        print("Saving dataset to disk ...")
        np.savez_compressed(DATASET_FILE_PATH,
                            train_question1_feature_array=train_question1_feature_array, train_question2_feature_array=train_question2_feature_array,
                            train_common_feature_array=train_common_feature_array, train_label_array=train_label_array,
                            test_question1_feature_array=test_question1_feature_array, test_question2_feature_array=test_question2_feature_array,
                            test_common_feature_array=test_common_feature_array)

    return train_question1_feature_array, train_question2_feature_array, train_common_feature_array, train_label_array, \
        test_question1_feature_array, test_question2_feature_array, test_common_feature_array

def get_augmented_data(question1_feature_array, question2_feature_array, common_feature_array, label_array=None):
    augmented_feature_array = np.vstack((np.hstack((question1_feature_array, question2_feature_array, common_feature_array)), np.hstack((question2_feature_array, question1_feature_array, common_feature_array))))
    if label_array is not None:
        augmented_label_array = np.hstack((label_array, label_array))
        return augmented_feature_array, augmented_label_array
    else:
        return augmented_feature_array

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
        ensemble_submission_file_content.to_csv(ensemble_submission_file_path, index=False)

def run():
    print("Creating folders ...")
    os.makedirs(OPTIMAL_WEIGHTS_FOLDER_PATH, exist_ok=True)
    os.makedirs(SUBMISSION_FOLDER_PATH, exist_ok=True)

    print("Loading dataset ...")
    train_question1_feature_array, train_question2_feature_array, train_common_feature_array, train_label_array, \
        test_question1_feature_array, test_question2_feature_array, test_common_feature_array = load_dataset()
    test_feature_array = get_augmented_data(test_question1_feature_array, test_question2_feature_array, test_common_feature_array)

    cv_object = StratifiedKFold(n_splits=SPLIT_NUM, random_state=RANDOM_STATE)
    best_params = {"subsample": 0.9, "colsample_bytree": 0.9, "objective": "binary", "metric": "binary_logloss"}  # Use empirical parameters
    for split_index, (train_index_array, valid_index_array) in enumerate(cv_object.split(np.zeros((len(train_label_array), 1)), train_label_array), start=1):
        print("Working on splitting fold {} ...".format(split_index))

        submission_file_path = os.path.join(SUBMISSION_FOLDER_PATH, "submission_{}.csv".format(split_index))
        if os.path.isfile(submission_file_path):
            print("The submission file already exists.")
            continue

        optimal_weights_file_path = os.path.join(OPTIMAL_WEIGHTS_FOLDER_PATH, "optimal_weights_{}.txt".format(split_index))
        if os.path.isfile(optimal_weights_file_path):
            print("The optimal weights file already exists.")
        else:
            print("Dividing the vanilla training dataset to actual training/validation dataset ...")
            actual_train_question1_feature_array, actual_train_question2_feature_array, \
            actual_train_common_feature_array, actual_train_label_array = \
            train_question1_feature_array[train_index_array], train_question2_feature_array[train_index_array], \
            train_common_feature_array[train_index_array], train_label_array[train_index_array]
            actual_valid_question1_feature_array, actual_valid_question2_feature_array, \
            actual_valid_common_feature_array, actual_valid_label_array = \
            train_question1_feature_array[valid_index_array], train_question2_feature_array[valid_index_array], \
            train_common_feature_array[valid_index_array], train_label_array[valid_index_array]

            print("Calculating class weight ...")
            train_mean_prediction = np.mean(actual_train_label_array)
            train_class_weight = {0: (1 - TARGET_MEAN_PREDICTION) / (1 - train_mean_prediction), 1: TARGET_MEAN_PREDICTION / train_mean_prediction}
            valid_mean_prediction = np.mean(actual_valid_label_array)
            valid_class_weight = {0: (1 - TARGET_MEAN_PREDICTION) / (1 - valid_mean_prediction), 1: TARGET_MEAN_PREDICTION / valid_mean_prediction}

            print("Performing data augmentation ...")
            actual_train_feature_array, actual_train_label_array = get_augmented_data(actual_train_question1_feature_array, actual_train_question2_feature_array, actual_train_common_feature_array, actual_train_label_array)
            actual_train_weight_list = [train_class_weight[label] for label in actual_train_label_array]
            actual_train_data = lgb.Dataset(actual_train_feature_array, label=actual_train_label_array, weight=actual_train_weight_list)
            actual_valid_feature_array, actual_valid_label_array = get_augmented_data(actual_valid_question1_feature_array, actual_valid_question2_feature_array, actual_valid_common_feature_array, actual_valid_label_array)
            actual_valid_weight_list = [valid_class_weight[label] for label in actual_valid_label_array]
            actual_valid_data = lgb.Dataset(actual_valid_feature_array, label=actual_valid_label_array, weight=actual_valid_weight_list, reference=actual_train_data)

            print("Performing the training procedure ...")
            model = lgb.train(params=best_params, train_set=actual_train_data, valid_sets=[actual_valid_data], num_boost_round=NUM_BOOST_ROUND, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
            model.save_model(optimal_weights_file_path, num_iteration=model.best_iteration)

        assert os.path.isfile(optimal_weights_file_path)
        model = lgb.Booster(model_file=optimal_weights_file_path)

        print("Performing the testing procedure ...")
        prediction_array = model.predict(test_feature_array)
        prediction_array = np.mean(np.reshape(prediction_array, (-1, 2), order="F"), axis=1)
        submission_file_content = pd.DataFrame({"test_id":np.arange(len(prediction_array)), "is_duplicate":np.squeeze(prediction_array)})
        submission_file_content.to_csv(submission_file_path, index=False)

    print("Performing ensembling ...")
    ensemble_predictions(submission_folder_path=SUBMISSION_FOLDER_PATH, proba_column_name="is_duplicate")

    print("All done!")

if __name__ == "__main__":
    run()
