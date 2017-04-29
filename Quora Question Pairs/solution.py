from __future__ import absolute_import, division, print_function

import os
import re
import pandas as pd
from string import punctuation

PROJECT_NAME = "Quora Question Pairs"
PROJECT_FOLDER_PATH = os.path.join(os.path.expanduser("~"), "Documents/Dataset", PROJECT_NAME)
TRAIN_FILE_PATH = os.path.join(PROJECT_FOLDER_PATH, "train.csv")
TEST_FILE_PATH = os.path.join(PROJECT_FOLDER_PATH, "test.csv")

def clean_sentence(original_sentence, result_when_failure="qdkwzo"):
    """
        https://www.kaggle.com/currie32/quora-question-pairs/the-importance-of-cleaning-text
    """
    try:
        # Convert to lower case
        cleaned_sentence = " ".join(original_sentence.lower().split())

        # Replace elements
        cleaned_sentence = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", cleaned_sentence)
        cleaned_sentence = re.sub(r"what's", "what is ", cleaned_sentence)
        cleaned_sentence = re.sub(r"\'s", " ", cleaned_sentence)
        cleaned_sentence = re.sub(r"\'ve", " have ", cleaned_sentence)
        cleaned_sentence = re.sub(r"can't", "cannot ", cleaned_sentence)
        cleaned_sentence = re.sub(r"n't", " not ", cleaned_sentence)
        cleaned_sentence = re.sub(r"i'm", "i am ", cleaned_sentence)
        cleaned_sentence = re.sub(r"\'re", " are ", cleaned_sentence)
        cleaned_sentence = re.sub(r"\'d", " would ", cleaned_sentence)
        cleaned_sentence = re.sub(r"\'ll", " will ", cleaned_sentence)
        cleaned_sentence = re.sub(r",", " ", cleaned_sentence)
        cleaned_sentence = re.sub(r"\.", " ", cleaned_sentence)
        cleaned_sentence = re.sub(r"!", " ! ", cleaned_sentence)
        cleaned_sentence = re.sub(r"\/", " ", cleaned_sentence)
        cleaned_sentence = re.sub(r"\^", " ^ ", cleaned_sentence)
        cleaned_sentence = re.sub(r"\+", " + ", cleaned_sentence)
        cleaned_sentence = re.sub(r"\-", " - ", cleaned_sentence)
        cleaned_sentence = re.sub(r"\=", " = ", cleaned_sentence)
        cleaned_sentence = re.sub(r"'", " ", cleaned_sentence)
        cleaned_sentence = re.sub(r"(\d+)(k)", r"\g<1>000", cleaned_sentence)
        cleaned_sentence = re.sub(r":", " : ", cleaned_sentence)
        cleaned_sentence = re.sub(r" e g ", " eg ", cleaned_sentence)
        cleaned_sentence = re.sub(r" b g ", " bg ", cleaned_sentence)
        cleaned_sentence = re.sub(r" u s ", " american ", cleaned_sentence)
        cleaned_sentence = re.sub(r"\0s", "0", cleaned_sentence)
        cleaned_sentence = re.sub(r" 9 11 ", "911", cleaned_sentence)
        cleaned_sentence = re.sub(r"e - mail", "email", cleaned_sentence)
        cleaned_sentence = re.sub(r"j k", "jk", cleaned_sentence)
        cleaned_sentence = re.sub(r"\s{2,}", " ", cleaned_sentence)

        # Remove punctuation and possible redundant spaces
        cleaned_sentence = "".join([character for character in cleaned_sentence if character not in punctuation])
        cleaned_sentence = " ".join(cleaned_sentence.split())
        assert cleaned_sentence

        return cleaned_sentence
    except Exception as exception:
        print("Exception for {}: {}".format(original_sentence, exception))
        return result_when_failure

def load_file(original_file_path):
    processed_file_path = os.path.join(os.path.dirname(original_file_path), "processed_" + os.path.basename(original_file_path))
    if os.path.isfile(processed_file_path):
        print("Loading {} ...".format(processed_file_path))
        file_content = pd.read_csv(processed_file_path, encoding="utf-8")
    else:
        print("Loading {} ...".format(original_file_path))
        file_content = pd.read_csv(original_file_path, encoding="utf-8")
        file_content["question1"] = file_content["question1"].apply(clean_sentence)
        file_content["question2"] = file_content["question2"].apply(clean_sentence)
        file_content.to_csv(processed_file_path, index=False)

    question1_list = file_content["question1"].tolist()
    question2_list = file_content["question2"].tolist()
    if "is_duplicate" in file_content.columns:
        is_duplicate_list = file_content["is_duplicate"].tolist()
        return question1_list, question2_list, is_duplicate_list
    else:
        return question1_list, question2_list

def run():
    print("Loading text files ...")
    train_text_1_list, train_text_2_list, train_label_list = load_file(TRAIN_FILE_PATH)
    test_text_1_list, test_text_2_list = load_file(TEST_FILE_PATH)

    print("All done!")

if __name__ == "__main__":
    run()
