from __future__ import absolute_import, division, print_function

import os
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from string import ascii_lowercase, punctuation

# Dataset
PROJECT_NAME = "Quora Question Pairs"
PROJECT_FOLDER_PATH = os.path.join(os.path.expanduser("~"), "Documents/Dataset", PROJECT_NAME)
TRAIN_FILE_PATH = os.path.join(PROJECT_FOLDER_PATH, "train.csv")
TEST_FILE_PATH = os.path.join(PROJECT_FOLDER_PATH, "test.csv")
EMBEDDING_FILE_PATH = os.path.join(PROJECT_FOLDER_PATH, "GoogleNews-vectors-negative300.bin")

def correct_typo(word, word_to_index_dict, known_typo_dict, min_word_length=8):
    def get_candidate_word_list(word):
        # https://www.kaggle.com/cpmpml/spell-checker-using-word2vec/notebook
        left_word_with_right_word_list = [(word[:index], word[index:]) for index in range(len(word) + 1)]
        deleted_word_list = [left_word + right_word[1:] for left_word, right_word in left_word_with_right_word_list if right_word]
        transposed_word_list = [left_word + right_word[1] + right_word[0] + right_word[2:] for left_word, right_word in left_word_with_right_word_list if len(right_word) > 1]
        replaced_word_list = [left_word + character + right_word[1:] for left_word, right_word in left_word_with_right_word_list if right_word for character in ascii_lowercase]
        inserted_word_list = [left_word + character + right_word for left_word, right_word in left_word_with_right_word_list for character in ascii_lowercase]
        return list(set(deleted_word_list + transposed_word_list + replaced_word_list + inserted_word_list))

    if len(word) < min_word_length or word in word_to_index_dict:
        return word

    if word in known_typo_dict:
        return known_typo_dict[word]

    candidate_word_list = get_candidate_word_list(word)
    candidate_word_with_index_array = np.array([(candidate_word, word_to_index_dict[candidate_word]) for candidate_word in candidate_word_list if candidate_word in word_to_index_dict])
    if len(candidate_word_with_index_array) == 0:
        selected_candidate_word = word
    else:
        selected_candidate_word = candidate_word_with_index_array[np.argmin(candidate_word_with_index_array[:, -1].astype(np.int))][0]
        print("Replacing {} with {} ...".format(word, selected_candidate_word))

    known_typo_dict[word] = selected_candidate_word
    return selected_candidate_word

def clean_sentence(original_sentence, word_to_index_dict, known_typo_dict):
    # https://www.kaggle.com/currie32/quora-question-pairs/the-importance-of-cleaning-text
    try:
        # Convert to lower case
        cleaned_sentence = " ".join(str(original_sentence).lower().split())

        # Remove punctuation
        cleaned_sentence = "".join([character for character in cleaned_sentence if character not in punctuation])

        # Correct simple typos
        cleaned_sentence = " ".join([correct_typo(word, word_to_index_dict, known_typo_dict) for word in cleaned_sentence.split()])
        cleaned_sentence = " ".join([word for word in cleaned_sentence.split()])

        return cleaned_sentence
    except Exception as exception:
        print("Exception for {}: {}".format(original_sentence, exception))
        return original_sentence

def process_file(original_file_path, word_to_index_dict, known_typo_dict):
    print("Loading {} ...".format(original_file_path))
    file_content = pd.read_csv(original_file_path, encoding="utf-8")

    print("Cleaning sentences ...")
    file_content["question1"] = file_content["question1"].apply(lambda original_sentence: clean_sentence(original_sentence, word_to_index_dict, known_typo_dict))
    file_content["question2"] = file_content["question2"].apply(lambda original_sentence: clean_sentence(original_sentence, word_to_index_dict, known_typo_dict))

    print("Saving processed file ...")
    file_content.to_csv(original_file_path, index=False)

def run():
    print("Initiating word2vec ...")
    word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE_PATH, binary=True)
    word_to_index_dict = dict([(word, index) for index, word in enumerate(word2vec.index2word)])
    print("word2vec contains {} unique words.".format(len(word_to_index_dict)))

    print("Processing text files ...")
    known_typo_dict = {}
    process_file(TRAIN_FILE_PATH, word_to_index_dict, known_typo_dict)
    process_file(TEST_FILE_PATH, word_to_index_dict, known_typo_dict)

    print("All done!")

if __name__ == "__main__":
    run()
