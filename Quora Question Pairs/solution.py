from __future__ import absolute_import, division, print_function

import os
import re
import numpy as np
import pandas as pd
from string import punctuation
from gensim.models import KeyedVectors
from keras import backend as K
from keras.layers import Dense, Dropout, Input, Embedding, Lambda, LSTM, merge
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import RMSprop
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

PROJECT_NAME = "Quora Question Pairs"
PROJECT_FOLDER_PATH = os.path.join(os.path.expanduser("~"), "Documents/Dataset", PROJECT_NAME)
TRAIN_FILE_PATH = os.path.join(PROJECT_FOLDER_PATH, "train.csv")
TEST_FILE_PATH = os.path.join(PROJECT_FOLDER_PATH, "test.csv")
EMBEDDING_FILE_PATH = os.path.join(PROJECT_FOLDER_PATH, "GoogleNews-vectors-negative300.bin")
DATASET_FILE_PATH = os.path.join(PROJECT_FOLDER_PATH, "dataset.npz")
MAX_SEQUENCE_LENGTH = 30

def clean_sentence(original_sentence, available_vocabulary, result_when_failure="empty"):
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

        # Remove punctuation
        cleaned_sentence = "".join([character for character in cleaned_sentence if character not in punctuation])

        # Remove words that are not in vocabulary
        cleaned_sentence = " ".join([word for word in cleaned_sentence.split() if word in available_vocabulary])

        # Check the length of the cleaned sentence
        assert cleaned_sentence

        return cleaned_sentence
    except Exception as exception:
        print("Exception for {}: {}".format(original_sentence, exception))
        return result_when_failure

def load_file(original_file_path, available_vocabulary):
    processed_file_path = os.path.join(os.path.dirname(original_file_path), "processed_" + os.path.basename(original_file_path))
    if os.path.isfile(processed_file_path):
        print("Loading {} ...".format(processed_file_path))
        file_content = pd.read_csv(processed_file_path, encoding="utf-8")
    else:
        print("Loading {} ...".format(original_file_path))
        file_content = pd.read_csv(original_file_path, encoding="utf-8")
        file_content["question1"] = file_content["question1"].apply(lambda original_sentence: clean_sentence(original_sentence, available_vocabulary))
        file_content["question2"] = file_content["question2"].apply(lambda original_sentence: clean_sentence(original_sentence, available_vocabulary))
        file_content.to_csv(processed_file_path, index=False)

    question1_list = file_content["question1"].tolist()
    question2_list = file_content["question2"].tolist()
    if "is_duplicate" in file_content.columns:
        is_duplicate_list = file_content["is_duplicate"].tolist()
        return question1_list, question2_list, is_duplicate_list
    else:
        return question1_list, question2_list

def load_dataset():
    if os.path.isfile(DATASET_FILE_PATH):
        print("Loading dataset from disk ...")
        dataset_file_content = np.load(DATASET_FILE_PATH)
        train_data_1_array = dataset_file_content["train_data_1_array"]
        train_data_2_array = dataset_file_content["train_data_2_array"]
        test_data_1_array = dataset_file_content["test_data_1_array"]
        test_data_2_array = dataset_file_content["test_data_2_array"]
        train_label_array = dataset_file_content["train_label_array"]
        embedding_matrix = dataset_file_content["embedding_matrix"]

        return train_data_1_array, train_data_2_array, test_data_1_array, test_data_2_array, train_label_array, embedding_matrix
    else:
        print("Initiating word2vec ...")
        word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE_PATH, binary=True)
        available_vocabulary = word2vec.vocab
        print("word2vec contains {} unique words.".format(len(available_vocabulary)))

        print("Loading text files ...")
        train_text_1_list, train_text_2_list, train_label_list = load_file(TRAIN_FILE_PATH, available_vocabulary)
        test_text_1_list, test_text_2_list = load_file(TEST_FILE_PATH, available_vocabulary)

        print("Initiating tokenizer ...")
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(train_text_1_list + train_text_2_list + test_text_1_list + test_text_2_list)
        print("dataset contains {} unique words.".format(len(tokenizer.word_index)))

        print("Turning texts into sequences ...")
        train_sequence_1_list = tokenizer.texts_to_sequences(train_text_1_list)
        train_sequence_2_list = tokenizer.texts_to_sequences(train_text_2_list)
        test_sequence_1_list = tokenizer.texts_to_sequences(test_text_1_list)
        test_sequence_2_list = tokenizer.texts_to_sequences(test_text_2_list)

        print("Padding sequences with fixed length ...")
        train_data_1_array = pad_sequences(train_sequence_1_list, maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post")
        train_data_2_array = pad_sequences(train_sequence_2_list, maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post")
        test_data_1_array = pad_sequences(test_sequence_1_list, maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post")
        test_data_2_array = pad_sequences(test_sequence_2_list, maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post")
        train_label_array = np.array(train_label_list, dtype=np.bool)

        print("Initiating embedding matrix ...")
        embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, word2vec.vector_size), dtype=np.float32)
        for word, index in tokenizer.word_index.items():
            assert word in available_vocabulary
            embedding_matrix[index] = word2vec.word_vec(word)
        assert np.sum(np.isclose(np.sum(embedding_matrix, axis=1), 0)) == 1

        print("Saving dataset to disk ...")
        np.savez_compressed(DATASET_FILE_PATH,
                            train_data_1_array=train_data_1_array, train_data_2_array=train_data_2_array,
                            test_data_1_array=test_data_1_array, test_data_2_array=test_data_2_array,
                            train_label_array=train_label_array, embedding_matrix=embedding_matrix)

        return train_data_1_array, train_data_2_array, test_data_1_array, test_data_2_array, train_label_array, embedding_matrix

def init_model(embedding_matrix, learning_rate=0.0001):
    def get_sentence_feature_extractor(embedding_matrix):
        input_tensor = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype="int32")
        output_tensor = Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1],
                                input_length=MAX_SEQUENCE_LENGTH, mask_zero=True, weights=[embedding_matrix], trainable=False)(input_tensor)
        output_tensor = LSTM(output_dim=256, dropout_W=0.2, dropout_U=0.2, return_sequences=False)(output_tensor)

        model = Model(input_tensor, output_tensor)
        return model

    def get_binary_classifier(input_shape):
        input_tensor = Input(shape=input_shape)
        output_tensor = input_tensor
        for _ in range(3):
            output_tensor = Dense(256, activation="linear")(output_tensor)
            output_tensor = LeakyReLU()(output_tensor)
            output_tensor = BatchNormalization()(output_tensor)
            output_tensor = Dropout(0.5)(output_tensor)
        output_tensor = Dense(1, activation="sigmoid")(output_tensor)

        model = Model(input_tensor, output_tensor)
        return model

    # Initiate the input tensors
    input_1_tensor = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype="int32")
    input_2_tensor = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype="int32")

    # Define the sentence feature extractor
    sentence_feature_extractor = get_sentence_feature_extractor(embedding_matrix)
    input_1_feature_tensor = sentence_feature_extractor(input_1_tensor)
    input_2_feature_tensor = sentence_feature_extractor(input_2_tensor)
    input_1_feature_with_input_2_feature_tensor = merge([input_1_feature_tensor, input_2_feature_tensor], mode="concat", concat_axis=1)
    input_2_feature_with_input_1_feature_tensor = merge([input_2_feature_tensor, input_1_feature_tensor], mode="concat", concat_axis=1)

    # Define the binary classifier
    binary_classifier = get_binary_classifier(input_shape=(input_1_feature_with_input_2_feature_tensor._keras_shape[1],))  # pylint: disable=W0212
    output_1_tensor = binary_classifier(input_1_feature_with_input_2_feature_tensor)
    output_2_tensor = binary_classifier(input_2_feature_with_input_1_feature_tensor)

    # Compute the mean value
    output_tensor = merge([output_1_tensor, output_2_tensor], mode="concat", concat_axis=1)
    output_tensor = Lambda(lambda x: K.mean(x, axis=1, keepdims=True), output_shape=(1,))(output_tensor)

    # Define the overall model
    model = Model([input_1_tensor, input_2_tensor], output_tensor)
    model.compile(optimizer=RMSprop(lr=learning_rate), loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()

    return model

def run():
    print("Loading dataset ...")
    train_data_1_array, train_data_2_array, test_data_1_array, test_data_2_array, train_label_array, embedding_matrix = load_dataset()

    print("Initializing model ...")
    model = init_model(embedding_matrix)

    print("All done!")

if __name__ == "__main__":
    run()
