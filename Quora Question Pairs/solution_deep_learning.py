from __future__ import absolute_import, division, print_function

import matplotlib
matplotlib.use("Agg")

import os
import re
import glob
import pylab
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from string import ascii_lowercase, punctuation
from keras import backend as K
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Embedding, Input, Lambda, LSTM, merge
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Nadam
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.visualize_util import plot
from sklearn.model_selection import StratifiedKFold

# Dataset
PROJECT_NAME = "Quora Question Pairs"
PROJECT_FOLDER_PATH = os.path.join(os.path.expanduser("~"), "Documents/Dataset", PROJECT_NAME)
TRAIN_FILE_PATH = os.path.join(PROJECT_FOLDER_PATH, "train.csv")
TEST_FILE_PATH = os.path.join(PROJECT_FOLDER_PATH, "test.csv")
EMBEDDING_FILE_PATH = os.path.join(PROJECT_FOLDER_PATH, "GoogleNews-vectors-negative300.bin")
DATASET_FILE_PATH = os.path.join(PROJECT_FOLDER_PATH, "deep_learning_dataset.npz")
MAX_SEQUENCE_LENGTH = 30

# Output
OUTPUT_FOLDER_PATH = os.path.join(PROJECT_FOLDER_PATH, "{}_output".format(os.path.basename(__file__).split(".")[0]))
OPTIMAL_WEIGHTS_FOLDER_PATH = os.path.join(OUTPUT_FOLDER_PATH, "Optimal Weights")
SUBMISSION_FOLDER_PATH = os.path.join(OUTPUT_FOLDER_PATH, "Submission")

# Training and Testing procedure
SPLIT_NUM = 10
RANDOM_STATE = 666666
PATIENCE = 4
BATCH_SIZE = 2048
MAXIMUM_EPOCH_NUM = 1000
TARGET_MEAN_PREDICTION = 0.175  # https://www.kaggle.com/davidthaler/how-many-1-s-are-in-the-public-lb

def correct_typo(word, word_to_index_dict, known_typo_dict, min_word_length=8):
    def get_candidate_word_list(word):
        # https://www.kaggle.com/cpmpml/spell-checker-using-word2vec/notebook
        left_word_with_right_word_list = [(word[:index], word[index:]) for index in range(len(word) + 1)]
        deleted_word_list = [left_word + right_word[1:] for left_word, right_word in left_word_with_right_word_list if right_word]
        transposed_word_list = [left_word + right_word[1] + right_word[0] + right_word[2:] for left_word, right_word in left_word_with_right_word_list if len(right_word) > 1]
        replaced_word_list = [left_word + character + right_word[1:] for left_word, right_word in left_word_with_right_word_list if right_word for character in ascii_lowercase]
        inserted_word_list = [left_word + character + right_word for left_word, right_word in left_word_with_right_word_list for character in ascii_lowercase]
        return list(set(deleted_word_list + transposed_word_list + replaced_word_list + inserted_word_list))

    if word in word_to_index_dict:
        return word

    if len(word) < min_word_length:
        return ""

    if word in known_typo_dict:
        return known_typo_dict[word]

    candidate_word_list = get_candidate_word_list(word)
    candidate_word_with_index_array = np.array([(candidate_word, word_to_index_dict[candidate_word]) for candidate_word in candidate_word_list if candidate_word in word_to_index_dict])
    if len(candidate_word_with_index_array) == 0:
        selected_candidate_word = ""
    else:
        selected_candidate_word = candidate_word_with_index_array[np.argmin(candidate_word_with_index_array[:, -1].astype(np.int))][0]
        print("Replacing {} with {} ...".format(word, selected_candidate_word))

    known_typo_dict[word] = selected_candidate_word
    return selected_candidate_word

def clean_sentence(original_sentence, word_to_index_dict, known_typo_dict, result_when_failure="empty"):
    # https://www.kaggle.com/currie32/quora-question-pairs/the-importance-of-cleaning-text
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

        # Correct simple typos
        cleaned_sentence = " ".join([correct_typo(word, word_to_index_dict, known_typo_dict) for word in cleaned_sentence.split()])
        cleaned_sentence = " ".join([word for word in cleaned_sentence.split()])

        # Check the length of the cleaned sentence
        assert cleaned_sentence

        return cleaned_sentence
    except Exception as exception:
        print("Exception for {}: {}".format(original_sentence, exception))
        return result_when_failure

def load_file(original_file_path, word_to_index_dict, known_typo_dict):
    processed_file_path = os.path.join(os.path.dirname(original_file_path), "processed_" + os.path.basename(original_file_path))
    if os.path.isfile(processed_file_path):
        print("Loading {} ...".format(processed_file_path))
        file_content = pd.read_csv(processed_file_path, encoding="utf-8")
    else:
        print("Loading {} ...".format(original_file_path))
        file_content = pd.read_csv(original_file_path, encoding="utf-8")

        print("Cleaning sentences ...")
        file_content["processed_question1"] = file_content["question1"].apply(lambda original_sentence: clean_sentence(original_sentence, word_to_index_dict, known_typo_dict))
        file_content["processed_question2"] = file_content["question2"].apply(lambda original_sentence: clean_sentence(original_sentence, word_to_index_dict, known_typo_dict))

        print("Saving processed file ...")
        interesting_column_name_list = ["processed_question1", "processed_question2"]
        if "is_duplicate" in file_content.columns:
            interesting_column_name_list.append("is_duplicate")
        file_content = file_content[interesting_column_name_list]
        file_content.to_csv(processed_file_path, index=False)

    question1_text_list = file_content["processed_question1"].tolist()
    question2_text_list = file_content["processed_question2"].tolist()

    if "is_duplicate" in file_content.columns:
        is_duplicate_list = file_content["is_duplicate"].tolist()
        return question1_text_list, question2_text_list, is_duplicate_list
    else:
        return question1_text_list, question2_text_list

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
    else:
        print("Initiating word2vec ...")
        word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE_PATH, binary=True)
        word_to_index_dict = dict([(word, index) for index, word in enumerate(word2vec.index2word)])
        print("word2vec contains {} unique words.".format(len(word_to_index_dict)))

        print("Loading text files ...")
        known_typo_dict = {}
        train_text_1_list, train_text_2_list, train_label_list = load_file(TRAIN_FILE_PATH, word_to_index_dict, known_typo_dict)
        test_text_1_list, test_text_2_list = load_file(TEST_FILE_PATH, word_to_index_dict, known_typo_dict)

        print("Initiating tokenizer ...")
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(train_text_1_list + train_text_2_list + test_text_1_list + test_text_2_list)
        print("Dataset contains {} unique words.".format(len(tokenizer.word_index)))

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

        print("Initiating embedding matrix ...")
        embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, word2vec.vector_size), dtype=np.float32)
        for word, index in tokenizer.word_index.items():
            assert word in word_to_index_dict
            embedding_matrix[index] = word2vec.word_vec(word)
        assert np.sum(np.isclose(np.sum(embedding_matrix, axis=1), 0)) == 1

        print("Converting to numpy array ...")
        train_label_array = np.array(train_label_list, dtype=np.bool)

        print("Saving dataset to disk ...")
        np.savez_compressed(DATASET_FILE_PATH,
                            train_data_1_array=train_data_1_array, train_data_2_array=train_data_2_array,
                            test_data_1_array=test_data_1_array, test_data_2_array=test_data_2_array,
                            train_label_array=train_label_array, embedding_matrix=embedding_matrix)

    return train_data_1_array, train_data_2_array, test_data_1_array, test_data_2_array, \
        train_label_array, embedding_matrix

def init_model(embedding_matrix, learning_rate=0.002):
    def get_sentence_feature_extractor(embedding_matrix):
        input_tensor = Input(shape=(None,), dtype="int32")
        output_tensor = Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1],
                                input_length=None, mask_zero=True, weights=[embedding_matrix], trainable=False)(input_tensor)
        output_tensor = LSTM(output_dim=256, dropout_W=0.3, dropout_U=0.3, activation="tanh", return_sequences=False)(output_tensor)
        output_tensor = BatchNormalization()(output_tensor)
        output_tensor = Dropout(0.3)(output_tensor)

        model = Model(input_tensor, output_tensor)
        return model

    def get_binary_classifier(input_shape):
        input_tensor = Input(shape=input_shape)
        output_tensor = Dense(128, activation="relu")(input_tensor)
        output_tensor = BatchNormalization()(output_tensor)
        output_tensor = Dropout(0.3)(output_tensor)
        output_tensor = Dense(1, activation="sigmoid")(output_tensor)

        model = Model(input_tensor, output_tensor)
        return model

    # Initiate the input tensors
    input_data_1_tensor = Input(shape=(None,), dtype="int32")
    input_data_2_tensor = Input(shape=(None,), dtype="int32")

    # Define the sentence feature extractor
    sentence_feature_extractor = get_sentence_feature_extractor(embedding_matrix)
    input_1_feature_tensor = sentence_feature_extractor(input_data_1_tensor)
    input_2_feature_tensor = sentence_feature_extractor(input_data_2_tensor)
    merged_feature_1_tensor = merge([input_1_feature_tensor, input_2_feature_tensor], mode="concat")
    merged_feature_2_tensor = merge([input_2_feature_tensor, input_1_feature_tensor], mode="concat")

    # Define the binary classifier
    binary_classifier = get_binary_classifier(input_shape=(K.int_shape(merged_feature_1_tensor)[1],))
    output_1_tensor = binary_classifier(merged_feature_1_tensor)
    output_2_tensor = binary_classifier(merged_feature_2_tensor)
    output_tensor = merge([output_1_tensor, output_2_tensor], mode="concat", concat_axis=1)
    output_tensor = Lambda(lambda x: K.mean(x, axis=1, keepdims=True), output_shape=(1,))(output_tensor)

    # Define the overall model
    model = Model([input_data_1_tensor, input_data_2_tensor], output_tensor)
    model.compile(optimizer=Nadam(lr=learning_rate), loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()

    # Plot the model structures
    plot(sentence_feature_extractor, to_file=os.path.join(OPTIMAL_WEIGHTS_FOLDER_PATH, "sentence_feature_extractor.png"), show_shapes=True, show_layer_names=True)
    plot(binary_classifier, to_file=os.path.join(OPTIMAL_WEIGHTS_FOLDER_PATH, "binary_classifier.png"), show_shapes=True, show_layer_names=True)
    plot(model, to_file=os.path.join(OPTIMAL_WEIGHTS_FOLDER_PATH, "model.png"), show_shapes=True, show_layer_names=True)

    return model

class InspectLossAccuracy(Callback):
    def __init__(self, *args, **kwargs):
        self.split_index = kwargs.pop("split_index", None)
        super(InspectLossAccuracy, self).__init__(*args, **kwargs)

        self.train_loss_list = []
        self.valid_loss_list = []

        self.train_acc_list = []
        self.valid_acc_list = []

    def on_epoch_end(self, epoch, logs=None):
        # Loss
        train_loss = logs.get("loss")
        valid_loss = logs.get("val_loss")
        self.train_loss_list.append(train_loss)
        self.valid_loss_list.append(valid_loss)
        epoch_index_array = np.arange(len(self.train_loss_list)) + 1

        pylab.figure()
        pylab.plot(epoch_index_array, self.train_loss_list, "yellowgreen", label="train_loss")
        pylab.plot(epoch_index_array, self.valid_loss_list, "lightskyblue", label="valid_loss")
        pylab.grid()
        pylab.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2, ncol=2, mode="expand", borderaxespad=0.)
        pylab.savefig(os.path.join(OUTPUT_FOLDER_PATH, "loss_curve_{}.png".format(self.split_index)))
        pylab.close()

        # Accuracy
        train_acc = logs.get("acc")
        valid_acc = logs.get("val_acc")
        self.train_acc_list.append(train_acc)
        self.valid_acc_list.append(valid_acc)
        epoch_index_array = np.arange(len(self.train_acc_list)) + 1

        pylab.figure()
        pylab.plot(epoch_index_array, self.train_acc_list, "yellowgreen", label="train_acc")
        pylab.plot(epoch_index_array, self.valid_acc_list, "lightskyblue", label="valid_acc")
        pylab.grid()
        pylab.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2, ncol=2, mode="expand", borderaxespad=0.)
        pylab.savefig(os.path.join(OUTPUT_FOLDER_PATH, "accuracy_curve_{}.png".format(self.split_index)))
        pylab.close()

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
    train_data_1_array, train_data_2_array, test_data_1_array, test_data_2_array, train_label_array, embedding_matrix = load_dataset()

    print("Initializing model ...")
    model = init_model(embedding_matrix)
    vanilla_weights = model.get_weights()

    cv_object = StratifiedKFold(n_splits=SPLIT_NUM, random_state=RANDOM_STATE)
    for split_index, (train_index_array, valid_index_array) in enumerate(cv_object.split(np.zeros((len(train_label_array), 1)), train_label_array), start=1):
        print("Working on splitting fold {} ...".format(split_index))

        submission_file_path = os.path.join(SUBMISSION_FOLDER_PATH, "submission_{}.csv".format(split_index))
        if os.path.isfile(submission_file_path):
            print("The submission file already exists.")
            continue

        optimal_weights_file_path = os.path.join(OPTIMAL_WEIGHTS_FOLDER_PATH, "optimal_weights_{}.h5".format(split_index))
        if os.path.isfile(optimal_weights_file_path):
            print("The optimal weights file already exists.")
        else:
            print("Dividing the vanilla training dataset to actual training/validation dataset ...")
            actual_train_data_1_array, actual_train_data_2_array, actual_train_label_array = train_data_1_array[train_index_array], train_data_2_array[train_index_array], train_label_array[train_index_array]
            actual_valid_data_1_array, actual_valid_data_2_array, actual_valid_label_array = train_data_1_array[valid_index_array], train_data_2_array[valid_index_array], train_label_array[valid_index_array]

            print("Calculating class weight ...")
            train_mean_prediction = np.mean(actual_train_label_array)
            train_class_weight = {0: (1 - TARGET_MEAN_PREDICTION) / (1 - train_mean_prediction), 1: TARGET_MEAN_PREDICTION / train_mean_prediction}
            valid_mean_prediction = np.mean(actual_valid_label_array)
            valid_class_weight = {0: (1 - TARGET_MEAN_PREDICTION) / (1 - valid_mean_prediction), 1: TARGET_MEAN_PREDICTION / valid_mean_prediction}

            print("Startting with vanilla weights ...")
            model.set_weights(vanilla_weights)

            print("Performing the training procedure ...")
            valid_sample_weights = np.ones(len(actual_valid_label_array)) * valid_class_weight[1]
            valid_sample_weights[np.logical_not(actual_valid_label_array)] = valid_class_weight[0]
            earlystopping_callback = EarlyStopping(monitor="val_loss", patience=PATIENCE)
            modelcheckpoint_callback = ModelCheckpoint(optimal_weights_file_path, monitor="val_loss", save_best_only=True, save_weights_only=True)
            inspectlossaccuracy_callback = InspectLossAccuracy(split_index=split_index)
            model.fit([actual_train_data_1_array, actual_train_data_2_array], actual_train_label_array, batch_size=BATCH_SIZE,
                    validation_data=([actual_valid_data_1_array, actual_valid_data_2_array], actual_valid_label_array, valid_sample_weights),
                    callbacks=[earlystopping_callback, modelcheckpoint_callback, inspectlossaccuracy_callback],
                    class_weight=train_class_weight, nb_epoch=MAXIMUM_EPOCH_NUM, verbose=2)

        assert os.path.isfile(optimal_weights_file_path)
        model.load_weights(optimal_weights_file_path)

        print("Performing the testing procedure ...")
        prediction_array = model.predict([test_data_1_array, test_data_2_array], batch_size=BATCH_SIZE, verbose=2)
        submission_file_content = pd.DataFrame({"test_id": np.arange(len(prediction_array)), "is_duplicate": np.squeeze(prediction_array)})
        submission_file_content.to_csv(submission_file_path, index=False)

    print("Performing ensembling ...")
    ensemble_predictions(submission_folder_path=SUBMISSION_FOLDER_PATH, proba_column_name="is_duplicate")

    print("All done!")

if __name__ == "__main__":
    run()
