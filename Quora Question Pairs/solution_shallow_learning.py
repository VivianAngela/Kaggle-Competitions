from __future__ import absolute_import, division, print_function

import matplotlib
matplotlib.use("Agg")

import os
import glob
import pylab
import numpy as np
import pandas as pd
from collections import Counter
from joblib import Parallel, delayed
from keras import backend as K
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Input, Lambda, merge
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.utils.visualize_util import plot
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
PATIENCE = 4
BATCH_SIZE = 256
MAXIMUM_EPOCH_NUM = 1000
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
        merged_file_content = get_magic_feature(merged_file_content)

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

def init_model(question_feature_dim, common_feature_dim, learning_rate=0.0001):
    def get_binary_classifier(input_shape, vanilla_dense_size=512, block_num=3):
        input_tensor = Input(shape=input_shape)
        output_tensor = BatchNormalization()(input_tensor)
        for block_index in np.arange(block_num):
            output_tensor = Dense(int(vanilla_dense_size / (2 ** block_index)), activation="relu")(output_tensor)
            output_tensor = BatchNormalization()(output_tensor)
            output_tensor = Dropout(0.3)(output_tensor)
        output_tensor = Dense(1, activation="sigmoid")(output_tensor)

        model = Model(input_tensor, output_tensor)
        return model

    # Initiate the input tensors
    question1_feature_tensor = Input(shape=(question_feature_dim,))
    question2_feature_tensor = Input(shape=(question_feature_dim,))
    common_feature_tensor = Input(shape=(common_feature_dim,))

    # Define the sentence feature extractor
    merged_feature_1_tensor = merge([question1_feature_tensor, question2_feature_tensor, common_feature_tensor], mode="concat")
    merged_feature_2_tensor = merge([question2_feature_tensor, question1_feature_tensor, common_feature_tensor], mode="concat")

    # Define the binary classifier
    binary_classifier = get_binary_classifier(input_shape=(K.int_shape(merged_feature_1_tensor)[1],))
    output_1_tensor = binary_classifier(merged_feature_1_tensor)
    output_2_tensor = binary_classifier(merged_feature_2_tensor)
    output_tensor = merge([output_1_tensor, output_2_tensor], mode="concat", concat_axis=1)
    output_tensor = Lambda(lambda x: K.mean(x, axis=1, keepdims=True), output_shape=(1,))(output_tensor)

    # Define the overall model
    model = Model([question1_feature_tensor, question2_feature_tensor, common_feature_tensor], output_tensor)
    model.compile(optimizer=Adam(lr=learning_rate), loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()

    # Plot the model structures
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
    train_question1_feature_array, train_question2_feature_array, train_common_feature_array, train_label_array, \
        test_question1_feature_array, test_question2_feature_array, test_common_feature_array = load_dataset()

    print("Initializing model ...")
    model = init_model(question_feature_dim=train_question1_feature_array.shape[-1], common_feature_dim=train_common_feature_array.shape[-1])
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

            print("Startting with vanilla weights ...")
            model.set_weights(vanilla_weights)

            print("Performing the training procedure ...")
            valid_sample_weights = np.ones(len(actual_valid_label_array)) * valid_class_weight[1]
            valid_sample_weights[np.logical_not(actual_valid_label_array)] = valid_class_weight[0]
            earlystopping_callback = EarlyStopping(monitor="val_loss", patience=PATIENCE)
            modelcheckpoint_callback = ModelCheckpoint(optimal_weights_file_path, monitor="val_loss", save_best_only=True, save_weights_only=True)
            inspectlossaccuracy_callback = InspectLossAccuracy(split_index=split_index)
            model.fit([actual_train_question1_feature_array, actual_train_question2_feature_array, actual_train_common_feature_array], actual_train_label_array, batch_size=BATCH_SIZE,
                    validation_data=([actual_valid_question1_feature_array, actual_valid_question2_feature_array, actual_valid_common_feature_array], actual_valid_label_array, valid_sample_weights),
                    callbacks=[earlystopping_callback, modelcheckpoint_callback, inspectlossaccuracy_callback],
                    class_weight=train_class_weight, nb_epoch=MAXIMUM_EPOCH_NUM, verbose=2)

        assert os.path.isfile(optimal_weights_file_path)
        model.load_weights(optimal_weights_file_path)

        print("Performing the testing procedure ...")
        prediction_array = model.predict([test_question1_feature_array, test_question2_feature_array, test_common_feature_array], batch_size=BATCH_SIZE, verbose=2)
        submission_file_content = pd.DataFrame({"test_id":np.arange(len(prediction_array)), "is_duplicate":np.squeeze(prediction_array)})
        submission_file_content.to_csv(submission_file_path, index=False)

    print("Performing ensembling ...")
    ensemble_predictions(submission_folder_path=SUBMISSION_FOLDER_PATH, proba_column_name="is_duplicate")

    print("All done!")

if __name__ == "__main__":
    run()
