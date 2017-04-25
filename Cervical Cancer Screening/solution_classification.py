from __future__ import absolute_import, division, print_function

import matplotlib
matplotlib.use("Agg")

import os
import glob
import shutil
import pylab
import numpy as np
import pandas as pd
from keras.applications.resnet50 import conv_block, identity_block, preprocess_input, ResNet50, TH_WEIGHTS_PATH_NO_TOP
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Input, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.visualize_util import plot
from sklearn.model_selection import GroupShuffleSplit

# Dataset
# TODO: Modify variables related to paths
DATASET_FOLDER_PATH = os.path.join(os.path.expanduser("~"), "Documents/Dataset/The Nature Conservancy Fisheries Monitoring")
TRAIN_FOLDER_PATH = os.path.join(DATASET_FOLDER_PATH, "train")
TEST_FOLDER_PATH = os.path.join(DATASET_FOLDER_PATH, "test_stg1")

# Workspace
WORKSPACE_FOLDER_PATH = os.path.join("/tmp", os.path.basename(DATASET_FOLDER_PATH))
ACTUAL_DATASET_FOLDER_PATH = os.path.join(WORKSPACE_FOLDER_PATH, "actual_dataset")
ACTUAL_TRAIN_FOLDER_PATH = os.path.join(ACTUAL_DATASET_FOLDER_PATH, "train")
ACTUAL_VALID_FOLDER_PATH = os.path.join(ACTUAL_DATASET_FOLDER_PATH, "valid")

# Output
OUTPUT_FOLDER_PATH = os.path.join(DATASET_FOLDER_PATH, "{}_output".format(os.path.basename(__file__).split(".")[0]))
OPTIMAL_WEIGHTS_FOLDER_PATH = os.path.join(OUTPUT_FOLDER_PATH, "Optimal Weights")
OPTIMAL_WEIGHTS_FILE_RULE = os.path.join(OPTIMAL_WEIGHTS_FOLDER_PATH, "epoch_{epoch:03d}-loss_{loss:.5f}-val_loss_{val_loss:.5f}.h5")
SUBMISSION_FOLDER_PATH = os.path.join(OUTPUT_FOLDER_PATH, "submission")
TRIAL_NUM = 10

# Image processing
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256

# Training and Testing procedure
PERFORM_TRAINING = True
WEIGHTS_FILE_PATH = None
MAXIMUM_EPOCH_NUM = 1000
PATIENCE = 100
BATCH_SIZE = 32
SEED = 0

def perform_CV(image_path_list):
    # TODO: Use 20% training samples for validation. Make sure to that there is no information leakage. Check http://scikit-learn.org/stable/modules/cross_validation.html#leave-p-groups-out.
    cv_object = GroupShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    for cv_index, (train_index_array, valid_index_array) in enumerate(cv_object.split(X=np.zeros((len(cluster_ID_array), 1)), groups=cluster_ID_array), start=1):
        print("Checking cv {} ...".format(cv_index))
        valid_sample_ratio = len(valid_index_array) / (len(train_index_array) + len(valid_index_array))
        if -1 in np.unique(cluster_ID_array[train_index_array]) and valid_sample_ratio > 0.15 and valid_sample_ratio < 0.25:
            train_unique_label, train_unique_counts = np.unique([image_path.split("/")[-2] for image_path in np.array(image_path_list)[train_index_array]], return_counts=True)
            valid_unique_label, valid_unique_counts = np.unique([image_path.split("/")[-2] for image_path in np.array(image_path_list)[valid_index_array]], return_counts=True)
            if np.array_equal(train_unique_label, valid_unique_label):
                train_unique_ratio = train_unique_counts / np.sum(train_unique_counts)
                valid_unique_ratio = valid_unique_counts / np.sum(valid_unique_counts)
                print("Using {:.2f}% original training samples as validation samples ...".format(valid_sample_ratio * 100))
                print("For training samples: {}".format(train_unique_ratio))
                print("For validation samples: {}".format(valid_unique_ratio))
                return train_index_array, valid_index_array

    assert False

def reorganize_dataset():
    # Get list of files
    original_image_path_list = sorted(glob.glob(os.path.join(TRAIN_FOLDER_PATH, "*/*")))

    # Perform Cross Validation
    train_index_array, valid_index_array = perform_CV(original_image_path_list)

    # Create symbolic links
    shutil.rmtree(ACTUAL_DATASET_FOLDER_PATH, ignore_errors=True)
    for folder_path, index_array in zip((ACTUAL_TRAIN_FOLDER_PATH, ACTUAL_VALID_FOLDER_PATH), (train_index_array, valid_index_array)):
        for index_value in index_array:
            original_image_path = original_image_path_list[index_value]
            path_suffix = original_image_path[len(TRAIN_FOLDER_PATH):]
            actual_original_image_path = folder_path + path_suffix
            os.makedirs(os.path.abspath(os.path.join(actual_original_image_path, os.pardir)), exist_ok=True)
            os.symlink(original_image_path, actual_original_image_path)

    return len(glob.glob(os.path.join(ACTUAL_TRAIN_FOLDER_PATH, "*/*"))), len(glob.glob(os.path.join(ACTUAL_VALID_FOLDER_PATH, "*/*")))

def init_model(image_height=224, image_width=224, unique_label_num=1000, learning_rate=0.0001):
    def set_model_trainable_properties(model, trainable):
        for layer in model.layers:
            layer.trainable = trainable
        model.trainable = trainable

    def get_feature_extractor(input_shape):
        feature_extractor = ResNet50(include_top=False, weights="imagenet", input_shape=input_shape)
        feature_extractor = Model(input=feature_extractor.input, output=feature_extractor.get_layer("activation_40").output)
        set_model_trainable_properties(feature_extractor, False)
        return feature_extractor

    def get_trainable_classifier(input_shape, unique_label_num):
        input_tensor = Input(shape=input_shape)

        x = conv_block(input_tensor, 3, [512, 512, 2048], stage=5, block="a")
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block="b")
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block="c")

        x = GlobalAveragePooling2D(name="global_avg_pool")(x)
        x = Dense(unique_label_num, activation="softmax", name="fc{}".format(unique_label_num))(x)

        model = Model(input_tensor, x)
        return model

    # Initiate the input tensor
    input_tensor = Input(shape=(3, image_height, image_width))

    # Define the feature extractor
    feature_extractor = get_feature_extractor(input_shape=input_tensor._keras_shape[1:])
    output_tensor = feature_extractor(input_tensor)

    # Define the trainable classifier
    trainable_classifier = get_trainable_classifier(input_shape=feature_extractor.output_shape[1:], unique_label_num=unique_label_num)
    output_tensor = trainable_classifier(output_tensor)

    # Define the overall model
    model = Model(input_tensor, output_tensor)
    model.compile(optimizer=Adam(lr=learning_rate), loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    # Plot the model structures
    plot(feature_extractor, to_file=os.path.join(OPTIMAL_WEIGHTS_FOLDER_PATH, "feature_extractor.png"), show_shapes=True, show_layer_names=True)
    plot(trainable_classifier, to_file=os.path.join(OPTIMAL_WEIGHTS_FOLDER_PATH, "trainable_classifier.png"), show_shapes=True, show_layer_names=True)
    plot(model, to_file=os.path.join(OPTIMAL_WEIGHTS_FOLDER_PATH, "model.png"), show_shapes=True, show_layer_names=True)

    # Load weights for the trainable classifier
    trainable_classifier.load_weights(os.path.expanduser(os.path.join("~", ".keras/models", TH_WEIGHTS_PATH_NO_TOP.split("/")[-1])), by_name=True)

    # Load weights if applicable
    if WEIGHTS_FILE_PATH is not None:
        assert os.path.isfile(WEIGHTS_FILE_PATH), "Could not find file {}!".format(WEIGHTS_FILE_PATH)
        print("Loading weights from {} ...".format(WEIGHTS_FILE_PATH))
        model.load_weights(WEIGHTS_FILE_PATH)

    return model

def load_dataset(folder_path, classes=None, class_mode=None, batch_size=BATCH_SIZE, shuffle=True, seed=None):
    # Get the generator of the dataset
    data_generator_object = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        zoom_range=0.05,
        horizontal_flip=True,
        vertical_flip=True,
        preprocessing_function=lambda sample: preprocess_input(np.array([sample]))[0])
    data_generator = data_generator_object.flow_from_directory(
        directory=folder_path,
        target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        color_mode="rgb",
        classes=classes,
        class_mode=class_mode,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed)

    return data_generator

class InspectLossAccuracy(Callback):
    def __init__(self):
        super(InspectLossAccuracy, self).__init__()

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
        pylab.savefig(os.path.join(OUTPUT_FOLDER_PATH, "Loss Curve.png"))
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
        pylab.savefig(os.path.join(OUTPUT_FOLDER_PATH, "Accuracy Curve.png"))
        pylab.close()

def ensemble_predictions(submission_folder_path):
    def _ensemble_predictions(ensemble_func, ensemble_submission_file_name):
        ensemble_proba = ensemble_func(proba_array, axis=0)
        ensemble_proba = ensemble_proba / np.sum(ensemble_proba, axis=1)[:, np.newaxis]
        ensemble_submission_file_content.loc[:, proba_columns] = ensemble_proba
        ensemble_submission_file_content.to_csv(os.path.join(submission_folder_path, ensemble_submission_file_name), index=False)

    # Read predictions
    submission_file_path_list = glob.glob(os.path.join(submission_folder_path, "Trial_*.csv"))
    print("There are {} submissions in total.".format(len(submission_file_path_list)))
    submission_file_content_list = [pd.read_csv(submission_file_path) for submission_file_path in submission_file_path_list]
    ensemble_submission_file_content = submission_file_content_list[0]

    # Concatenate predictions
    proba_columns = ensemble_submission_file_content.columns[1:]
    proba_list = [np.expand_dims(submission_file_content.as_matrix(proba_columns), axis=0)
                  for submission_file_content in submission_file_content_list]
    proba_array = np.vstack(proba_list)

    # Ensemble predictions
    for ensemble_func, ensemble_submission_file_name in \
        zip([np.max, np.min, np.mean, np.median], ["max.csv", "min.csv", "mean.csv", "median.csv"]):
        _ensemble_predictions(ensemble_func, ensemble_submission_file_name)

def run():
    print("Creating folders ...")
    os.makedirs(OPTIMAL_WEIGHTS_FOLDER_PATH, exist_ok=True)
    os.makedirs(SUBMISSION_FOLDER_PATH, exist_ok=True)

    print("Reorganizing dataset ...")
    train_sample_num, valid_sample_num = reorganize_dataset()

    print("Getting the labels ...")
    unique_label_list = sorted([folder_name for folder_name in os.listdir(TRAIN_FOLDER_PATH) if os.path.isdir(os.path.join(TRAIN_FOLDER_PATH, folder_name))])

    print("Initializing model ...")
    model = init_model(image_height=IMAGE_HEIGHT, image_width=IMAGE_WIDTH, unique_label_num=len(unique_label_list))

    if PERFORM_TRAINING:
        print("Performing the training procedure ...")
        train_generator = load_dataset(ACTUAL_TRAIN_FOLDER_PATH, classes=unique_label_list, class_mode="categorical", shuffle=True, seed=SEED)
        valid_generator = load_dataset(ACTUAL_VALID_FOLDER_PATH, classes=unique_label_list, class_mode="categorical", shuffle=True, seed=SEED)
        earlystopping_callback = EarlyStopping(monitor="val_loss", patience=PATIENCE)
        modelcheckpoint_callback = ModelCheckpoint(OPTIMAL_WEIGHTS_FILE_RULE, monitor="val_loss", save_best_only=True, save_weights_only=True)
        inspectlossaccuracy_callback = InspectLossAccuracy()
        model.fit_generator(generator=train_generator,
                            samples_per_epoch=train_sample_num,
                            validation_data=valid_generator,
                            nb_val_samples=valid_sample_num,
                            callbacks=[earlystopping_callback, modelcheckpoint_callback, inspectlossaccuracy_callback],
                            nb_epoch=MAXIMUM_EPOCH_NUM, verbose=2)
    else:
        assert WEIGHTS_FILE_PATH is not None

        print("Performing the testing procedure ...")
        for trial_index in np.arange(TRIAL_NUM) + 1:
            print("Working on trial {}/{} ...".format(trial_index, TRIAL_NUM))
            submission_file_path = os.path.join(SUBMISSION_FOLDER_PATH, "Trial_{}.csv".format(trial_index))
            if not os.path.isfile(submission_file_path):
                print("Performing the testing procedure ...")
                test_generator = load_dataset(TEST_FOLDER_PATH, shuffle=False, seed=trial_index)
                prediction_array = model.predict_generator(generator=test_generator, val_samples=len(test_generator.filenames))
                image_name_array = np.expand_dims([os.path.basename(image_path) for image_path in test_generator.filenames], axis=-1)
                index_array_for_sorting = np.argsort(image_name_array, axis=0)
                submission_file_content = pd.DataFrame(np.hstack((image_name_array, prediction_array))[index_array_for_sorting.flat])
                submission_file_content.to_csv(submission_file_path, header=["image"] + unique_label_list, index=False)

        print("Performing ensembling ...")
        ensemble_predictions(SUBMISSION_FOLDER_PATH)

    print("All done!")

if __name__ == "__main__":
    run()
