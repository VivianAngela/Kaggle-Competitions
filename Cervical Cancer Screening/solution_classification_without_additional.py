from __future__ import absolute_import, division, print_function


import matplotlib
matplotlib.use("Agg")


import re
import os
import glob
import shutil
import pylab
import numpy as np
import pandas as pd
from keras import backend as K
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Input, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Nadam
from keras.preprocessing.image import ImageDataGenerator
#from keras.utils.vis_utils import plot_model
from keras.utils.visualize_util import plot #for 1.2.2
from sklearn.model_selection import StratifiedKFold
from data_preprocessing import PROJECT_FOLDER_PATH
from data_preprocessing import PROCESSED_DATASET_FOLDER_PATH as DATASET_FOLDER_PATH
from data_preprocessing import PROCESSED_IMAGE_HEIGHT as IMAGE_HEIGHT
from data_preprocessing import PROCESSED_IMAGE_WIDTH as IMAGE_WIDTH
from solution_classification_with_additional import OPTIMAL_WEIGHTS_FOLDER_PATH as PREVIOUS_OPTIMAL_WEIGHTS_FOLDER_PATH

<<<<<<< Updated upstream
# Choose ResNet50 or InceptionV3 or VGG16
MODEL_NAME = "ResNet50"  # "ResNet50" or "InceptionV3" or "VGG16"
=======
# Choose ResNet50 or InceptionV3
MODEL_NAME = "InceptionV3"  # "ResNet50" or "InceptionV3"
>>>>>>> Stashed changes
if MODEL_NAME == "ResNet50":
    from keras.applications.resnet50 import preprocess_input as PREPROCESS_INPUT
    from keras.applications.resnet50 import ResNet50 as INIT_FUNC
    BOTTLENECK_LAYER_NAME = "activation_40"
    DROPOUT_RATIO = 0.5
    LEARNING_RATE = 0.00001
elif MODEL_NAME == "InceptionV3":
    from keras.applications.inception_v3 import preprocess_input as PREPROCESS_INPUT
    from keras.applications.inception_v3 import InceptionV3 as INIT_FUNC
    BOTTLENECK_LAYER_NAME = "mixed8"
    DROPOUT_RATIO = 0.5
    LEARNING_RATE = 0.00001
elif MODEL_NAME == "VGG16":
    from keras.applications.vgg16 import preprocess_input as PREPROCESS_INPUT
    from keras.applications.vgg16 import VGG16 as INIT_FUNC
    BOTTLENECK_LAYER_NAME = "block4_pool"
    DROPOUT_RATIO = 0.5
    LEARNING_RATE = 0.00005
else:
    assert False

# Dataset

TRAIN_FOLDER_PATH = os.path.join(DATASET_FOLDER_PATH, "train")
TEST_FOLDER_PATH = os.path.join(DATASET_FOLDER_PATH, "test")

# Workspace
WORKSPACE_FOLDER_PATH = os.path.join("/mnt/Data1/xu", os.path.basename(DATASET_FOLDER_PATH))
ACTUAL_DATASET_FOLDER_PATH = os.path.join(WORKSPACE_FOLDER_PATH, "actual_dataset")
ACTUAL_TRAIN_FOLDER_PATH = os.path.join(ACTUAL_DATASET_FOLDER_PATH, "train")
ACTUAL_VALID_FOLDER_PATH = os.path.join(ACTUAL_DATASET_FOLDER_PATH, "valid")

# Output
OUTPUT_FOLDER_PATH = os.path.join(PROJECT_FOLDER_PATH, "phase_2")
OPTIMAL_WEIGHTS_FOLDER_PATH = os.path.join(OUTPUT_FOLDER_PATH, "optimal weights")
OPTIMAL_WEIGHTS_FILE_RULE = os.path.join(OPTIMAL_WEIGHTS_FOLDER_PATH, "epoch_{epoch:03d}-loss_{loss:.5f}-val_loss_{val_loss:.5f}.h5")
SUBMISSION_FOLDER_PATH = os.path.join(PROJECT_FOLDER_PATH, "submission")
TRIAL_NUM = 10


# Training and Testing procedure
PERFORM_TRAINING = True
<<<<<<< Updated upstream
=======
WEIGHTS_FILE_PATH = os.path.join(PROJECT_FOLDER_PATH, "phase_1/optimal weights/InceptionV3.h5")
#WEIGHTS_FILE_PATH = None
>>>>>>> Stashed changes
MAXIMUM_EPOCH_NUM = 1000
PATIENCE = 100
BATCH_SIZE = 32
SEED = 0
numbers = re.compile(r'(\d+)')
NUM_FOLD = 5

def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def label_index(image_path_list):
    label_array = np.zeros(len(image_path_list))
    for idx in range(0, len(image_path_list)):
        parts = image_path_list[idx]
        parts = numbers.split(parts)
        parts[1:2] = map(int, parts[1:2])
        label_array[idx] = parts[1:2][0]
    return (label_array)

def reorganize_dataset(original_image_path_list, train_index_array, valid_index_array):

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

def init_model(image_height, image_width, unique_label_num, init_func=INIT_FUNC, bottleneck_layer_name=BOTTLENECK_LAYER_NAME, dropout_ratio=DROPOUT_RATIO, learning_rate=LEARNING_RATE):
    def set_model_trainable_properties(model, trainable, bottleneck_layer_name):
        for layer in model.layers:
            layer.trainable = trainable
            if layer.name == bottleneck_layer_name:
                break

    def get_feature_extractor(input_shape):
        feature_extractor = init_func(include_top=False, weights="imagenet", input_shape=input_shape)
        set_model_trainable_properties(model=feature_extractor, trainable=False, bottleneck_layer_name=bottleneck_layer_name)
        return feature_extractor

    def get_dense_classifier(input_shape, unique_label_num):
        input_tensor = Input(shape=input_shape)
        output_tensor = GlobalAveragePooling2D()(input_tensor)
        output_tensor = Dropout(dropout_ratio)(output_tensor)
        output_tensor = Dense(unique_label_num, activation="softmax")(output_tensor)
        model = Model(input_tensor, output_tensor)
        return model

    # Initiate the input tensor
    if K.image_dim_ordering() == "tf":
        input_tensor = Input(shape=(image_height, image_width, 3))
    else:
        input_tensor = Input(shape=(3, image_height, image_width))

    # Define the feature extractor
    feature_extractor = get_feature_extractor(input_shape=K.int_shape(input_tensor)[1:])
    output_tensor = feature_extractor(input_tensor)

    # Define the dense classifier
    dense_classifier = get_dense_classifier(input_shape=feature_extractor.output_shape[1:], unique_label_num=unique_label_num)
    output_tensor = dense_classifier(output_tensor)

    # Define the overall model
    model = Model(input_tensor, output_tensor)
    model.compile(optimizer=Nadam(lr=learning_rate), loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    # Plot the model structures
    plot(feature_extractor, to_file=os.path.join(OPTIMAL_WEIGHTS_FOLDER_PATH, "{}_feature_extractor.png".format(MODEL_NAME)), show_shapes=True, show_layer_names=True)
    plot(dense_classifier, to_file=os.path.join(OPTIMAL_WEIGHTS_FOLDER_PATH, "{}_dense_classifier.png".format(MODEL_NAME)), show_shapes=True, show_layer_names=True)
    plot(model, to_file=os.path.join(OPTIMAL_WEIGHTS_FOLDER_PATH, "{}_model.png".format(MODEL_NAME)), show_shapes=True, show_layer_names=True)

    # Load weights from previous phase
    previous_optimal_weights_file_path = os.path.join(PREVIOUS_OPTIMAL_WEIGHTS_FOLDER_PATH, "{}.h5".format(MODEL_NAME))
    assert os.path.isfile(previous_optimal_weights_file_path), "Could not find file {}!".format(previous_optimal_weights_file_path)
    print("Loading weights from {} ...".format(previous_optimal_weights_file_path))
    model.load_weights(previous_optimal_weights_file_path)

    return model

def load_dataset(folder_path, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH), classes=None, class_mode=None, batch_size=BATCH_SIZE, shuffle=True, seed=None, preprocess_input=PREPROCESS_INPUT):
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
        target_size=target_size,
        color_mode="rgb",
        classes=classes,
        class_mode=class_mode,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed)

    return data_generator

def reformat_testing_dataset():
# Create a dummy folder
    dummy_test_folder_path = os.path.join(TEST_FOLDER_PATH, "dummy")
    os.makedirs(dummy_test_folder_path, exist_ok=True)
# Move files to the dummy folder if needed
    file_path_list = glob.glob(os.path.join(TEST_FOLDER_PATH, "*"))
    for file_path in file_path_list:
        if os.path.isfile(file_path):
            shutil.move(file_path, os.path.join(dummy_test_folder_path, os.path.basename(file_path)))


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
        pylab.savefig(os.path.join(OUTPUT_FOLDER_PATH, "{}_loss_curve.png".format(MODEL_NAME)))
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
        pylab.savefig(os.path.join(OUTPUT_FOLDER_PATH, "{}_accuracy_curve.png".format(MODEL_NAME)))
        pylab.close()

def ensemble_predictions(submission_folder_path):
    def _ensemble_predictions(ensemble_func, ensemble_submission_file_name):
        ensemble_proba = ensemble_func(proba_array, axis=0)
        ensemble_proba = ensemble_proba / np.sum(ensemble_proba, axis=1)[:, np.newaxis]
        ensemble_submission_file_content.loc[:, proba_columns] = ensemble_proba
        ensemble_submission_file_content.to_csv(os.path.abspath(os.path.join(submission_folder_path, os.pardir, ensemble_submission_file_name)), index=False)

    # Read predictions
    submission_file_path_list = glob.glob(os.path.join(submission_folder_path, "*_Trial_*.csv"))
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

    print("Getting the labels ...")
    unique_label_list = sorted([folder_name for folder_name in os.listdir(TRAIN_FOLDER_PATH) if os.path.isdir(os.path.join(TRAIN_FOLDER_PATH, folder_name))])

    print("Reorganizing dataset ...")
    # Get list of files
    original_image_path_list = sorted(glob.glob(os.path.join(TRAIN_FOLDER_PATH, "*/*.jpg")), key=numericalSort)

    print("Reformatting testing dataset ...")
    reformat_testing_dataset()

    print("Initializing model ...")
    model = init_model(image_height=IMAGE_HEIGHT, image_width=IMAGE_WIDTH, unique_label_num=len(unique_label_list))


    # Perform Cross Validation
    cv_object = StratifiedKFold(n_splits=NUM_FOLD)
    num_image = len(original_image_path_list)
    y = label_index(original_image_path_list)

    for cv_index, (train_index_array, valid_index_array) in enumerate(cv_object.split(np.zeros(num_image), y), start=0):
        print("Working on splitting fold {} ...".format(cv_index))
        train_sample_num, valid_sample_num = reorganize_dataset(original_image_path_list, train_index_array, valid_index_array)

        submission_file_path = os.path.join(SUBMISSION_FOLDER_PATH, "submission_{}.csv".format(cv_index))
        if os.path.isfile(submission_file_path):
            print("The submission file already exists.")
            continue

        print("Performing the training procedure ...")
        train_generator = load_dataset(ACTUAL_TRAIN_FOLDER_PATH, classes=unique_label_list, class_mode="categorical", shuffle=True, seed=SEED)
        valid_generator = load_dataset(ACTUAL_VALID_FOLDER_PATH, classes=unique_label_list, class_mode="categorical", shuffle=True, seed=SEED)
        earlystopping_callback = EarlyStopping(monitor="val_loss", patience=PATIENCE)
        modelcheckpoint_callback = ModelCheckpoint(OPTIMAL_WEIGHTS_FILE_RULE, monitor="val_loss", save_best_only=True, save_weights_only=True)
        inspectlossaccuracy_callback = InspectLossAccuracy()
        model.fit_generator(generator=train_generator,
<<<<<<< Updated upstream
                            samples_per_epoch=train_sample_num,
                            validation_data=valid_generator,
                            nb_val_samples=valid_sample_num,
                            callbacks=[earlystopping_callback, modelcheckpoint_callback, inspectlossaccuracy_callback],
                            nb_epoch=MAXIMUM_EPOCH_NUM, verbose=2)
    else:
=======
                        samples_per_epoch=train_sample_num,
                        validation_data=valid_generator,
                        nb_val_samples=valid_sample_num,
                        callbacks=[earlystopping_callback, modelcheckpoint_callback, inspectlossaccuracy_callback],
                        nb_epoch=MAXIMUM_EPOCH_NUM, verbose=2)
        model.load_weights(OPTIMAL_WEIGHTS_FOLDER_PATH)

>>>>>>> Stashed changes
        print("Performing the testing procedure ...")
        for trial_index in np.arange(TRIAL_NUM) + 1:
            print("Working on trial {}/{} ...".format(trial_index, TRIAL_NUM))
            submission_file_path = os.path.join(SUBMISSION_FOLDER_PATH, "{}_Trial_{}.csv".format(MODEL_NAME, trial_index))
            if not os.path.isfile(submission_file_path):
                print("Performing the testing procedure ...")
                test_generator = load_dataset(TEST_FOLDER_PATH, shuffle=False, seed=trial_index)
                prediction_array = model.predict_generator(generator=test_generator, val_samples=len(test_generator.filenames))
                image_name_array = np.expand_dims([os.path.basename(image_path) for image_path in test_generator.filenames], axis=-1)
                index_array_for_sorting = np.argsort(image_name_array, axis=0)
                submission_file_content = pd.DataFrame(np.hstack((image_name_array, prediction_array))[index_array_for_sorting.flat])
                submission_file_content.to_csv(submission_file_path, header=["image_name"] + unique_label_list, index=False)

        print("Performing ensembling ...")
        ensemble_predictions(SUBMISSION_FOLDER_PATH)

        print("All done!")

if __name__ == "__main__":
    run()
