import matplotlib
matplotlib.use("Agg")

import os
import glob
import pylab
import numpy as np
import pandas as pd
from keras.applications.vgg16 import VGG16
from keras.callbacks import Callback, ModelCheckpoint
from keras.layers import Activation, Input
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model

# Dataset
DATASET_FOLDER_PATH = os.path.join(os.path.expanduser("~"), "Documents/Dataset/The Nature Conservancy Fisheries Monitoring")
CROPPED_TRAIN_FOLDER_PATH = os.path.join(DATASET_FOLDER_PATH, "cropped_train")
CROPPED_TEST_FOLDER_PATH = os.path.join(DATASET_FOLDER_PATH, "cropped_test_stg1")

# Output
OUTPUT_FOLDER_PATH = os.path.join(DATASET_FOLDER_PATH, "{}_output".format(os.path.basename(__file__).split(".")[0]))
MODEL_FOLDER_PATH = os.path.join(OUTPUT_FOLDER_PATH, "model")
MODEL_STRUCTURE_FILE_PATH = os.path.join(MODEL_FOLDER_PATH, "model.png")
MODEL_WEIGHTS_FILE_PATH = os.path.join(MODEL_FOLDER_PATH, "model.h5")
SUBMISSION_FOLDER_PATH = os.path.join(OUTPUT_FOLDER_PATH, "submission")
LOSS_CURVE_FILE_PATH = os.path.join(OUTPUT_FOLDER_PATH, "Loss Curve.png")
TRIAL_NUM = 10

# Image processing
IMAGE_ROW_SIZE = 256
IMAGE_COLUMN_SIZE = 256

# Training and Testing procedure
MAXIMUM_EPOCH_NUM = 1000
LOSS_THRESHOLD = 0.6
BATCH_SIZE = 32

def init_model(target_num, additional_block_num=3, additional_filter_num=128, learning_rate=0.0001, freeze_pretrained_model=True):
    # Get the input tensor
    input_tensor = Input(shape=(3, IMAGE_ROW_SIZE, IMAGE_COLUMN_SIZE))

    # Convolutional blocks
    pretrained_model = VGG16(include_top=False, weights="imagenet", input_shape=input_tensor._keras_shape[1:])
    if freeze_pretrained_model:
        for layer in pretrained_model.layers:
            layer.trainable = False
    output_tensor = pretrained_model(input_tensor)

    # Additional convolutional blocks
    output_tensor = BatchNormalization(mode=0, axis=1)(output_tensor)
    for _ in range(additional_block_num):
        output_tensor = Convolution2D(additional_filter_num, 3, 3, subsample=(1, 1), activation="relu", border_mode="same")(output_tensor)
        output_tensor = BatchNormalization(mode=0, axis=1)(output_tensor)
    output_tensor = Convolution2D(target_num, 3, 3, subsample=(1, 1), activation="linear", border_mode="same")(output_tensor)
    output_tensor = GlobalAveragePooling2D()(output_tensor)
    output_tensor = Activation("softmax")(output_tensor)

    # Define and compile the model
    model = Model(input_tensor, output_tensor)
    model.compile(optimizer=Adam(lr=learning_rate), loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    plot_model(model, to_file=MODEL_STRUCTURE_FILE_PATH, show_shapes=True, show_layer_names=True)

    return model

def load_dataset(folder_path, classes=None, class_mode=None, batch_size=BATCH_SIZE, shuffle=True):
    # Get the generator of the dataset
    data_generator_object = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        zoom_range=0.2,
        horizontal_flip=True,
        rescale=1.0 / 255)
    data_generator = data_generator_object.flow_from_directory(
        directory=folder_path,
        target_size=(IMAGE_ROW_SIZE, IMAGE_COLUMN_SIZE),
        color_mode="rgb",
        classes=classes,
        class_mode=class_mode,
        batch_size=batch_size,
        shuffle=shuffle)

    return data_generator

class CustomizedStopping(Callback):
    def __init__(self):
        super(CustomizedStopping, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("loss")
        if current < LOSS_THRESHOLD:
            print("Epoch {:05d}: customized stopping".format(epoch))
            self.model.stop_training = True

class InspectLoss(Callback):
    def __init__(self):
        super(InspectLoss, self).__init__()

        self.train_loss_list = []

    def on_epoch_end(self, epoch, logs=None):
        train_loss = logs.get("loss")
        self.train_loss_list.append(train_loss)
        epoch_index_array = np.arange(len(self.train_loss_list)) + 1

        pylab.figure()
        pylab.plot(epoch_index_array, self.train_loss_list, "yellowgreen", label="train_loss")
        pylab.grid()
        pylab.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=1, ncol=1, mode="expand", borderaxespad=0.)
        pylab.savefig(LOSS_CURVE_FILE_PATH)
        pylab.close()

def ensemble_predictions(submission_folder_path=SUBMISSION_FOLDER_PATH):
    def _ensemble_predictions(ensemble_func, ensemble_submission_file_name):
        ensemble_proba = ensemble_func(proba_array, axis=0)
        ensemble_proba = ensemble_proba / np.sum(ensemble_proba, axis=1)[:, np.newaxis]
        ensemble_submission_file_content.loc[:, proba_columns] = ensemble_proba
        ensemble_submission_file_content.to_csv(os.path.join(submission_folder_path, ensemble_submission_file_name), index=False)

    # Read predictions
    submission_file_path_list = glob.glob(os.path.join(submission_folder_path, "trial_*.csv"))
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
    os.makedirs(MODEL_FOLDER_PATH, exist_ok=True)
    os.makedirs(SUBMISSION_FOLDER_PATH, exist_ok=True)

    print("Getting the labels ...")
    unique_label_list = sorted([folder_name for folder_name in os.listdir(CROPPED_TRAIN_FOLDER_PATH) if os.path.isdir(os.path.join(CROPPED_TRAIN_FOLDER_PATH, folder_name))])

    print("Initializing model ...")
    model = init_model(target_num=len(unique_label_list))

    if not os.path.isfile(MODEL_WEIGHTS_FILE_PATH):
        print("Performing the training procedure ...")
        train_generator = load_dataset(CROPPED_TRAIN_FOLDER_PATH, classes=unique_label_list, class_mode="categorical", shuffle=True)
        customizedstopping_callback = CustomizedStopping()
        modelcheckpoint_callback = ModelCheckpoint(MODEL_WEIGHTS_FILE_PATH, monitor="loss", save_best_only=True, save_weights_only=True)
        inspectloss_callback = InspectLoss()
        model.fit_generator(generator=train_generator,
                            samples_per_epoch=len(train_generator.filenames),
                            callbacks=[customizedstopping_callback, modelcheckpoint_callback, inspectloss_callback],
                            nb_epoch=MAXIMUM_EPOCH_NUM, verbose=2)

    assert os.path.isfile(MODEL_WEIGHTS_FILE_PATH)
    model.load_weights(MODEL_WEIGHTS_FILE_PATH)

    for trial_index in np.arange(TRIAL_NUM) + 1:
        print("Working on trial {}/{} ...".format(trial_index, TRIAL_NUM))
        submission_file_path = os.path.join(SUBMISSION_FOLDER_PATH, "trial_{}.csv".format(trial_index))
        if not os.path.isfile(submission_file_path):
            print("Performing the testing procedure ...")
            test_generator = load_dataset(CROPPED_TEST_FOLDER_PATH, shuffle=False)
            prediction_array = model.predict_generator(generator=test_generator, val_samples=len(test_generator.filenames))
            image_name_array = np.expand_dims([os.path.basename(image_path) for image_path in test_generator.filenames], axis=-1)
            index_array_for_sorting = np.argsort(image_name_array, axis=0)
            submission_file_content = pd.DataFrame(np.hstack((image_name_array, prediction_array))[index_array_for_sorting.flat])
            submission_file_content.to_csv(submission_file_path, header=["image"] + unique_label_list, index=False)

    print("Performing ensembling ...")
    ensemble_predictions()

    print("All done!")

if __name__ == "__main__":
    run()
