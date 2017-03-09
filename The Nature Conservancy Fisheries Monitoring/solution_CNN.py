import os
import glob
import shutil
import numpy as np
import pandas as pd
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

# Dataset
DATASET_FOLDER_PATH = os.path.join(os.path.expanduser("~"), "Documents/Dataset/The Nature Conservancy Fisheries Monitoring")
TRAIN_FOLDER_PATH = os.path.join(DATASET_FOLDER_PATH, "train")
TEST_FOLDER_PATH = os.path.join(DATASET_FOLDER_PATH, "test_stg1")

# Output
OUTPUT_FOLDER_PATH = os.path.join(DATASET_FOLDER_PATH, "{}_output".format(os.path.basename(__file__).split(".")[0]))
OPTIMAL_WEIGHTS_FILE_PATH = os.path.join(OUTPUT_FOLDER_PATH, "optimal_weights.h5")
TRIAL_NUM = 10

# Image processing
IMAGE_ROW_SIZE = 320
IMAGE_COLUMN_SIZE = 320

# Training and Testing procedure
MAXIMUM_EPOCH_NUM = 30
BATCH_SIZE = 32

def reformat_testing_dataset():
    # Create a dummy folder
    dummy_test_folder_path = os.path.join(TEST_FOLDER_PATH, "dummy")
    os.makedirs(dummy_test_folder_path, exist_ok=True)

    # Move files to the dummy folder if needed
    file_path_list = glob.glob(os.path.join(TEST_FOLDER_PATH, "*"))
    for file_path in file_path_list:
        if os.path.isfile(file_path):
            shutil.move(file_path, os.path.join(dummy_test_folder_path, os.path.basename(file_path)))

def init_model(unique_label_num, FC_block_num=2, FC_feature_dim=512, dropout_ratio=0.5, learning_rate=0.0001):
    # Get the input tensor
    input_tensor = Input(shape=(3, IMAGE_ROW_SIZE, IMAGE_COLUMN_SIZE))

    # Convolutional blocks
    pretrained_model = VGG16(include_top=False, weights="imagenet")
    for layer in pretrained_model.layers:
        layer.trainable = False
    output_tensor = pretrained_model(input_tensor)

    # FullyConnected blocks
    output_tensor = Flatten()(output_tensor)
    for _ in range(FC_block_num):
        output_tensor = Dense(FC_feature_dim, activation="relu")(output_tensor)
        output_tensor = BatchNormalization()(output_tensor)
        output_tensor = Dropout(dropout_ratio)(output_tensor)
    output_tensor = Dense(unique_label_num, activation="softmax")(output_tensor)

    # Define and compile the model
    model = Model(input_tensor, output_tensor)
    model.compile(optimizer=Adam(lr=learning_rate), loss="categorical_crossentropy", metrics=["accuracy"])

    return model

def load_dataset(folder_path, classes=None, class_mode=None, shuffle=True, seed=None):
    # Get the generator of the dataset
    data_generator_object = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        rescale=1.0 / 255)
    data_generator = data_generator_object.flow_from_directory(
        directory=folder_path,
        target_size=(IMAGE_ROW_SIZE, IMAGE_COLUMN_SIZE),
        classes=classes,
        class_mode=class_mode,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=seed)

    return data_generator

def run():
    print("Reformatting testing dataset ...")
    reformat_testing_dataset()

    print("Creating the output folder ...")
    os.makedirs(OUTPUT_FOLDER_PATH, exist_ok=True)

    print("Getting the labels ...")
    unique_label_list = sorted([folder_name for folder_name in os.listdir(TRAIN_FOLDER_PATH) if os.path.isdir(os.path.join(TRAIN_FOLDER_PATH, folder_name))])

    print("Initializing model ...")
    model = init_model(unique_label_num=len(unique_label_list))

    if not os.path.isfile(OPTIMAL_WEIGHTS_FILE_PATH):
        print("Performing the training procedure ...")
        train_generator = load_dataset(TRAIN_FOLDER_PATH, classes=unique_label_list, class_mode="categorical", shuffle=True, seed=0)
        model.fit_generator(generator=train_generator,
                            samples_per_epoch=len(train_generator.filenames),
                            nb_epoch=MAXIMUM_EPOCH_NUM, verbose=2)

    print("Loading weights at {} ...".format(OPTIMAL_WEIGHTS_FILE_PATH))
    model.load_weights(OPTIMAL_WEIGHTS_FILE_PATH)

    for trial_index in np.arange(TRIAL_NUM) + 1:
        print("Working on trial {}/{} ...".format(trial_index, TRIAL_NUM))
        submission_file_path = os.path.join(OUTPUT_FOLDER_PATH, "Trial_{}.csv".format(trial_index))
        if not os.path.isfile(submission_file_path):
            print("Performing the testing procedure ...")
            test_generator = load_dataset(TEST_FOLDER_PATH, shuffle=False, seed=trial_index)
            prediction_array = model.predict_generator(generator=test_generator, val_samples=len(test_generator.filenames))
            image_name_array = np.expand_dims([os.path.basename(image_path) for image_path in test_generator.filenames], axis=-1)
            index_array_for_sorting = np.argsort(image_name_array, axis=0)
            submission_file_content = pd.DataFrame(np.hstack((image_name_array, prediction_array))[index_array_for_sorting.flat])
            submission_file_content.to_csv(submission_file_path, header=["image"] + unique_label_list, index=False)

    print("All done!")

if __name__ == "__main__":
    run()
