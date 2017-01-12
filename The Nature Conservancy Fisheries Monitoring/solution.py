import os
import glob
import shutil
import numpy as np
import pandas as pd
from itertools import product
from keras.applications.vgg16 import VGG16
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import StratifiedShuffleSplit

# Dataset
DATASET_FOLDER_PATH = os.path.join(os.path.expanduser("~"), "Documents/Dataset/The Nature Conservancy Fisheries Monitoring")
TRAIN_FOLDER_PATH = os.path.join(DATASET_FOLDER_PATH, "train")
TEST_FOLDER_PATH = os.path.join(DATASET_FOLDER_PATH, "test_stg1")
OPTIMAL_WEIGHTS_FILE_PATH = os.path.join(DATASET_FOLDER_PATH, "optimal_weights.h5")
SUBMISSION_FILE_PATH = os.path.join(DATASET_FOLDER_PATH, "submission.csv")
WORKSPACE_FOLDER_PATH = os.path.join("/tmp", os.path.basename(DATASET_FOLDER_PATH))
ACTUAL_TRAIN_FOLDER_PATH = os.path.join(WORKSPACE_FOLDER_PATH, "train")
ACTUAL_VALID_FOLDER_PATH = os.path.join(WORKSPACE_FOLDER_PATH, "valid")
ACTUAL_TEST_FOLDER_PATH = os.path.join(WORKSPACE_FOLDER_PATH, "test")

# Image processing
IMAGE_ROW_SIZE = 224
IMAGE_COLUMN_SIZE = 224

# Training and Testing procedure
MAXIMUM_EPOCH_NUM = 1000000
PATIENCE = 5
BATCH_SIZE = 32

def load_dataset():
    # Get the labels
    unique_label_list = os.listdir(TRAIN_FOLDER_PATH)
    unique_label_list = sorted([unique_label for unique_label in unique_label_list if os.path.isdir(os.path.join(TRAIN_FOLDER_PATH, unique_label))])

    # Cross validation
    whole_train_file_path_list = glob.glob(os.path.join(TRAIN_FOLDER_PATH, "*/*.jpg"))
    whole_train_file_label_list = [train_file_path.split("/")[-2] for train_file_path in whole_train_file_path_list]
    cv_object = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    for train_index_array, valid_index_array in cv_object.split(np.zeros(len(whole_train_file_label_list)), whole_train_file_label_list):
        break

    # Create symbolic links
    if os.path.isdir(WORKSPACE_FOLDER_PATH):
        shutil.rmtree(WORKSPACE_FOLDER_PATH)
    for folder_path, unique_label in product([ACTUAL_TRAIN_FOLDER_PATH, ACTUAL_VALID_FOLDER_PATH], unique_label_list):
        os.makedirs(os.path.join(folder_path, unique_label))
    for folder_path, index_array in zip((ACTUAL_TRAIN_FOLDER_PATH, ACTUAL_VALID_FOLDER_PATH), (train_index_array, valid_index_array)):
        for index in index_array:
            file_path = whole_train_file_path_list[index]
            file_label = whole_train_file_label_list[index]
            os.symlink(file_path, os.path.join(folder_path, file_label, os.path.basename(file_path)))
    os.makedirs(ACTUAL_TEST_FOLDER_PATH)
    os.symlink(TEST_FOLDER_PATH, os.path.join(ACTUAL_TEST_FOLDER_PATH, "dummy"))

    # Get the generator of the training dataset
    train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    train_generator = train_datagen.flow_from_directory(
        directory=ACTUAL_TRAIN_FOLDER_PATH,
        target_size=(IMAGE_ROW_SIZE, IMAGE_COLUMN_SIZE),
        classes=unique_label_list,
        class_mode="categorical",
        batch_size=BATCH_SIZE,
        shuffle=True)

    # Get the generator of the validation dataset
    valid_datagen = ImageDataGenerator(rescale=1. / 255)
    valid_generator = valid_datagen.flow_from_directory(
        directory=ACTUAL_VALID_FOLDER_PATH,
        target_size=(IMAGE_ROW_SIZE, IMAGE_COLUMN_SIZE),
        classes=unique_label_list,
        class_mode="categorical",
        batch_size=BATCH_SIZE,
        shuffle=False)

    # Get the generator of the testing dataset
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(
        directory=ACTUAL_TEST_FOLDER_PATH,
        target_size=(IMAGE_ROW_SIZE, IMAGE_COLUMN_SIZE),
        classes=None,
        class_mode=None,
        batch_size=BATCH_SIZE,
        shuffle=False)

    return train_generator, valid_generator, test_generator, unique_label_list

def init_model(unique_label_num, FC_block_num=2, FC_feature_dim=512, dropout_ratio=0.2, learning_rate=0.0001):
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

def run():
    print("Loading dataset ...")
    train_generator, valid_generator, test_generator, unique_label_list = load_dataset()

    print("Initializing model ...")
    model = init_model(unique_label_num=len(unique_label_list))

    if not os.path.isfile(OPTIMAL_WEIGHTS_FILE_PATH):
        print("Performing the training procedure ...")
        earlystopping_callback = EarlyStopping(monitor="val_loss", patience=PATIENCE)
        modelcheckpoint_callback = ModelCheckpoint(OPTIMAL_WEIGHTS_FILE_PATH, monitor="val_loss", save_best_only=True)
        model.fit_generator(generator=train_generator,
                            samples_per_epoch=len(train_generator.filenames),
                            validation_data=valid_generator,
                            nb_val_samples=len(valid_generator.filenames),
                            callbacks=[earlystopping_callback, modelcheckpoint_callback],
                            nb_epoch=MAXIMUM_EPOCH_NUM, verbose=2)

    print("Performing the testing procedure ...")
    model.load_weights(OPTIMAL_WEIGHTS_FILE_PATH)
    prediction_array = model.predict_generator(generator=test_generator, val_samples=len(test_generator.filenames))
    file_name_array = np.expand_dims([os.path.basename(file_path) for file_path in test_generator.filenames], axis=-1)
    index_array_for_sorting = np.argsort(file_name_array, axis=0)
    submission_file_content = pd.DataFrame(np.hstack((file_name_array, prediction_array))[index_array_for_sorting.flat])
    submission_file_content.to_csv(SUBMISSION_FILE_PATH, header=["image"] + unique_label_list, index=False)

    print("All done!")

if __name__ == "__main__":
    run()
