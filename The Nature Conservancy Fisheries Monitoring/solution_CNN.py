import os
import glob
import shutil
import numpy as np
import pandas as pd
from keras.applications.vgg16 import VGG16
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from skimage.io import imread
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder

# Dataset
DATASET_FOLDER_PATH = os.path.join(os.path.expanduser("~"), "Documents/Dataset/The Nature Conservancy Fisheries Monitoring")
TRAIN_FOLDER_PATH = os.path.join(DATASET_FOLDER_PATH, "train")
TEST_FOLDER_PATH = os.path.join(DATASET_FOLDER_PATH, "test_stg1")
RESOLUTION_RESULT_FILE_PATH = os.path.join(DATASET_FOLDER_PATH, "resolution_result.npy")

# Workspace
WORKSPACE_FOLDER_PATH = os.path.join("/tmp", os.path.basename(DATASET_FOLDER_PATH))
CLUSTERING_FOLDER_PATH = os.path.join(WORKSPACE_FOLDER_PATH, "clustering")
ACTUAL_DATASET_FOLDER_PATH = os.path.join(WORKSPACE_FOLDER_PATH, "actual_dataset")
ACTUAL_TRAIN_FOLDER_PATH = os.path.join(ACTUAL_DATASET_FOLDER_PATH, "train")
ACTUAL_VALID_FOLDER_PATH = os.path.join(ACTUAL_DATASET_FOLDER_PATH, "valid")
ACTUAL_TEST_FOLDER_PATH = os.path.join(ACTUAL_DATASET_FOLDER_PATH, "test")

# Output
OUTPUT_FOLDER_PATH = os.path.join(DATASET_FOLDER_PATH, "{}_output".format(os.path.basename(__file__).split(".")[0]))
OPTIMAL_WEIGHTS_FILE_PATH = os.path.join(OUTPUT_FOLDER_PATH, "optimal_weights.h5")
SUBMISSION_FILE_PATH = os.path.join(OUTPUT_FOLDER_PATH, "submission.csv")

# Image processing
IMAGE_ROW_SIZE = 224
IMAGE_COLUMN_SIZE = 224

# Training and Testing procedure
MAXIMUM_EPOCH_NUM = 1000000
PATIENCE = 5
BATCH_SIZE = 32

def perform_CV(image_path_list):
    if os.path.isfile(RESOLUTION_RESULT_FILE_PATH):
        print("Loading resolution result ...")
        image_name_with_image_shape_array = np.load(RESOLUTION_RESULT_FILE_PATH)
    else:
        print("Retrieving image shape ...")
        image_shape_array = np.array([imread(image_path).shape for image_path in image_path_list])

        print("Saving resolution result ...")
        image_name_with_image_shape_array = np.hstack((np.expand_dims([os.path.basename(image_path) for image_path in image_path_list], axis=-1), image_shape_array))
        np.save(RESOLUTION_RESULT_FILE_PATH, image_name_with_image_shape_array)

    print("Performing clustering ...")
    image_name_to_cluster_ID_dict = dict(zip(image_name_with_image_shape_array[:, 0],
                LabelEncoder().fit_transform([str(image_name_with_image_shape[1:]) for image_name_with_image_shape in image_name_with_image_shape_array])))
    cluster_ID_array = np.array([image_name_to_cluster_ID_dict[os.path.basename(image_path)] for image_path in image_path_list], dtype=np.int)

    print("The ID value and count are as follows:")
    cluster_ID_values, cluster_ID_counts = np.unique(cluster_ID_array, return_counts=True)
    for cluster_ID_value, cluster_ID_count in zip(cluster_ID_values, cluster_ID_counts):
        print("{}\t{}".format(cluster_ID_value, cluster_ID_count))

    print("Visualizing clustering result ...")
    shutil.rmtree(CLUSTERING_FOLDER_PATH, ignore_errors=True)
    for image_path, cluster_ID in zip(image_path_list, cluster_ID_array):
        sub_clustering_folder_path = os.path.join(CLUSTERING_FOLDER_PATH, str(cluster_ID))
        if not os.path.isdir(sub_clustering_folder_path):
            os.makedirs(sub_clustering_folder_path)
        os.symlink(image_path, os.path.join(sub_clustering_folder_path, os.path.basename(image_path)))

    cv_object = GroupShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    for train_index_array, valid_index_array in cv_object.split(X=np.zeros((len(cluster_ID_array), 1)), groups=cluster_ID_array):
        valid_sample_ratio = len(valid_index_array) / (len(train_index_array) + len(valid_index_array))
        if valid_sample_ratio > 0.15 and valid_sample_ratio < 0.25:
            return train_index_array, valid_index_array

    assert False

def load_dataset():
    # Get the labels
    unique_label_list = os.listdir(TRAIN_FOLDER_PATH)
    unique_label_list = sorted([unique_label for unique_label in unique_label_list if os.path.isdir(os.path.join(TRAIN_FOLDER_PATH, unique_label))])

    # Cross validation
    whole_train_image_path_list = glob.glob(os.path.join(TRAIN_FOLDER_PATH, "*/*.jpg"))
    whole_train_image_label_list = [image_path.split("/")[-2] for image_path in whole_train_image_path_list]
    train_index_array, valid_index_array = perform_CV(whole_train_image_path_list)

    # Create symbolic links
    shutil.rmtree(ACTUAL_DATASET_FOLDER_PATH, ignore_errors=True)
    for folder_path, index_array in zip((ACTUAL_TRAIN_FOLDER_PATH, ACTUAL_VALID_FOLDER_PATH), (train_index_array, valid_index_array)):
        for index in index_array:
            image_path = whole_train_image_path_list[index]
            image_label = whole_train_image_label_list[index]
            sub_folder_path = os.path.join(folder_path, image_label)
            if not os.path.isdir(sub_folder_path):
                os.makedirs(sub_folder_path)
            os.symlink(image_path, os.path.join(sub_folder_path, os.path.basename(image_path)))
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

    if not os.path.isdir(OUTPUT_FOLDER_PATH):
        print("Creating the output folder ...")
        os.makedirs(OUTPUT_FOLDER_PATH)

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

    if not os.path.isfile(SUBMISSION_FILE_PATH):
        print("Performing the testing procedure ...")
        model.load_weights(OPTIMAL_WEIGHTS_FILE_PATH)
        prediction_array = model.predict_generator(generator=test_generator, val_samples=len(test_generator.filenames))
        image_name_array = np.expand_dims([os.path.basename(image_path) for image_path in test_generator.filenames], axis=-1)
        index_array_for_sorting = np.argsort(image_name_array, axis=0)
        submission_file_content = pd.DataFrame(np.hstack((image_name_array, prediction_array))[index_array_for_sorting.flat])
        submission_file_content.to_csv(SUBMISSION_FILE_PATH, header=["image"] + unique_label_list, index=False)

    print("All done!")

if __name__ == "__main__":
    run()
