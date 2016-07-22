import os
import glob
import h5py
import pyprind
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Flatten, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import np_utils
from skimage.io import imread
from skimage.transform import resize
from sklearn.cross_validation import LabelKFold
from sklearn.preprocessing import LabelEncoder

# Cross Validation
FOLD_NUM = 8

# Image Processing
IMAGE_SIZE = 224

# Training/Testing Procedure
FIRST_TRAINING_BATCH_SIZE = 64
FIRST_INITIAL_LEARNING_RATE = 0.001
FIRST_PATIENCE = 0
SECOND_TRAINING_BATCH_SIZE = 16
SECOND_INITIAL_LEARNING_RATE = 0.0001
SECOND_PATIENCE = 2
TESTING_BATCH_SIZE = 64
MAXIMUM_EPOCH_NUM = 1000000

# Data Set
VANILLA_WEIGHTS_PATH = "/external/Pretrained Models/Keras/VGG16/vgg16_weights.h5"
INPUT_FOLDER_PATH = "/external/Data/Distracted Driver Detection"
TRAINING_FOLDER_PATH = os.path.join(INPUT_FOLDER_PATH, "train")
TESTING_FOLDER_PATH = os.path.join(INPUT_FOLDER_PATH, "test")
DRIVER_FILE_PATH = os.path.join(INPUT_FOLDER_PATH, "driver_imgs_list.csv")
SAMPLE_SUBMISSION_FILE_PATH = os.path.join(INPUT_FOLDER_PATH, "sample_submission.csv")
MODEL_FOLDER_PATH = os.path.join(INPUT_FOLDER_PATH, "models")
SUBMISSION_FOLDER_PATH = os.path.join(INPUT_FOLDER_PATH, "submissions")
FIRST_MODEL_WEIGHTS_PREFIX = "First_Model_Weights"
SECOND_MODEL_WEIGHTS_PREFIX = "Second_Model_Weights"
SUBMISSION_PREFIX = "Aurora"

def split_training_data_set(selected_fold_index):
    # Read file content
    driver_file_content = pd.read_csv(DRIVER_FILE_PATH)
    image_path_array = np.array([os.path.join(TRAINING_FOLDER_PATH, current_row["classname"], current_row["img"])
                                 for _, current_row in driver_file_content.iterrows()])
    label_array = driver_file_content["classname"].as_matrix()
    subject_array = driver_file_content["subject"].as_matrix()

    # Split the training data set with respect to the subject
    cv_object = LabelKFold(subject_array, n_folds=FOLD_NUM)
    for fold_index, (train_indexes, validate_indexes) in enumerate(cv_object):
        if fold_index == selected_fold_index:
            print("Choosing subjects {} as the validation data set.".format(str(np.unique(subject_array[validate_indexes]))))
            print("The training and validation data sets contain {} and {} images, respectively.".format(len(train_indexes), len(validate_indexes)))
            return image_path_array[train_indexes], label_array[train_indexes], \
                image_path_array[validate_indexes], label_array[validate_indexes]

def preprocess_labels(labels, encoder=None, categorical=True):
    if not encoder:
        encoder = LabelEncoder()
        encoder.fit(labels)

    categorical_labels = encoder.transform(labels).astype(np.int32)
    if categorical:
        categorical_labels = np_utils.to_categorical(categorical_labels)
    return categorical_labels, encoder

def preprocess_image(image_path):
    try:
        # Read the image and omit totally black images
        image = imread(image_path)
        assert np.mean(image) != 0

        # Resize image with scaling
        image = resize(image, (IMAGE_SIZE, IMAGE_SIZE), preserve_range=False)

        # Convert to BGR color space
        image = image[:, :, ::-1]

        # Transpose the image
        image = image.transpose((2, 0, 1))
        return image.astype("float32")
    except:
        return None

def data_generator(image_path_array, additional_info_array,
                   infinity_loop=True, batch_size=32):
    def _data_generator(image_path_array, additional_info_array, infinity_loop):
        assert len(image_path_array) == len(additional_info_array)

        while True:
            for entry_index in np.random.permutation(len(image_path_array)):
                image_path = image_path_array[entry_index]
                additional_info = additional_info_array[entry_index]
                image = preprocess_image(image_path)
                if image is not None:
                    yield (image, additional_info)

            if not infinity_loop:
                break

    image_list = []
    additional_info_list = []
    for image, additional_info in _data_generator(image_path_array, additional_info_array, infinity_loop):
        if len(image_list) < batch_size:
            image_list.append(image)
            additional_info_list.append(additional_info)

        if len(image_list) == batch_size:
            yield (np.array(image_list), np.array(additional_info_list))
            image_list = []
            additional_info_list = []

    if len(image_list) > 0:
        yield (np.array(image_list), np.array(additional_info_list))

def init_model(unique_label_num, first_trainable_layer_index=None, learning_rate=0.001):
    # Initiate the convolutional blocks
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, IMAGE_SIZE, IMAGE_SIZE)))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Load vanilla weights of the convolutional blocks
    assert os.path.isfile(VANILLA_WEIGHTS_PATH), "Vanilla weights {} does not exist!".format(VANILLA_WEIGHTS_PATH)
    with h5py.File(VANILLA_WEIGHTS_PATH) as weights_file:
        for layer_index in range(weights_file.attrs["nb_layers"]):
            if layer_index >= len(model.layers):
                break

            layer_info = weights_file["layer_{}".format(layer_index)]
            layer_weights = [layer_info["param_{}".format(param_index)] \
                             for param_index in range(layer_info.attrs["nb_params"])]
            model.layers[layer_index].set_weights(layer_weights)

    # Initiate the customized fully-connected layers
    model.add(Flatten())
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(unique_label_num, activation="softmax"))

    # Freeze layers
    if first_trainable_layer_index is not None:
        print("Freezing layers until layer {} ...".format(first_trainable_layer_index))
        for layer in model.layers[:first_trainable_layer_index]:
            layer.trainable = False

    # List the trainable properties
    print("The trainable properties of all layers are as follows:")
    for layer in model.layers:
        print(type(layer), layer.trainable)

    # Compile the neural network
    optimizer = SGD(lr=learning_rate, momentum=0.9, decay=0.005, nesterov=True)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    return model

def generate_prediction(selected_fold_index):
    submission_file_path = os.path.join(SUBMISSION_FOLDER_PATH, "{}_{}.csv".format(
        SUBMISSION_PREFIX, selected_fold_index))
    if os.path.isfile(submission_file_path):
        print("{} already exists!".format(submission_file_path))
        return

    for folder_path in [MODEL_FOLDER_PATH, SUBMISSION_FOLDER_PATH]:
        if not os.path.isdir(folder_path):
            print("Creating folder {} ...".format(folder_path))
            os.makedirs(folder_path)

    print("Splitting the training data set by using selected_fold_index {} ...".format(selected_fold_index))
    train_image_path_array, train_label_array, \
    validate_image_path_array, validate_label_array = split_training_data_set(selected_fold_index)

    print("Performing conversion ...")
    categorical_train_label_array, encoder = preprocess_labels(train_label_array)
    categorical_validate_label_array, _ = preprocess_labels(validate_label_array, encoder)

    # The first training procedure
    first_model_weights_path = os.path.join(MODEL_FOLDER_PATH, "{}_{}.h5".format(
        FIRST_MODEL_WEIGHTS_PREFIX, selected_fold_index))
    if not os.path.isfile(first_model_weights_path):
        print("Initiating the first model ...")
        first_model = init_model(len(encoder.classes_), -6, FIRST_INITIAL_LEARNING_RATE)

        print("Performing the first training procedure ...")
        earlystopping_callback = EarlyStopping(monitor="val_loss", patience=FIRST_PATIENCE)
        modelcheckpoint_callback = ModelCheckpoint(first_model_weights_path, monitor="val_loss", save_best_only=True)
        first_model.fit_generator(data_generator(train_image_path_array, categorical_train_label_array,
                                    infinity_loop=True, batch_size=FIRST_TRAINING_BATCH_SIZE),
                                  samples_per_epoch=int(len(train_image_path_array) / FIRST_TRAINING_BATCH_SIZE) * FIRST_TRAINING_BATCH_SIZE,
                                  validation_data=data_generator(validate_image_path_array, categorical_validate_label_array,
                                    infinity_loop=True, batch_size=TESTING_BATCH_SIZE),
                                  nb_val_samples=len(validate_image_path_array),
                                  callbacks=[earlystopping_callback, modelcheckpoint_callback],
                                  nb_epoch=MAXIMUM_EPOCH_NUM, verbose=2)
    assert os.path.isfile(first_model_weights_path)

    print("Initiating the second model ...")
    second_model = init_model(len(encoder.classes_), None, SECOND_INITIAL_LEARNING_RATE)

    # The second training procedure
    second_model_weights_path = os.path.join(MODEL_FOLDER_PATH, "{}_{}.h5".format(
        SECOND_MODEL_WEIGHTS_PREFIX, selected_fold_index))
    if not os.path.isfile(second_model_weights_path):
        print("Loading the weights of the first model ...")
        second_model.load_weights(first_model_weights_path)

        print("Performing the second training procedure ...")
        earlystopping_callback = EarlyStopping(monitor="val_loss", patience=SECOND_PATIENCE)
        modelcheckpoint_callback = ModelCheckpoint(second_model_weights_path, monitor="val_loss", save_best_only=True)
        second_model.fit_generator(data_generator(train_image_path_array, categorical_train_label_array,
                                    infinity_loop=True, batch_size=SECOND_TRAINING_BATCH_SIZE),
                                   samples_per_epoch=int(len(train_image_path_array) / SECOND_TRAINING_BATCH_SIZE) * SECOND_TRAINING_BATCH_SIZE,
                                   validation_data=data_generator(validate_image_path_array, categorical_validate_label_array,
                                    infinity_loop=True, batch_size=TESTING_BATCH_SIZE),
                                   nb_val_samples=len(validate_image_path_array),
                                   callbacks=[earlystopping_callback, modelcheckpoint_callback],
                                   nb_epoch=MAXIMUM_EPOCH_NUM, verbose=2)
    assert os.path.isfile(second_model_weights_path)

    print("Loading the weights of the second model ...")
    second_model.load_weights(second_model_weights_path)

    print("Performing the testing procedure ...")
    submission_file_content = pd.read_csv(SAMPLE_SUBMISSION_FILE_PATH)
    test_image_name_array = submission_file_content["img"].as_matrix()
    test_image_path_list = [os.path.join(TESTING_FOLDER_PATH, test_image_name) for test_image_name in test_image_name_array]
    test_image_index_list = range(len(test_image_path_list))

    progress_bar = pyprind.ProgBar(np.ceil(len(test_image_path_list) / TESTING_BATCH_SIZE))
    for image_array, index_array in data_generator(test_image_path_list, test_image_index_list,
                                                   infinity_loop=False,
                                                   batch_size=TESTING_BATCH_SIZE):
        proba = second_model.predict_proba(image_array, batch_size=TESTING_BATCH_SIZE, verbose=0)
        submission_file_content.loc[index_array, encoder.classes_] = proba
        progress_bar.update()
    print(progress_bar)

    print("Writing submission to disk ...")
    submission_file_content.to_csv(submission_file_path, index=False)

def ensemble_predictions():
    def _ensemble_predictions(ensemble_func, ensemble_submission_file_name):
        ensemble_proba = ensemble_func(proba_array, axis=0)
        ensemble_proba = ensemble_proba / np.sum(ensemble_proba, axis=1)[:, np.newaxis]
        ensemble_submission_file_content.loc[:, proba_columns] = ensemble_proba
        ensemble_submission_file_content.to_csv(os.path.join(SUBMISSION_FOLDER_PATH, ensemble_submission_file_name), index=False)

    # Read predictions
    submission_file_path_list = glob.glob(os.path.join(SUBMISSION_FOLDER_PATH, SUBMISSION_PREFIX + "*.csv"))
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
    for selected_fold_index in range(FOLD_NUM):
        generate_prediction(selected_fold_index)
    ensemble_predictions()

    print("All done!")

if __name__ == "__main__":
    run()
