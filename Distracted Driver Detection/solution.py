import os
import glob
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

FOLD_NUM = 8
VALIDATION_NUM_ONE_WHOLE_EPOCH = 5
PATIENCE_NUM_WHOLE_EPOCHS = 3
TRAINING_BATCH_SIZE = 16
TESTING_BATCH_SIZE = 64
USE_CENTER_CROP = False

VANILLA_WEIGHTS_PATH = "/external/Pretrained Models/Keras/VGG19/vgg19_weights.h5"
INPUT_FOLDER_PATH = "/external/Data/Distracted Driver Detection"
TRAINING_FOLDER_PATH = os.path.join(INPUT_FOLDER_PATH, "train")
TESTING_FOLDER_PATH = os.path.join(INPUT_FOLDER_PATH, "test")
DRIVER_FILE_PATH = os.path.join(INPUT_FOLDER_PATH, "driver_imgs_list.csv")
SAMPLE_SUBMISSION_FILE_PATH = os.path.join(INPUT_FOLDER_PATH, "sample_submission.csv")
MODEL_FOLDER_PATH = os.path.join(INPUT_FOLDER_PATH, "models")
SUBMISSION_FOLDER_PATH = os.path.join(INPUT_FOLDER_PATH, "submissions")
MODEL_PREFIX = "Optimal_Weights"
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
            print("Choosing subjects {:s} as the validation data set.".format(str(np.unique(subject_array[validate_indexes]))))
            print("The training and validation data sets contain {:d} and {:d} images, respectively.".format(len(train_indexes), len(validate_indexes)))
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
    if not os.path.isfile(image_path):
        return None

    # Read the image and omit totally black images
    image = imread(image_path)
    if np.mean(image) == 0:
        return None

    # Take center crop of the image and resize it
    if USE_CENTER_CROP:
        central_pixel_coordinates = (np.array(image.shape[0:2]) / 2).astype(np.int)
        half_range_len = min(central_pixel_coordinates)
        image = image[central_pixel_coordinates[0] - half_range_len:central_pixel_coordinates[0] + half_range_len,
                      central_pixel_coordinates[1] - half_range_len:central_pixel_coordinates[1] + half_range_len, :]
    image = resize(image, (224, 224), preserve_range=True)

    # Convert to BGR color space and subtract the mean pixel
    image = image[:, :, ::-1]
    image[:, :, 0] -= 103.939
    image[:, :, 1] -= 116.779
    image[:, :, 2] -= 123.68

    # Transpose the image
    image = image.transpose((2, 0, 1))
    return image.astype("float32")

def data_generator(image_path_array, additional_info_array, infinity_loop=True, batch_size=32):
    def _data_generator(image_path_array, additional_info_array, infinity_loop):
        assert len(image_path_array) == len(additional_info_array)

        seed = 0
        while True:
            np.random.seed(seed)
            for entry_index in np.random.permutation(len(image_path_array)):
                image_path = image_path_array[entry_index]
                additional_info = additional_info_array[entry_index]
                image = preprocess_image(image_path)
                if image is not None:
                    yield (image, additional_info)
            seed += 1

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

def init_model():
    def _pop_layer(model):
        """
        Reference: https://github.com/fchollet/keras/issues/2640
        """

        if not model.outputs:
            raise Exception("Sequential model cannot be popped: model is empty.")

        model.layers.pop()
        if not model.layers:
            model.outputs = []
            model.inbound_nodes = []
            model.outbound_nodes = []
        else:
            model.layers[-1].outbound_nodes = []
            model.outputs = [model.layers[-1].output]
        model.built = False

    # Initiate a vanilla VGG19 model
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, 224, 224)))
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
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation="relu"))
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
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(4096, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation="softmax"))

    # Load vanilla weights
    if os.path.isfile(VANILLA_WEIGHTS_PATH):
        model.load_weights(VANILLA_WEIGHTS_PATH)

    # Customize the neural network structure
    _pop_layer(model)
    model.add(Dense(10, activation="softmax"))

    # Compile the neural network
    optimizer = SGD(lr=0.001, decay=0.005, momentum=0.9, nesterov=True)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def generate_prediction(selected_fold_index):
    # Create folders when necessary
    for folder_path in [MODEL_FOLDER_PATH, SUBMISSION_FOLDER_PATH]:
        if not os.path.isdir(folder_path):
            print("Creating folder {:s} ...".format(folder_path))
            os.makedirs(folder_path)

    optimal_weights_path = os.path.join(MODEL_FOLDER_PATH, "{:s}_{:s}_{:d}.h5".format(
                                            MODEL_PREFIX, str(USE_CENTER_CROP), selected_fold_index))
    submission_file_path = os.path.join(SUBMISSION_FOLDER_PATH, "{:s}_{:s}_{:d}.csv".format(
                                            SUBMISSION_PREFIX, str(USE_CENTER_CROP), selected_fold_index))
    if os.path.isfile(submission_file_path):
        print("{:s} already exists!".format(submission_file_path))
        return

    print("Splitting the training data set by using selected_fold_index {:d} ...".format(selected_fold_index))
    train_image_path_array, train_label_array, validate_image_path_array, validate_label_array = split_training_data_set(selected_fold_index)

    print("Performing conversion ...")
    categorical_train_label_array, encoder = preprocess_labels(train_label_array)
    categorical_validate_label_array, _ = preprocess_labels(validate_label_array, encoder)

    print("Initiating model ...")
    model = init_model()

    if not os.path.isfile(optimal_weights_path):
        print("Performing the training procedure ...")
        earlystopping_callback = EarlyStopping(monitor="val_loss", patience=int(VALIDATION_NUM_ONE_WHOLE_EPOCH * PATIENCE_NUM_WHOLE_EPOCHS))
        modelcheckpoint_callback = ModelCheckpoint(optimal_weights_path, monitor="val_loss", save_best_only=True)
        model.fit_generator(data_generator(train_image_path_array, categorical_train_label_array, infinity_loop=True, batch_size=TRAINING_BATCH_SIZE),
                            samples_per_epoch=int(len(train_image_path_array) / VALIDATION_NUM_ONE_WHOLE_EPOCH / TRAINING_BATCH_SIZE) * TRAINING_BATCH_SIZE,
                            validation_data=data_generator(validate_image_path_array, categorical_validate_label_array, infinity_loop=True, batch_size=TESTING_BATCH_SIZE),
                            nb_val_samples=len(validate_image_path_array),
                            callbacks=[earlystopping_callback, modelcheckpoint_callback],
                            nb_epoch=1000000, verbose=2)

    print("Performing the testing procedure ...")
    assert os.path.isfile(optimal_weights_path)
    model.load_weights(optimal_weights_path)

    # Generate the submission file
    submission_file_content = pd.read_csv(SAMPLE_SUBMISSION_FILE_PATH)
    test_image_name_array = submission_file_content["img"].as_matrix()
    test_image_path_list = [os.path.join(TESTING_FOLDER_PATH, test_image_name) for test_image_name in test_image_name_array]
    test_image_index_list = range(len(test_image_path_list))

    progress_bar = pyprind.ProgBar(np.ceil(len(test_image_path_list) / TESTING_BATCH_SIZE))
    for image_array, index_array in data_generator(test_image_path_list, test_image_index_list,
                                                   infinity_loop=False, batch_size=TESTING_BATCH_SIZE):
        proba = model.predict_proba(image_array, batch_size=TESTING_BATCH_SIZE, verbose=0)
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
    print("There are {:d} submissions in total.".format(len(submission_file_path_list)))
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
