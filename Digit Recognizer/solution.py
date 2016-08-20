import numpy as np
np.random.seed(666666)

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import os
import preprocessing
import time

MODEL_FOLDER_PATH = "./models"
OPTIMAL_MODEL_FILE_PATH = os.path.join(MODEL_FOLDER_PATH, "optimal_model.hdf5")
ROW_NUM = 28
COLUMN_NUM = 28
BATCH_SIZE = 128

def preprocess_images(X):
    X = np.reshape(X, (X.shape[0], ROW_NUM, COLUMN_NUM))
    X = np.expand_dims(X, axis=1)
    return X / 255

def preprocess_labels(labels, encoder=None, categorical=True):
    if not encoder:
        encoder = LabelEncoder()
        encoder.fit(labels)

    categorical_labels = encoder.transform(labels).astype(np.int32)
    if categorical:
        categorical_labels = np_utils.to_categorical(categorical_labels)
    return categorical_labels, encoder

def init_model(class_num):
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode="same",
                            input_shape=(1, ROW_NUM, COLUMN_NUM)))
    model.add(PReLU())
    model.add(Convolution2D(32, 3, 3))
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode="same"))
    model.add(PReLU())
    model.add(Convolution2D(64, 3, 3))
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(class_num))
    model.add(Activation("softmax"))

    optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer)
    return model

def run():
    print("Loading data ...")
    X_train, Y_train, X_test, submission_file_content = preprocessing.load_data()

    print("Performing conversion ...")
    X_train = preprocess_images(X_train)
    X_test = preprocess_images(X_test)
    categorical_Y_train, encoder = preprocess_labels(Y_train)

    model = init_model(np.unique(Y_train).size)
    if not os.path.isfile(OPTIMAL_MODEL_FILE_PATH):
        print("Performing the training phase ...")

        if not os.path.isdir(MODEL_FOLDER_PATH):
            os.makedirs(MODEL_FOLDER_PATH)

        earlystopping_callback = EarlyStopping(patience=1)
        modelcheckpoint_callback = ModelCheckpoint(OPTIMAL_MODEL_FILE_PATH, save_best_only=True)
        model.fit(X_train, categorical_Y_train, batch_size=BATCH_SIZE, nb_epoch=1,
                  callbacks=[earlystopping_callback, modelcheckpoint_callback],
                  validation_split=0.2, show_accuracy=True)

    print("Loading the optimal model ...")
    model.load_weights(OPTIMAL_MODEL_FILE_PATH)

    print("Generating prediction ...")
    temp_predictions = model.predict(X_test, batch_size=BATCH_SIZE)
    prediction = encoder.inverse_transform(temp_predictions)

    print("Writing prediction to disk ...")
    submission_file_name = "Aurora_{:.4f}_{:d}.csv".format(EarlyStopping.best, int(time.time()))
    submission_file_content[preprocessing.LABEL_COLUMN_NAME_IN_SUBMISSION] = prediction
    submission_file_content.to_csv(submission_file_name, index=False)

    print("All done!")

if __name__ == "__main__":
    run()
