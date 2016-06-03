from keras.callbacks import ModelCheckpoint
from keras.layers.advanced_activations import PReLU
from keras.layers.core import Activation, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

def preprocess_data(X, scaler=None):
    if not scaler:
        scaler = StandardScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler

def preprocess_labels(labels, encoder=None, categorical=True):
    if not encoder:
        encoder = LabelEncoder()
        encoder.fit(labels)
    y = encoder.transform(labels).astype(np.int32)
    if categorical:
        y = np_utils.to_categorical(y)
    return y, encoder

def generate_prediction(X_train, Y_train, X_test, is_classification, \
                        layer_size=512, layer_num=3, dropout_ratio=0.5, \
                        batch_size=128, nb_epoch=100, validation_split=0.2):
    print("Initiate a model ...")
    model = Sequential()

    model.add(Dense(layer_size, input_shape=(X_train.shape[1],)))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(dropout_ratio))

    for _ in range(layer_num - 1):
        model.add(Dense(layer_size))
        model.add(PReLU())
        model.add(BatchNormalization())
        model.add(Dropout(dropout_ratio))

    if is_classification:
        unique_labels = np.unique(Y_train)
        model.add(Dense(unique_labels.size))
        model.add(Activation("softmax"))
        model.compile(loss="categorical_crossentropy", optimizer="adam")
    else:
        model.add(Dense(1))
        model.compile(loss="mean_absolute_error", optimizer="rmsprop")

    print("Perform normalization ...")
    X_train, scaler = preprocess_data(X_train)
    X_test, _ = preprocess_data(X_test, scaler)

    print("Perform training phase ...")
    optimal_model_file_path = "/tmp/optimal_model.hdf5"
    checkpointer = ModelCheckpoint(filepath=optimal_model_file_path, save_best_only=True)
    if is_classification:
        categorical_Y_train, encoder = preprocess_labels(Y_train)
        model.fit(X_train, categorical_Y_train, batch_size=batch_size, nb_epoch=nb_epoch, \
                  callbacks=[checkpointer], validation_split=validation_split)
    else:
        model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, \
                  callbacks=[checkpointer], validation_split=validation_split)

    print("Load optimal coefficients ...")
    model.load_weights(optimal_model_file_path)

    print("Generate prediction ...")
    if is_classification:
        classes = model.predict_classes(X_test, batch_size=batch_size)
        prediction = encoder.inverse_transform(classes)
    else:
        prediction = model.predict(X_test, batch_size=batch_size)
    return prediction
