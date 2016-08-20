from keras.callbacks import ModelCheckpoint
from keras.layers.advanced_activations import PReLU
from keras.layers.core import Dense, Dropout, Activation
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

def generate_prediction(X_train, Y_train, X_test):
    # Encode labels with value between 0 and n_classes-1.
    y, encoder = preprocess_labels(Y_train)

    print("Initiate a model ...")
    dims = X_train.shape[1]
    nb_classes = y.shape[1]

    model = Sequential()
    model.add(Dense(128, input_shape=(dims,)))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(128))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(128))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam')

    print("Perform training phase ...")
    optimal_model_file_path = "/tmp/optimal_model.h5"
    checkpointer = ModelCheckpoint(filepath=optimal_model_file_path, save_best_only=True)
    model.fit(X_train, y, nb_epoch=100, callbacks=[checkpointer], validation_split=0.2)

    print("Load optimal coefficients ...")
    model.load_weights(optimal_model_file_path)

    print("Generate prediction ...")
    classes = model.predict_classes(X_test)
    prediction = encoder.inverse_transform(classes)

    return prediction
