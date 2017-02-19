import os
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Input
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

# Dataset
DATASET_FOLDER_PATH = os.path.join(os.path.expanduser("~"), "Documents/Dataset/Gene Expression Prediction")
MODEL_FOLDER_PATH = os.path.join(DATASET_FOLDER_PATH, "model")
SUBMISSION_FOLDER_PATH = os.path.join(DATASET_FOLDER_PATH, "submission")

# The training/testing process
BATCH_SIZE = 32
MAX_EPOCH_NUM = 30

class CustomizedModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args , **kwargs):
        self.validation_data = kwargs.pop("validation_data", None)
        super(CustomizedModelCheckpoint, self).__init__(*args , **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        val_auc = roc_auc_score(self.validation_data[1], self.model.predict(self.validation_data[0], batch_size=BATCH_SIZE))
        logs["val_auc"] = val_auc
        super(CustomizedModelCheckpoint, self).on_epoch_end(epoch, logs)

def load_dataset():
    # Read csv files
    x_train = pd.read_csv(os.path.join(DATASET_FOLDER_PATH, "train/x_train.csv")).as_matrix()
    x_test = pd.read_csv(os.path.join(DATASET_FOLDER_PATH, "test/x_test.csv")).as_matrix()
    y_train = pd.read_csv(os.path.join(DATASET_FOLDER_PATH, "train/y_train.csv")).as_matrix()

    # Remove the first column
    x_train = x_train[:, 1:]
    x_test = x_test[:, 1:]
    y_train = y_train[:, 1:]

    # Every 100 rows correspond to one gene. Extract all 100-row-blocks into a list using np.split.
    num_genes_train = x_train.shape[0] / 100
    num_genes_test = x_test.shape[0] / 100
    x_train = np.split(x_train, num_genes_train)
    x_test = np.split(x_test, num_genes_test)

    # Reshape by raveling each 100x5 array into a 500-length vector
    x_train = [g.ravel() for g in x_train]
    x_test = [g.ravel() for g in x_test]

    # Convert data from list to array
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_train = np.ravel(y_train)

    return x_train.astype(np.float32), y_train.astype(np.float32), x_test.astype(np.float32)

def perform_scaling(x_train, x_test):
    scaler = StandardScaler()
    scaler.fit(x_train)
    return scaler.transform(x_train), scaler.transform(x_test)

def init_model(input_feature_dim, FC_block_num=3, FC_feature_dim=256, dropout_ratio=0.5, learning_rate=0.0001):
    # Get the input tensor
    input_tensor = Input(shape=(input_feature_dim,))

    # FullyConnected blocks
    output_tensor = input_tensor
    for _ in range(FC_block_num):
        output_tensor = Dense(FC_feature_dim, activation="linear")(output_tensor)
        output_tensor = LeakyReLU()(output_tensor)
        output_tensor = BatchNormalization()(output_tensor)
        output_tensor = Dropout(dropout_ratio)(output_tensor)
    output_tensor = Dense(1, activation="sigmoid")(output_tensor)

    # Define and compile the model
    model = Model(input_tensor, output_tensor)
    model.compile(optimizer=Adam(lr=learning_rate), loss="binary_crossentropy", metrics=["accuracy"])

    return model

def run():
    print("Creating folder ...")
    os.makedirs(MODEL_FOLDER_PATH, exist_ok=True)
    os.makedirs(SUBMISSION_FOLDER_PATH, exist_ok=True)

    print("Loading dataset ...")
    x_train, y_train, x_test = load_dataset()

    print("Performing scaling ...")
    x_train, x_test = perform_scaling(x_train, x_test)

    print("Initializing model ...")
    model = init_model(input_feature_dim=x_train.shape[1])
    vanilla_weights = model.get_weights()

    prediction_array_list = []
    cv_object = StratifiedShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    for cv_index, (train_index, valid_index) in enumerate(cv_object.split(x_train, y_train), start=1):
        submission_file_path = os.path.join(SUBMISSION_FOLDER_PATH, "submission_{}.csv".format(cv_index))
        if not os.path.isfile(submission_file_path):
            model_file_path = os.path.join(MODEL_FOLDER_PATH, "model_{}.h5".format(cv_index))
            if not os.path.isfile(model_file_path):
                train_data = (x_train[train_index], y_train[train_index])
                validation_data = (x_train[valid_index], y_train[valid_index])
                customizedmodelcheckpoint_callback = CustomizedModelCheckpoint(validation_data=validation_data,
                        filepath=model_file_path, monitor="val_auc", mode="max", save_best_only=True, save_weights_only=True, verbose=1)
                model.set_weights(vanilla_weights)
                model.fit(train_data[0], train_data[1], validation_data=validation_data, batch_size=BATCH_SIZE, nb_epoch=MAX_EPOCH_NUM, callbacks=[customizedmodelcheckpoint_callback], verbose=2)

            assert os.path.isfile(model_file_path)
            print("Loading weights from {}".format(os.path.basename(model_file_path)))
            model.load_weights(model_file_path)

            # Generate prediction
            prediction_array = model.predict(x_test, batch_size=BATCH_SIZE)
            prediction_array_list.append(prediction_array)

            # Save prediction to disk
            submission_file_content = pd.DataFrame(np.hstack((np.expand_dims(np.arange(len(prediction_array)), axis=-1) + 1, prediction_array)), columns=["GeneId", "Prediction"])
            submission_file_content.GeneId = submission_file_content.GeneId.astype(int)
            submission_file_content.to_csv(submission_file_path, index=False)
            print("Submission saved at {}".format(submission_file_path))
        else:
            # Load prediction
            prediction_array = np.expand_dims(pd.read_csv(submission_file_path).as_matrix()[:, 1], axis=-1)
            prediction_array_list.append(prediction_array)

    # Ensemble predictions
    for ensemble_func, ensemble_func_name in zip([np.max, np.min, np.mean, np.median], ["max", "min", "mean", "median"]):
        submission_file_path = os.path.join(SUBMISSION_FOLDER_PATH, "submission_{}.csv".format(ensemble_func_name))
        if not os.path.isfile(submission_file_path):
            prediction_array = ensemble_func(prediction_array_list, axis=0)

            # Save prediction to disk
            submission_file_content = pd.DataFrame(np.hstack((np.expand_dims(np.arange(len(prediction_array)), axis=-1) + 1, prediction_array)), columns=["GeneId", "Prediction"])
            submission_file_content.GeneId = submission_file_content.GeneId.astype(int)
            submission_file_content.to_csv(submission_file_path, index=False)
            print("Submission saved at {}".format(submission_file_path))

    print("All done!")

if __name__ == "__main__":
    run()
