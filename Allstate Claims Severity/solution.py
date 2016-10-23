import os
import glob
import numpy as np
import pandas as pd

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.advanced_activations import PReLU
from keras.layers.core import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# Data Set
DATASET_FOLDER_PATH = "./"
INPUT_FOLDER_PATH = os.path.join(DATASET_FOLDER_PATH, "input")
TRAIN_FILE_PATH = os.path.join(INPUT_FOLDER_PATH, "train.csv")
TEST_FILE_PATH = os.path.join(INPUT_FOLDER_PATH, "test.csv")
SUBMISSION_FOLDER_PATH = os.path.join(DATASET_FOLDER_PATH, "submission")
ID_COLUMN_NAME = "id"
LABEL_COLUMN_NAME = "loss"

# Model Structure
BLOCK_NUM = 3
DENSE_DIM = 512
DROPOUT_RATIO = 0.5

# Training Procedure
CROSS_VALIDATION_NUM = 10
MAXIMUM_EPOCH_NUM = 1000000
EARLYSTOPPING_PATIENCE = 20
TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 1024

def load_data():
    # Read file content
    train_file_content = pd.read_csv(TRAIN_FILE_PATH)
    test_file_content = pd.read_csv(TEST_FILE_PATH)
    combined_file_content = pd.concat([train_file_content, test_file_content])
    del(train_file_content, test_file_content)

    # Seperate the feature columns
    feature_column_list = list(combined_file_content.drop([ID_COLUMN_NAME, LABEL_COLUMN_NAME], axis=1))
    categorical_feature_column_list = [feature_column for feature_column in feature_column_list if feature_column.startswith("cat")]
    continuous_feature_column_list = [feature_column for feature_column in feature_column_list if feature_column.startswith("cont")]

    # Process categorical features
    for categorical_feature_column in categorical_feature_column_list:
        combined_file_content[categorical_feature_column] = LabelEncoder().fit_transform(combined_file_content[categorical_feature_column])
    categorical_feature_array = OneHotEncoder(dtype=np.bool, sparse=False).fit_transform(combined_file_content[categorical_feature_column_list])
    combined_file_content.drop(categorical_feature_column_list, axis=1, inplace=True)

    # Process continuous features
    combined_file_content[continuous_feature_column_list] = StandardScaler().fit_transform(combined_file_content[continuous_feature_column_list])
    continuous_feature_array = combined_file_content[continuous_feature_column_list].as_matrix()
    combined_file_content.drop(continuous_feature_column_list, axis=1, inplace=True)

    # Combine categorical and continuous features
    X_array = np.hstack((categorical_feature_array, continuous_feature_array)).astype(np.float32)
    Y_array = combined_file_content[LABEL_COLUMN_NAME].as_matrix()
    ID_array = combined_file_content[ID_COLUMN_NAME].as_matrix()
    del(categorical_feature_array, continuous_feature_array)

    # Separate the training and testing data set
    test_data_mask = np.isnan(Y_array)
    train_data_mask = np.logical_not(test_data_mask)
    X_train = X_array[train_data_mask]
    Y_train = Y_array[train_data_mask]
    X_test = X_array[test_data_mask]
    ID_test = ID_array[test_data_mask]
    submission_file_content = pd.DataFrame({ID_COLUMN_NAME:ID_test, LABEL_COLUMN_NAME:np.zeros(ID_test.shape[0])})

    return X_train, Y_train, X_test, submission_file_content

def init_model(feature_dim):
    model = Sequential()

    for block_index in range(BLOCK_NUM):
        if block_index == 0:
            model.add(Dense(DENSE_DIM, input_dim=feature_dim))
        else:
            model.add(Dense(DENSE_DIM))

        model.add(PReLU())
        model.add(BatchNormalization())
        model.add(Dropout(DROPOUT_RATIO))

    model.add(Dense(1))

    optimizer = Adam(lr=0.001, decay=1e-6)
    model.compile(loss="mean_absolute_error", optimizer=optimizer)

    return model

def ensemble_predictions():
    def _ensemble_predictions(ensemble_func, ensemble_submission_file_name):
        ensemble_proba = ensemble_func(proba_array, axis=0)
        ensemble_submission_file_content.loc[:, proba_columns] = ensemble_proba
        ensemble_submission_file_content.to_csv(os.path.join(SUBMISSION_FOLDER_PATH, ensemble_submission_file_name), index=False)

    # Read predictions
    submission_file_path_list = glob.glob(os.path.join(SUBMISSION_FOLDER_PATH, "submission_*.csv"))
    submission_file_content_list = [pd.read_csv(submission_file_path) for submission_file_path in submission_file_path_list]
    ensemble_submission_file_content = submission_file_content_list[0]

    # Concatenate predictions
    proba_columns = list(set(ensemble_submission_file_content) - {ID_COLUMN_NAME})
    proba_list = [np.expand_dims(submission_file_content.as_matrix(proba_columns), axis=0)
                  for submission_file_content in submission_file_content_list]
    proba_array = np.vstack(proba_list)

    # Ensemble predictions
    for ensemble_func, ensemble_submission_file_name in \
        zip([np.max, np.min, np.mean, np.median], ["max.csv", "min.csv", "mean.csv", "median.csv"]):
        _ensemble_predictions(ensemble_func, ensemble_submission_file_name)

def run():
    # Load data set
    X_train, Y_train, X_test, submission_file_content = load_data()

    # Initiate model
    model = init_model(X_train.shape[1])
    vanilla_weights = model.get_weights()

    # Cross validation
    cross_validation_iterator = ShuffleSplit(n_splits=CROSS_VALIDATION_NUM, test_size=0.2, random_state=0)
    for cross_validation_index, (train_index, valid_index) in enumerate(cross_validation_iterator.split(X_train), start=1):
        print("Working on {}/{} ...".format(cross_validation_index, CROSS_VALIDATION_NUM))

        optimal_weights_path = "/tmp/Optimal_Weights_{}.h5".format(cross_validation_index)
        submission_file_path = os.path.join(SUBMISSION_FOLDER_PATH, "submission_{}.csv".format(cross_validation_index))

        if os.path.isfile(submission_file_path):
            continue

        if not os.path.isfile(optimal_weights_path):
            # Load the vanilla weights
            model.set_weights(vanilla_weights)

            # Perform the training procedure
            earlystopping_callback = EarlyStopping(monitor="val_loss", patience=EARLYSTOPPING_PATIENCE)
            modelcheckpoint_callback = ModelCheckpoint(optimal_weights_path, monitor="val_loss", save_best_only=True)
            model.fit(X_train[train_index], Y_train[train_index],
                            batch_size=TRAIN_BATCH_SIZE, nb_epoch=MAXIMUM_EPOCH_NUM,
                            validation_data=(X_train[valid_index], Y_train[valid_index]),
                            callbacks=[earlystopping_callback, modelcheckpoint_callback], verbose=2)

        # Load the optimal weights
        model.load_weights(optimal_weights_path)

        # Perform the testing procedure
        Y_test = model.predict(X_test, batch_size=TEST_BATCH_SIZE, verbose=2)

        # Save submission to disk
        if not os.path.isdir(SUBMISSION_FOLDER_PATH):
            os.makedirs(SUBMISSION_FOLDER_PATH)
        submission_file_content[LABEL_COLUMN_NAME] = Y_test
        submission_file_content.to_csv(submission_file_path, index=False)

    # Perform ensembling
    ensemble_predictions()

    print("All done!")

if __name__ == "__main__":
    run()