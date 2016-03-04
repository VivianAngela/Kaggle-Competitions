from sklearn.cross_validation import LabelKFold
from sklearn.externals import joblib
import common
import glob
import itertools
import numpy as np
import os
import pandas as pd
import prepare_data
import pyprind
import sklearn_related
import solution_basic
import time

METRIC_LIST_DICT = {
    "_open_face.csv":["correlation", "l1", "euclidean", "braycurtis", "sqeuclidean", \
                    "cosine", "minkowski", "l2", "canberra", "chebyshev"], \
    "_vgg_face.csv":["cosine", "sokalsneath", "dice", "braycurtis", "kulsinski", \
                    "correlation", "russellrao", "matching", "sokalmichener", "rogerstanimoto"]}

def perform_training(image_feature_list, image_index_list, description, feature_extension):
    """Perform training phase.
    
    :param image_feature_list: the features of the images
    :type image_feature_list: list
    :param image_index_list: the indexes of the images
    :type image_index_list: list
    :param description: the folder name of the working directory
    :type description: string
    :param feature_extension: the extension of the feature files
    :type feature_extension: string
    :return: the model files will be saved to disk
    :rtype: None
    """

    print("Performing training phase ...")

    # Reset the working directory
    common.reset_working_directory(description)
    working_directory = common.get_working_directory(description)

    # Cross Validation
    fold_num = 5
    best_score_array = np.zeros(fold_num)
    label_kfold = LabelKFold(image_index_list, n_folds=fold_num)

    # Add progress bar
    progress_bar = pyprind.ProgBar(fold_num, monitor=True)

    metric_list = METRIC_LIST_DICT[feature_extension]
    for fold_index, fold_item in enumerate(label_kfold):
        print("\nWorking on the {:d}/{:d} fold ...".format(fold_index + 1, fold_num))

        # Generate final data set
        X_train, Y_train = solution_basic.convert_to_final_data_set(image_feature_list, image_index_list, fold_item[0], 1, metric_list)
        X_test, Y_test = solution_basic.convert_to_final_data_set(image_feature_list, image_index_list, fold_item[1], None, metric_list)

        # Perform training
        model_name = "Model_{:d}".format(fold_index + 1) + common.SCIKIT_LEARN_EXTENSION
        model_path = os.path.join(working_directory, model_name)
        best_score = sklearn_related.train_model(X_train, Y_train, X_test, Y_test, model_path)
        best_score_array[fold_index] = best_score

        print("For the {:d} fold, the sklearn model achieved the score {:.4f}.".format(fold_index + 1, best_score))

        # Update progress bar
        progress_bar.update()

    # Report tracking information
    print(progress_bar)

    print("\nThe best score is {:.4f}.".format(np.max(best_score_array)))

def generate_prediction(description, testing_file_content, testing_image_feature_dict, prediction_file_prefix, feature_extension):
    """Generate prediction.
    
    :param description: the folder name of the working directory
    :type description: string
    :param testing_file_content: the content in the testing file
    :type testing_file_content: numpy array
    :param testing_image_feature_dict: the features of the testing images which is saved in a dict
    :type testing_image_feature_dict: dict
    :param prediction_file_prefix: the prefix of the prediction file
    :type prediction_file_prefix: string
    :param feature_extension: the extension of the feature files
    :type feature_extension: string
    :return: the prediction file will be saved to disk
    :rtype: None
    """

    print("\nGenerating prediction ...")

    working_directory = common.get_working_directory(description)
    model_path_rule = os.path.join(working_directory, "*" + common.SCIKIT_LEARN_EXTENSION)
    metric_list = METRIC_LIST_DICT[feature_extension]
    for model_path in sorted(glob.glob(model_path_rule)):
        model_name = os.path.basename(os.path.splitext(model_path)[0])
        print("\nWorking on {} ...".format(model_name))

        # Load the sklearn model
        classifier = joblib.load(model_path)

        # Add progress bar
        progress_bar = pyprind.ProgBar(testing_file_content.shape[0], monitor=True)

        # Generate prediction
        prediction_list = []
        for _, file_1_name, file_2_name in testing_file_content:
            file_1_feature = testing_image_feature_dict[file_1_name]
            file_2_feature = testing_image_feature_dict[file_2_name]
            final_feature = solution_basic.get_final_feature(file_1_feature, file_2_feature, metric_list)
            final_feature = final_feature.reshape(1, -1)

            probability_estimates = classifier.predict_proba(final_feature)
            prediction = probability_estimates[0, 1]
            prediction_list.append(prediction)

            # Update progress bar
            progress_bar.update()

        # Report tracking information
        print(progress_bar)

        # Write prediction
        prediction_file_name = prediction_file_prefix + model_name + "_" + str(int(time.time())) + ".csv"
        solution_basic.write_prediction(testing_file_content, np.array(prediction_list), prediction_file_name)

def make_prediction(facial_image_extension, feature_extension):
    """Make prediction.
    
    :param facial_image_extension: the extension of the facial images
    :type facial_image_extension: string
    :param feature_extension: the extension of the feature files
    :type feature_extension: string
    :return: the prediction file will be saved to disk
    :rtype: None
    """

    selected_facial_image = os.path.splitext(facial_image_extension)[0][1:]
    selected_feature = os.path.splitext(feature_extension)[0][1:]
    print("Making prediction by using facial image \"{}\" with feature \"{}\" ...".format(selected_facial_image, selected_feature))

    # Load feature
    training_image_feature_list, training_image_index_list, testing_image_feature_dict = \
        solution_basic.load_feature(facial_image_extension, feature_extension)

    # Perform training
    description = selected_facial_image + " with " + selected_feature + " using sklearn"
    perform_training(training_image_feature_list, training_image_index_list, description, feature_extension)

    # Load testing file
    testing_file_path = os.path.join(common.DATA_PATH, common.TESTING_FILE_NAME)
    testing_file_content = pd.read_csv(testing_file_path, delimiter=",", engine="c", \
                                       skiprows=0, na_filter=False, low_memory=False).as_matrix()

    # Generate prediction
    prediction_file_prefix = "Aurora_" + selected_facial_image + "_" + selected_feature + "_sklearn_"
    generate_prediction(description, testing_file_content, testing_image_feature_dict, prediction_file_prefix, feature_extension)

def run():
    # Crop out facial images and retrieve features. Ideally, one only need to call this function once.
    prepare_data.run()

    # Make prediction by using different features
    for facial_image_extension, feature_extension in itertools.product(\
                prepare_data.FACIAL_IMAGE_EXTENSION_LIST, prepare_data.FEATURE_EXTENSION_LIST):
        make_prediction(facial_image_extension, feature_extension)

    print("All done!")

if __name__ == "__main__":
    run()
