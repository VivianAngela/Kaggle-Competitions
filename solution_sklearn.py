from sklearn.cross_validation import LabelKFold
from sklearn.grid_search import ParameterGrid
from sklearn.svm import SVC
import common
import evaluation
import itertools
import numpy as np
import os
import pandas as pd
import prepare_data
import pyprind
import solution_basic
import time

def get_best_classifier(image_feature_list, image_index_list):
    """Select among a list of classifiers and find the best one.
    
    :param image_feature_list: the features of the images
    :type image_feature_list: list
    :param image_index_list: the indexes of the images
    :type image_index_list: list
    :return: the best classifier
    :rtype: object
    """

    print("Evaluating the performance of the classifiers ...")

    # Set the parameters for SVC
    param_grid = [{"C": [1, 10, 100, 1000], "gamma": ["auto"], "kernel": ["linear"]}, \
                  {"C": [1, 10, 100, 1000], "gamma": [0.001, 0.0001], "kernel": ["rbf"]}]
    parameters_combinations = ParameterGrid(param_grid)

    # Get a list of different classifiers
    classifier_list = []
    for current_parameters in parameters_combinations:
        current_classifier = SVC(C=current_parameters["C"],
                                 kernel=current_parameters["kernel"],
                                 gamma=current_parameters["gamma"],
                                 probability=True)
        classifier_list.append(current_classifier)

    # Cross Validation
    fold_num = 5
    score_record = np.zeros((len(classifier_list), fold_num))
    label_kfold = LabelKFold(image_index_list, n_folds=fold_num)

    # Add progress bar
    progress_bar = pyprind.ProgBar(fold_num, monitor=True)

    for fold_index, fold_item in enumerate(label_kfold):
        print("\nWorking on the {:d}/{:d} fold ...".format(fold_index + 1, fold_num))

        # Generate final data set
        X_train, Y_train = solution_basic.convert_to_final_data_set(image_feature_list, image_index_list, fold_item[0], 1)
        X_test, Y_test = solution_basic.convert_to_final_data_set(image_feature_list, image_index_list, fold_item[1], None)

        # Loop through the classifiers
        for classifier_index, classifier in enumerate(classifier_list):
            classifier.fit(X_train, Y_train)
            probability_estimates = classifier.predict_proba(X_test)
            prediction = probability_estimates[:, 1]
            score = evaluation.compute_Weighted_AUC(Y_test, prediction)
            score_record[classifier_index, fold_index] = score

            print("Classifier {:d} achieved {:.4f}.".format(classifier_index, score))

        # Update progress bar
        progress_bar.update()

    # Report tracking information
    print(progress_bar)

    # Print the info of the best classifier
    arithmetic_mean = np.mean(score_record, axis=1)
    standard_deviation = np.std(score_record, axis=1)
    best_classifier_index = np.argmax(arithmetic_mean)
    print("\nThe classifier {:d} achieved best performance with mean score {:.4f} and standard deviation {:.4f}.".format(\
                best_classifier_index, arithmetic_mean[best_classifier_index], standard_deviation[best_classifier_index]))
    print("The optimal parameters are {}.\n".format(parameters_combinations[best_classifier_index]))

    return classifier_list[best_classifier_index]

def generate_prediction(classifier, testing_file_content, testing_image_feature_dict, prediction_file_prefix):
    """Generate prediction.
    
    :param classifier: the classifier
    :type classifier: object
    :param testing_file_content: the content in the testing file
    :type testing_file_content: numpy array
    :param testing_image_feature_dict: the features of the testing images which is saved in a dict
    :type testing_image_feature_dict: dict
    :param prediction_file_prefix: the prefix of the prediction file
    :type prediction_file_prefix: string
    :return: the prediction file will be saved to disk
    :rtype: None
    """

    print("\nGenerating prediction ...")

    # Add progress bar
    progress_bar = pyprind.ProgBar(testing_file_content.shape[0], monitor=True)

    # Generate prediction
    prediction_list = []
    for _, file_1_name, file_2_name in testing_file_content:
        file_1_feature = testing_image_feature_dict[file_1_name]
        file_2_feature = testing_image_feature_dict[file_2_name]
        final_feature = solution_basic.get_final_feature(file_1_feature, file_2_feature)
        final_feature = final_feature.reshape(1, -1)

        probability_estimates = classifier.predict_proba(final_feature)
        prediction = probability_estimates[0, 1]
        prediction_list.append(prediction)

        # Update progress bar
        progress_bar.update()

    # Report tracking information
    print(progress_bar)

    # Write prediction
    prediction_file_name = prediction_file_prefix + str(int(time.time())) + ".csv"
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

    # Find the best classifier
    classifier = get_best_classifier(training_image_feature_list, training_image_index_list)

    # Generate training data
    X_train, Y_train = solution_basic.convert_to_final_data_set(training_image_feature_list, \
                                                training_image_index_list, range(len(training_image_feature_list)), 1)
    classifier.fit(X_train, Y_train)

    # Load testing file
    testing_file_path = os.path.join(common.DATA_PATH, common.TESTING_FILE_NAME)
    testing_file_content = pd.read_csv(testing_file_path, delimiter=",", engine="c", \
                                       skiprows=0, na_filter=False, low_memory=False).as_matrix()

    # Generate prediction
    prediction_file_prefix = "prediction_" + selected_facial_image + "_" + selected_feature + "_sklearn_"
    generate_prediction(classifier, testing_file_content, testing_image_feature_dict, prediction_file_prefix)

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
