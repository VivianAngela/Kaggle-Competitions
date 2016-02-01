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
import time

def load_feature_from_file(image_paths, facial_image_extension, feature_extension):
    """Load feature from file.
    
    :param image_paths: the file paths of the images
    :type image_paths: list
    :param facial_image_extension: the extension of the facial images
    :type facial_image_extension: string
    :param feature_extension: the extension of the feature files
    :type feature_extension: string
    :return: the features
    :rtype: list
    """

    feature_list = []
    feature_file_paths = [image_path + facial_image_extension + feature_extension \
                          for image_path in image_paths]

    for feature_file_path in feature_file_paths:
        # Read feature directly from file
        if os.path.isfile(feature_file_path):
            feature = common.read_from_file(feature_file_path)
            feature_list.append(feature)
        else:
            feature_list.append(None)

    return feature_list

def load_feature(facial_image_extension, feature_extension):
    """Load feature.
    
    :param facial_image_extension: the extension of the facial images
    :type facial_image_extension: string
    :param feature_extension: the extension of the feature files
    :type feature_extension: string
    :return: valid_training_image_feature_list refers to the features of the training images, 
        valid_training_image_index_list refers to the indexes of the training images, 
        testing_image_feature_dict refers to the features of the testing images which is saved in a dict.
    :rtype: tuple
    """

    print("\nLoading feature ...")

    # Get image paths in the training and testing datasets
    image_paths_in_training_dataset, training_image_index_list = prepare_data.get_image_paths_in_training_dataset()
    image_paths_in_testing_dataset = prepare_data.get_image_paths_in_testing_dataset()

    # Load feature from file
    training_image_feature_list = load_feature_from_file(\
                                                         image_paths_in_training_dataset, facial_image_extension, feature_extension)
    testing_image_feature_list = load_feature_from_file(\
                                                        image_paths_in_testing_dataset, facial_image_extension, feature_extension)

    # Omit possible None element in training image feature list
    valid_training_image_feature_list = []
    valid_training_image_index_list = []
    for training_image_feature, training_image_index in zip(training_image_feature_list, training_image_index_list):
        if training_image_feature is not None:
            valid_training_image_feature_list.append(training_image_feature)
            valid_training_image_index_list.append(training_image_index)

    # Generate a dictionary to save the testing image feature
    testing_image_feature_dict = {}
    for testing_image_feature, testing_image_path in zip(testing_image_feature_list, image_paths_in_testing_dataset):
        testing_image_name = os.path.basename(testing_image_path)
        testing_image_feature_dict[testing_image_name] = testing_image_feature

    print("Feature loaded successfully.\n")
    return (valid_training_image_feature_list, valid_training_image_index_list, testing_image_feature_dict)

def get_record_map(index_array, true_false_ratio):
    """Get record map.
    
    :param index_array: the indexes of the images
    :type index_array: numpy array
    :param true_false_ratio: the number of occurrences of true cases over the number of occurrences of false cases
    :type true_false_ratio: int or float
    :return: record_index_pair_array refers to the indexes of the image pairs, 
        while record_index_pair_label_array refers to whether these two images represent the same person.
    :rtype: tuple
    """

    # Generate record_index_pair_array and record_index_pair_label_array
    record_index_pair_list = []
    record_index_pair_label_list = []
    for record_index_1, record_index_2 in itertools.combinations(range(index_array.size), 2):
        record_index_pair_list.append((record_index_1, record_index_2))
        record_index_pair_label_list.append(index_array[record_index_1] == index_array[record_index_2])
    record_index_pair_array = np.array(record_index_pair_list)
    record_index_pair_label_array = np.array(record_index_pair_label_list)

    # Do not need sampling
    if true_false_ratio is None:
        return (record_index_pair_array, record_index_pair_label_array)

    # Perform sampling based on the true_false_ratio
    pair_label_true_indexes = np.where(record_index_pair_label_array)[0]
    pair_label_false_indexes = np.where(~record_index_pair_label_array)[0]
    selected_pair_label_false_indexes = np.random.choice(\
                                                         pair_label_false_indexes, 1.0 * pair_label_true_indexes.size / true_false_ratio, replace=False)
    selected_pair_label_indexes = np.hstack((pair_label_true_indexes, selected_pair_label_false_indexes))
    return (record_index_pair_array[selected_pair_label_indexes, :], record_index_pair_label_array[selected_pair_label_indexes])

def get_final_feature(feature_1, feature_2):
    """Get the difference between two features.
    
    :param feature_1: the first feature
    :type feature_1: numpy array
    :param feature_2: the second feature
    :type feature_2: numpy array
    :return: the difference between two features
    :rtype: numpy array
    """

    if feature_1 is None or feature_2 is None:
        return None

    difference = feature_1 - feature_2
    return np.abs(difference)

def convert_to_final_data_set(image_feature_list, image_index_list, selected_indexes, true_false_ratio):
    """Convert to final data set.
    
    :param image_feature_list: the features of the images
    :type image_feature_list: list
    :param image_index_list: the indexes of the images
    :type image_index_list: list
    :param selected_indexes: the indexes of the selected records
    :type selected_indexes: numpy array
    :param true_false_ratio: the number of occurrences of true cases over the number of occurrences of false cases
    :type true_false_ratio: int or float
    :return: feature_array refers to the feature difference between two images, 
        while label_array refers to whether these two images represent the same person.
    :rtype: tuple
    """

    # Retrieve the selected records
    selected_feature_array = np.array(image_feature_list)[selected_indexes, :]
    selected_index_array = np.array(image_index_list)[selected_indexes]

    # Get record map
    pair_array, pair_label_array = get_record_map(selected_index_array, true_false_ratio)

    # Retrieve the final feature
    final_feature_list = []
    for single_pair in pair_array:
        final_feature = get_final_feature(selected_feature_array[single_pair[0], :], selected_feature_array[single_pair[1], :])
        final_feature_list.append(final_feature)

    return (np.array(final_feature_list), pair_label_array)

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
    for fold_index, fold_item in enumerate(label_kfold):
        print("\nWorking on the {:d}/{:d} fold ...".format(fold_index + 1, fold_num))

        # Generate final data set
        X_train, Y_train = convert_to_final_data_set(image_feature_list, image_index_list, fold_item[0], 1)
        X_test, Y_test = convert_to_final_data_set(image_feature_list, image_index_list, fold_item[1], None)

        # Loop through the classifiers
        for classifier_index, classifier in enumerate(classifier_list):
            classifier.fit(X_train, Y_train)
            probability_estimates = classifier.predict_proba(X_test)
            prediction = probability_estimates[:, 1]
            score = evaluation.compute_Weighted_AUC(Y_test, prediction)
            score_record[classifier_index, fold_index] = score

            print("Classifier {:d} achieved {:.4f}.".format(classifier_index, score))

    # Print the info of the best classifier
    arithmetic_mean = np.mean(score_record, axis=1)
    standard_deviation = np.std(score_record, axis=1)
    best_classifier_index = np.argmax(arithmetic_mean)
    print("\nThe classifier {:d} achieved best performance with mean score {:.4f} and standard deviation {:.4f}.".format(\
                best_classifier_index, arithmetic_mean[best_classifier_index], standard_deviation[best_classifier_index]))
    print("The optimal parameters are {}.\n".format(parameters_combinations[best_classifier_index]))

    return classifier_list[best_classifier_index]

def write_prediction(testing_file_content, prediction, prediction_file_name):
    """Write prediction file to disk.
    
    :param testing_file_content: the content in the testing file
    :type testing_file_content: numpy array
    :param prediction: the prediction
    :type prediction: numpy array
    :param prediction_file_name: the name of the prediciton file
    :type prediction_file_name: string
    :return: the prediction file will be saved to disk
    :rtype: None
    """

    prediction_file_path = os.path.join(common.SUBMISSIONS_FOLDER_PATH, prediction_file_name)
    prediction_file_content = pd.DataFrame({"Id": testing_file_content[:, 0], "Prediction": prediction})
    prediction_file_content.to_csv(prediction_file_path, index=False, header=True)

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
        final_feature = get_final_feature(file_1_feature, file_2_feature)
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
    write_prediction(testing_file_content, np.array(prediction_list), prediction_file_name)

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
        load_feature(facial_image_extension, feature_extension)

    # Find the best classifier
    classifier = get_best_classifier(training_image_feature_list, training_image_index_list)

    # Generate training data
    X_train, Y_train = convert_to_final_data_set(\
                                                 training_image_feature_list, training_image_index_list, range(len(training_image_feature_list)), 1)
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
