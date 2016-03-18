from sklearn.metrics.pairwise import pairwise_distances
import common
import itertools
import numpy as np
import os
import pandas as pd
import prepare_data

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
    training_image_feature_list = load_feature_from_file(image_paths_in_training_dataset, \
                                                        facial_image_extension, feature_extension)
    testing_image_feature_list = load_feature_from_file(image_paths_in_testing_dataset, \
                                                        facial_image_extension, feature_extension)

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
    selected_pair_label_false_indexes = np.random.choice(pair_label_false_indexes, \
                                                         1.0 * pair_label_true_indexes.size / true_false_ratio, \
                                                         replace=False)
    selected_pair_label_indexes = np.hstack((pair_label_true_indexes, selected_pair_label_false_indexes))
    return (record_index_pair_array[selected_pair_label_indexes, :], record_index_pair_label_array[selected_pair_label_indexes])

def get_final_feature(feature_1, feature_2, metric_list):
    """Get the difference between two features.
    
    :param feature_1: the first feature
    :type feature_1: numpy array
    :param feature_2: the second feature
    :type feature_2: numpy array
    :param metric_list: the metrics which will be used to compare two feature vectors
    :type metric_list: list
    :return: the difference between two features
    :rtype: numpy array
    """

    if feature_1 is None or feature_2 is None:
        return None

    if metric_list is None:
        return np.abs(feature_1 - feature_2)

    final_feature_list = []
    for metric in metric_list:
        distance_matrix = pairwise_distances(np.vstack((feature_1, feature_2)), metric=metric)
        final_feature_list.append(distance_matrix[0, 1])

    return np.array(final_feature_list)

def convert_to_final_data_set(image_feature_list, image_index_list, selected_indexes, true_false_ratio, metric_list):
    """Convert to final data set.
    
    :param image_feature_list: the features of the images
    :type image_feature_list: list
    :param image_index_list: the indexes of the images
    :type image_index_list: list
    :param selected_indexes: the indexes of the selected records
    :type selected_indexes: numpy array
    :param true_false_ratio: the number of occurrences of true cases over the number of occurrences of false cases
    :type true_false_ratio: int or float
    :param metric_list: the metrics which will be used to compare two feature vectors
    :type metric_list: list
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
        final_feature = get_final_feature(\
                                          selected_feature_array[single_pair[0], :], \
                                          selected_feature_array[single_pair[1], :], \
                                          metric_list)
        final_feature_list.append(final_feature)

    return (np.array(final_feature_list), pair_label_array)

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
