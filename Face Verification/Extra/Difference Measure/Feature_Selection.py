from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import pairwise_distances
import common
import itertools
import numpy as np
import os
import pandas as pd
import prepare_data
import pylab
import pyprind
import time

"""
cityblock, manhattan -> the same as l1
mahalanobis -> unable to compute. The number of observations (2) is too small;
the covariance matrix is singular. For observations with 128 dimensions, at least 129 observations are required.
yule -> nan
seuclidean -> sometimes is nan
"""

METRIC_LIST = ["cosine", "euclidean", "l1", "l2", "braycurtis", \
              "canberra", "chebyshev", "correlation", "dice", "hamming", "jaccard", "kulsinski", \
              "matching", "minkowski", "rogerstanimoto", "russellrao", \
              "sokalmichener", "sokalsneath", "sqeuclidean"]

# METRIC_LIST = ["sokalsneath", "dice"]  # "_deepfeatures.csv"
# METRIC_LIST = ["correlation", "l1"]  # "_open_face.csv"
METRIC_LIST = ["cosine", "sokalsneath"]  # "_vgg_face.csv"

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

    final_feature_list = []
    for metric in METRIC_LIST:
        try:
            distance_matrix = pairwise_distances(np.vstack((feature_1, feature_2)), metric=metric)
            # assert distance_matrix[0, 1] == distance_matrix[1, 0]
            final_feature_list.append(distance_matrix[0, 1])
        except:
            print metric

    return np.array(final_feature_list)

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

def get_feature_importances(image_feature_list, image_index_list, selected_indexes, true_false_ratio, seed):
    np.random.seed(seed)

    X_train, Y_train = convert_to_final_data_set(image_feature_list, image_index_list, selected_indexes, true_false_ratio)
    classifier = RandomForestClassifier()
    classifier.fit(X_train, Y_train)
    return classifier.feature_importances_

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
    training_image_feature_list, training_image_index_list, _ = \
        load_feature(facial_image_extension, feature_extension)

    image_feature_list = training_image_feature_list
    image_index_list = training_image_index_list
    selected_indexes = range(len(training_image_feature_list))
    true_false_ratio = 1

    repeated_num = 60
    seed_array = np.random.choice(range(repeated_num), size=repeated_num, replace=False)
    feature_importances_list = (Parallel(n_jobs=-1)(delayed(get_feature_importances)(image_feature_list, image_index_list, selected_indexes, true_false_ratio, seed) for seed in seed_array))

    feature_importances_array = np.array(feature_importances_list)
    np.save("Default.npy", feature_importances_array)

def analysis():
    feature_importances_array = np.load("Default_AlexNet.npy")
    mean_feature_importances = np.mean(feature_importances_array, axis=0)

    for feature_importance, metric in zip(mean_feature_importances, METRIC_LIST):
        print("{}\t{}".format(metric, feature_importance))

    time_indexes = np.arange(1, feature_importances_array.shape[0] + 1)
    feature_importances_cumsum = np.cumsum(feature_importances_array, axis=0)
    feature_importances_mean = feature_importances_cumsum
    for column_index in range(feature_importances_mean.shape[1]):
        feature_importances_mean[:, column_index] = feature_importances_cumsum[:, column_index] / time_indexes

    index_ranks = np.flipud(np.argsort(mean_feature_importances))

    chosen_records = np.cumsum(mean_feature_importances[index_ranks]) <= 0.95
    chosen_index_ranks = index_ranks[chosen_records]

    sorted_mean_feature_importances = mean_feature_importances[chosen_index_ranks]
    sorted_metric_list = np.array(METRIC_LIST)[chosen_index_ranks]

    remaining = np.sum(mean_feature_importances[index_ranks[~chosen_records]])
    print("remaining is {:.4f}.".format(remaining))
    sorted_mean_feature_importances = np.hstack((sorted_mean_feature_importances, remaining))
    sorted_metric_list = np.hstack((sorted_metric_list, 'others'))

    pylab.pie(sorted_mean_feature_importances, labels=sorted_metric_list, autopct='%1.1f%%', startangle=0)
    pylab.axis('equal')
    pylab.set_cmap('plasma')
    pylab.show()

def illustrate(facial_image_extension, feature_extension):
    selected_facial_image = os.path.splitext(facial_image_extension)[0][1:]
    selected_feature = os.path.splitext(feature_extension)[0][1:]
    print("Making prediction by using facial image \"{}\" with feature \"{}\" ...".format(selected_facial_image, selected_feature))

    # Load feature
    training_image_feature_list, training_image_index_list, _ = \
        load_feature(facial_image_extension, feature_extension)

    # Generate training data
    np.random.seed(0)

    X_train, Y_train = convert_to_final_data_set(training_image_feature_list, training_image_index_list, range(len(training_image_feature_list)), 1)
    true_records = Y_train == 1

    pylab.figure()
    pylab.plot(X_train[true_records, 0], X_train[true_records, 1], '.', color='yellowgreen' , label="True cases")
    pylab.plot(X_train[~true_records, 0], X_train[~true_records, 1], '.', color='lightskyblue' , label="False cases")
    pylab.xlabel(METRIC_LIST[0], fontsize="large")
    pylab.ylabel(METRIC_LIST[1], fontsize="large")
    # pylab.title("Top 2 distance metrics of AlexNet Feature")
    # pylab.title("Top 2 distance metrics of OpenFace Feature")
    pylab.title("Top 2 distance metrics of VGG Face Feature")
    pylab.legend(loc=2)
    pylab.show()

def run():
    # illustrate("_deep", "features.csv")
    # illustrate("_open_face.jpg", "_open_face.csv")
    illustrate("_bbox.jpg", "_vgg_face.csv")

    print("All done!")

if __name__ == "__main__":
    run()
