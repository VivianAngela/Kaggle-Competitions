import os
import glob
import shutil
import dlib
import json
import multiprocessing
import numpy as np
from scipy.misc import imread
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from xml.dom.minidom import parseString
from xml.etree.ElementTree import Element, SubElement, tostring

# Dataset
DATASET_FOLDER_PATH = os.path.join(os.path.expanduser("~"), "Documents/Dataset/The Nature Conservancy Fisheries Monitoring")
TRAIN_FOLDER_PATH = os.path.join(DATASET_FOLDER_PATH, "train")
TEST_FOLDER_PATH = os.path.join(DATASET_FOLDER_PATH, "test_stg1")
ANNOTATION_FOLDER_PATH = os.path.join(DATASET_FOLDER_PATH, "annotations")
RESOLUTION_RESULT_FILE_PATH = os.path.join(DATASET_FOLDER_PATH, "resolution_result.npy")

# Workspace
WORKSPACE_FOLDER_PATH = os.path.join("/tmp", os.path.basename(DATASET_FOLDER_PATH))
CLUSTERING_FOLDER_PATH = os.path.join(WORKSPACE_FOLDER_PATH, "clustering")
ACTUAL_DATASET_FOLDER_PATH = os.path.join(WORKSPACE_FOLDER_PATH, "actual_dataset")
ACTUAL_TRAIN_FOLDER_PATH = os.path.join(ACTUAL_DATASET_FOLDER_PATH, "train")
ACTUAL_VALID_FOLDER_PATH = os.path.join(ACTUAL_DATASET_FOLDER_PATH, "valid")
ACTUAL_TEST_FOLDER_PATH = os.path.join(ACTUAL_DATASET_FOLDER_PATH, "test")
ACTUAL_ANNOTATION_FILE_NAME = "annotation.xml"

# Output
OUTPUT_FOLDER_PATH = os.path.join(DATASET_FOLDER_PATH, "{}_output".format(os.path.basename(__file__).split(".")[0]))
OPTIMAL_WEIGHTS_FILE_PATH = os.path.join(OUTPUT_FOLDER_PATH, "detector.svm")

def perform_CV(image_path_list):
    if os.path.isfile(RESOLUTION_RESULT_FILE_PATH):
        print("Loading resolution result ...")
        image_name_with_image_shape_array = np.load(RESOLUTION_RESULT_FILE_PATH)
    else:
        print("Retrieving image shape ...")
        image_shape_array = np.array([imread(image_path).shape for image_path in image_path_list])

        print("Saving resolution result ...")
        image_name_with_image_shape_array = np.hstack((np.expand_dims([os.path.basename(image_path) for image_path in image_path_list], axis=-1), image_shape_array))
        np.save(RESOLUTION_RESULT_FILE_PATH, image_name_with_image_shape_array)

    print("Performing clustering ...")
    image_name_to_cluster_ID_dict = dict(zip(image_name_with_image_shape_array[:, 0],
                LabelEncoder().fit_transform([str(image_name_with_image_shape[1:]) for image_name_with_image_shape in image_name_with_image_shape_array])))
    cluster_ID_array = np.array([image_name_to_cluster_ID_dict[os.path.basename(image_path)] for image_path in image_path_list], dtype=np.int)

    print("The ID value and count are as follows:")
    cluster_ID_values, cluster_ID_counts = np.unique(cluster_ID_array, return_counts=True)
    for cluster_ID_value, cluster_ID_count in zip(cluster_ID_values, cluster_ID_counts):
        print("{}\t{}".format(cluster_ID_value, cluster_ID_count))

    print("Visualizing clustering result ...")
    shutil.rmtree(CLUSTERING_FOLDER_PATH, ignore_errors=True)
    for image_path, cluster_ID in zip(image_path_list, cluster_ID_array):
        sub_clustering_folder_path = os.path.join(CLUSTERING_FOLDER_PATH, str(cluster_ID))
        if not os.path.isdir(sub_clustering_folder_path):
            os.makedirs(sub_clustering_folder_path)
        os.symlink(image_path, os.path.join(sub_clustering_folder_path, os.path.basename(image_path)))

    cv_object = GroupShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    for train_index_array, valid_index_array in cv_object.split(X=np.zeros((len(cluster_ID_array), 1)), groups=cluster_ID_array):
        valid_sample_ratio = len(valid_index_array) / (len(train_index_array) + len(valid_index_array))
        if valid_sample_ratio > 0.15 and valid_sample_ratio < 0.25:
            return train_index_array, valid_index_array

    assert False

def initiate_dataset():
    # Get the labels
    unique_label_list = os.listdir(TRAIN_FOLDER_PATH)
    unique_label_list = sorted([unique_label for unique_label in unique_label_list if os.path.isdir(os.path.join(TRAIN_FOLDER_PATH, unique_label))])

    # Cross validation
    whole_train_image_path_list = glob.glob(os.path.join(TRAIN_FOLDER_PATH, "*/*.jpg"))
    whole_train_image_label_list = [image_path.split("/")[-2] for image_path in whole_train_image_path_list]
    train_index_array, valid_index_array = perform_CV(whole_train_image_path_list)

    # Create symbolic links
    shutil.rmtree(ACTUAL_DATASET_FOLDER_PATH, ignore_errors=True)
    for folder_path, index_array in zip((ACTUAL_TRAIN_FOLDER_PATH, ACTUAL_VALID_FOLDER_PATH), (train_index_array, valid_index_array)):
        for index in index_array:
            image_path = whole_train_image_path_list[index]
            image_label = whole_train_image_label_list[index]
            sub_folder_path = os.path.join(folder_path, image_label)
            if not os.path.isdir(sub_folder_path):
                os.makedirs(sub_folder_path)
            os.symlink(image_path, os.path.join(sub_folder_path, os.path.basename(image_path)))
    os.makedirs(ACTUAL_TEST_FOLDER_PATH)
    os.symlink(TEST_FOLDER_PATH, os.path.join(ACTUAL_TEST_FOLDER_PATH, "dummy"))

def initiate_annotation():
    # Load resolution result
    assert os.path.isfile(RESOLUTION_RESULT_FILE_PATH)
    image_name_with_image_shape_array = np.load(RESOLUTION_RESULT_FILE_PATH)
    image_name_to_image_shape_dict = dict([(image_name_with_image_shape[0], image_name_with_image_shape[1:].astype(np.int)) for image_name_with_image_shape in image_name_with_image_shape_array])

    # Load annotation
    annotation_dict = {}
    annotation_file_path_list = glob.glob(os.path.join(ANNOTATION_FOLDER_PATH, "*.json"))
    for annotation_file_path in annotation_file_path_list:
        with open(annotation_file_path) as annotation_file:
            annotation_file_content = json.load(annotation_file)
            for item in annotation_file_content:
                key = os.path.basename(item["filename"])
                value = []
                total_height, total_width, _ = image_name_to_image_shape_dict[key]
                for annotation in item["annotations"]:
                    xmiddle = (2 * annotation["x"] + annotation["width"]) / 2
                    ymiddle = (2 * annotation["y"] + annotation["height"]) / 2
                    half_max_side = max(annotation["width"], annotation["height"]) / 2
                    xmin = (xmiddle - half_max_side) / total_width
                    xmax = (xmiddle + half_max_side) / total_width
                    ymin = (ymiddle - half_max_side) / total_height
                    ymax = (ymiddle + half_max_side) / total_height
                    xmin, ymin, xmax, ymax = np.clip([xmin, ymin, xmax, ymax], 0, 1)
                    assert xmax > xmin and ymax > ymin

                    top = ymin * total_height
                    left = xmin * total_width
                    width = (xmax - xmin) * total_width
                    height = (ymax - ymin) * total_height
                    if width / height > 1.25 or height / width > 1.25:
                        continue

                    coordinate_array = np.array((top, left, width, height), dtype=np.uint32)
                    value.append(coordinate_array)
                if key in annotation_dict:
                    assert False, "Found existing key {}!!!".format(key)
                annotation_dict[key] = value

    # Save annotation file
    for folder_path in (ACTUAL_TRAIN_FOLDER_PATH, ACTUAL_VALID_FOLDER_PATH):
        dataset = Element("dataset")
        name = SubElement(dataset, "name")
        name.text = "NA"
        comment = SubElement(dataset, "comment")
        comment.text = "NA"
        images = SubElement(dataset, "images")

        image_path_list = glob.glob(os.path.join(folder_path, "*/*.jpg"))
        for image_path in image_path_list:
            image_relative_path = os.path.relpath(image_path, start=folder_path)
            coordinate_array_list = annotation_dict.get(os.path.basename(image_path), [])
            if len(coordinate_array_list) > 0:
                image = SubElement(images, "image")
                image.set("file", image_relative_path)

                for coordinate_array in coordinate_array_list:
                    top, left, width, height = coordinate_array.astype(np.str)

                    box = SubElement(image, "box")
                    box.attrib = {"top":top , "left":left, "width":width, "height":height}

        with open(os.path.join(folder_path, ACTUAL_ANNOTATION_FILE_NAME), "w") as annotation_file:
            annotation_file.write(parseString(tostring(dataset, "utf-8")).toprettyxml(indent="\t"))

def run():
    if not os.path.isdir(OUTPUT_FOLDER_PATH):
        print("Creating the output folder ...")
        os.makedirs(OUTPUT_FOLDER_PATH)

    print("Initiating dataset ...")
    initiate_dataset()

    print("Initiating annotation ...")
    initiate_annotation()

    if not os.path.isfile(OPTIMAL_WEIGHTS_FILE_PATH):
        print("Performing the training procedure ...")
        options = dlib.simple_object_detector_training_options()
        options.add_left_right_image_flips = True
        options.C = 5
        options.num_threads = multiprocessing.cpu_count() - 1
        options.be_verbose = True
        dlib.train_simple_object_detector(os.path.join(ACTUAL_TRAIN_FOLDER_PATH, ACTUAL_ANNOTATION_FILE_NAME), OPTIMAL_WEIGHTS_FILE_PATH, options)

        print("Training accuracy: {}".format(dlib.test_simple_object_detector(os.path.join(ACTUAL_TRAIN_FOLDER_PATH, ACTUAL_ANNOTATION_FILE_NAME), OPTIMAL_WEIGHTS_FILE_PATH)))
        print("Testing accuracy: {}".format(dlib.test_simple_object_detector(os.path.join(ACTUAL_TEST_FOLDER_PATH, ACTUAL_ANNOTATION_FILE_NAME), OPTIMAL_WEIGHTS_FILE_PATH)))

    print("All done!")

if __name__ == "__main__":
    run()
