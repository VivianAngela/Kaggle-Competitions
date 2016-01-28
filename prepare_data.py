import common
import cv2
import glob
import landmark
import numpy as np
import open_face
import os
import pyprind
import vgg_face

# The extensions of the facial images
FACIAL_IMAGE_EXTENSION_LIST = ["_bbox.jpg", "_open_face.jpg"]

# The file names of the mean facial images
MEAN_IMAGE_NAME_LIST = ["mean" + value for value in FACIAL_IMAGE_EXTENSION_LIST]

# The function objects that could crop faces
RETRIEVE_FACIAL_IMAGE_FUNC_LIST = [\
                                   getattr(landmark, "retrieve_facial_image_by_bbox"), \
                                   getattr(open_face, "retrieve_facial_image_by_open_face")]

# The extensions of the feature files
FEATURE_EXTENSION_LIST = ["_open_face.csv", "_vgg_face.csv"]

# The function objects that could retrieve feature
RETRIEVE_FEATURE_FUNC_LIST = [\
                              getattr(open_face, "retrieve_feature_by_open_face"), \
                              getattr(vgg_face, "retrieve_feature_by_vgg_face")]

def get_image_paths_in_training_dataset():
    """Get image paths in the training data set.
    
    :rtype: tuple, original_image_path_list refers to the image path, 
        while training_image_index_list refers to the image index.
    """

    original_image_path_list = []
    training_image_index_list = []
    training_dataset_path = os.path.join(common.DATA_PATH, common.TRAINING_DATASET_NAME)
    training_folder_name_list = os.listdir(training_dataset_path)

    for training_folder_name_index, training_folder_name in enumerate(training_folder_name_list):
        training_folder_path = os.path.join(training_dataset_path, training_folder_name)
        if not os.path.isdir(training_folder_path):
            continue

        bbox_file_path_rule = os.path.join(training_folder_path, "*" + common.BBOX_EXTENSION)
        for bbox_file_path in glob.glob(bbox_file_path_rule):
            index_end = len(bbox_file_path) - len(common.BBOX_EXTENSION)
            original_image_path = bbox_file_path[0:index_end]

            if not os.path.isfile(original_image_path):
                print("{} does not exist!!!".format(os.path.basename(original_image_path)))
                continue

            original_image_path_list.append(original_image_path)
            training_image_index_list.append(training_folder_name_index)

    return (original_image_path_list, training_image_index_list)

def get_image_paths_in_testing_dataset():
    """Get image paths in the testing data set.
    
    :rtype: list, original_image_path_list refers to the image path
    """

    original_image_path_list = []
    testing_dataset_path = os.path.join(common.DATA_PATH, common.TESTING_DATASET_NAME)

    bbox_file_path_rule = os.path.join(testing_dataset_path, "*" + common.BBOX_EXTENSION)
    for bbox_file_path in glob.glob(bbox_file_path_rule):
        index_end = len(bbox_file_path) - len(common.BBOX_EXTENSION)
        original_image_path = bbox_file_path[0:index_end]

        if not os.path.isfile(original_image_path):
            print("{} does not exist!!!".format(os.path.basename(original_image_path)))
            continue

        original_image_path_list.append(original_image_path)

    return original_image_path_list

def crop_facial_images_within_single_dataset(image_paths, facial_image_extension, \
                                             mean_image_name, retrieve_facial_image_func, force_continue):
    """Crop facial images within single dataset.
    
    :param image_paths: the file paths of the images
    :type image_paths: list
    :param facial_image_extension: the extension of the facial images
    :type facial_image_extension: string
    :param mean_image_name: the file name of the mean facial image
    :type mean_image_name: string
    :param retrieve_facial_image_func: the function object that could crop faces
    :type retrieve_facial_image_func: object
    :param force_continue: whether crop facial images by using bbox coordinates
    :type force_continue: boolean
    :rtype: None
    """

    # The sum of all images
    image_sum = np.zeros((common.FACIAL_IMAGE_SIZE, common.FACIAL_IMAGE_SIZE, 3))
    image_num = 0
    error_num = 0

    # Add progress bar
    progress_bar = pyprind.ProgBar(len(image_paths), monitor=True)

    for image_path in image_paths:
        # Update progress bar before the computation
        progress_bar.update()

        # Skip when the resized facial image file already exists
        facial_image_path = image_path + facial_image_extension
        if os.path.isfile(facial_image_path):
            continue

        # Retrieve facial image
        facial_image = retrieve_facial_image_func(image_path, force_continue)
        if facial_image is None:
            error_num = error_num + 1
            continue

        # Update the image sum
        for slice_index in range(3):
            image_sum[:, :, slice_index] += facial_image[:, :, slice_index]
        image_num = image_num + 1

        # Save the resized facial image
        cv2.imwrite(facial_image_path, facial_image)

    # Report tracking information
    print(progress_bar)

    # Report the percentage of failures
    print("Can't crop out faces from {:d}/{:d} images.".format(error_num, len(image_paths)))

    # Save the mean facial image when necessary
    if mean_image_name is not None and image_num != 0:
        mean_image = image_sum / image_num
        mean_image = mean_image.astype(np.uint8)
        mean_image_path = os.path.join(common.DATA_PATH, mean_image_name)
        if not os.path.isfile(mean_image_path):
            cv2.imwrite(mean_image_path, mean_image)
            print("Mean image saved.")

def crop_facial_images(facial_image_extension, mean_image_name, retrieve_facial_image_func):
    """Crop facial images.
    
    :param facial_image_extension: the extension of the facial images
    :type facial_image_extension: string
    :param mean_image_name: the file name of the mean facial image
    :type mean_image_name: string
    :param retrieve_facial_image_func: the function object that could crop faces
    :type retrieve_facial_image_func: object
    :rtype: None
    """

    print("\nCropping facial images with facial_image_extension is {}.".format(facial_image_extension))

    # Get image paths in the training and testing datasets
    image_paths_in_training_dataset, _ = get_image_paths_in_training_dataset()
    image_paths_in_testing_dataset = get_image_paths_in_testing_dataset()

    # Crop facial images in the training and testing datasets
    print("\nWorking on the training data set ...")
    crop_facial_images_within_single_dataset(image_paths_in_training_dataset, facial_image_extension, \
                           mean_image_name, retrieve_facial_image_func, False)

    print("\nWorking on the testing data set ...")
    crop_facial_images_within_single_dataset(image_paths_in_testing_dataset, facial_image_extension, \
                           None, retrieve_facial_image_func, True)

def compute_features(facial_image_extension, feature_extension, retrieve_feature_func):
    """Compute features.
    
    :param facial_image_extension: the extension of the facial images
    :type facial_image_extension: string
    :param feature_extension: the extension of the feature files
    :type feature_extension: string
    :param retrieve_feature_func: the function object that could retrieve feature
    :type retrieve_feature_func: object
    :rtype: None
    """

    print("\nComputing features with facial_image_extension is {} and feature_extension is {}.".format\
          (facial_image_extension, feature_extension))

    # Get image paths in the training and testing datasets
    image_paths_in_training_dataset, _ = get_image_paths_in_training_dataset()
    image_paths_in_testing_dataset = get_image_paths_in_testing_dataset()
    image_paths = image_paths_in_training_dataset + image_paths_in_testing_dataset

    facial_image_path_list = [image_path + facial_image_extension for image_path in image_paths]
    feature_file_path_list = [facial_image_path + feature_extension for facial_image_path in facial_image_path_list]

    error_num = 0

    # Add progress bar
    progress_bar = pyprind.ProgBar(len(facial_image_path_list), monitor=True)

    for facial_image_path, feature_file_path in zip(facial_image_path_list, feature_file_path_list):
        # Update progress bar before the computation
        progress_bar.update()

        # Retrieve feature
        feature = retrieve_feature_func(facial_image_path, feature_file_path)
        if feature is None:
            error_num = error_num + 1

    # Report tracking information
    print(progress_bar)

    # Report the percentage of failures
    print("Can't retrieve feature from {:d}/{:d} images.".format(error_num, len(facial_image_path_list)))

def run():
    # Initiate OpenFace Module
    open_face.init_open_face_module()

    # Initiate VGG Face Module
    vgg_face.init_vgg_face_module()

    # Generate facial images
    for facial_image_extension, mean_image_name, retrieve_facial_image_func in \
        zip(FACIAL_IMAGE_EXTENSION_LIST, MEAN_IMAGE_NAME_LIST, RETRIEVE_FACIAL_IMAGE_FUNC_LIST):
        crop_facial_images(facial_image_extension, mean_image_name, retrieve_facial_image_func)

    # Generate features
    for facial_image_extension in FACIAL_IMAGE_EXTENSION_LIST:
        for feature_extension, retrieve_feature_func in \
            zip(FEATURE_EXTENSION_LIST, RETRIEVE_FEATURE_FUNC_LIST):
            compute_features(facial_image_extension, feature_extension, retrieve_feature_func)

if __name__ == "__main__":
    run()
