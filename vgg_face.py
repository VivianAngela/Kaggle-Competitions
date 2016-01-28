import caffe
import common
import cv2
import numpy as np
import os

def init_vgg_face_module():
    """Initiate the vgg face module."""

    global net

    caffe.set_mode_gpu()

    model_definition_file_path = os.path.join(common.VGG_FACE_PATH, common.MODEL_DEFINITION_FILE_NAME)
    trained_model_file_path = os.path.join(common.VGG_FACE_PATH, common.TRAINED_MODEL_FILE_NAME)
    mean_content = np.array([93.5940, 104.7624, 129.1863])

    # Initialize the network
    net = caffe.Classifier(model_definition_file_path, trained_model_file_path, \
                           image_dims=(common.VGG_FACE_IMAGE_SIZE, common.VGG_FACE_IMAGE_SIZE), \
                           mean=mean_content)

def retrieve_feature_by_vgg_face(facial_image_path, feature_file_path):
    """Retrieve the deep feature by using vgg face.
    
    :param facial_image_path: the path of the facial image
    :type facial_image_path: string
    :param feature_file_path: the path of the feature file
    :type feature_file_path: string
    :return: the deep feature
    :rtype: numpy array
    """

    try:
        # Read feature directly from file
        if os.path.isfile(feature_file_path):
            feature = common.read_from_file(feature_file_path)
            return feature

        # Retrieve feature
        assert os.path.isfile(facial_image_path)
        facial_image = cv2.imread(facial_image_path)
        facial_image = cv2.resize(facial_image, dsize=(common.VGG_FACE_IMAGE_SIZE, common.VGG_FACE_IMAGE_SIZE))
        facial_image = facial_image.astype(np.float32)
        _ = net.predict([facial_image], oversample=False).ravel()
        feature = net.blobs["fc7"].data[0]

        # Successful case. Save feature to file.
        assert feature is not None
        common.write_to_file(feature_file_path, feature)
        return feature
    except:
        # Failure case
        return None
