from landmark import retrieve_facial_image_by_bbox
import argparse
import common
import cv2
import openface
import os

def init_open_face_module():
    """Initiate the open face module."""

    global args
    global align
    global net

    openface_path = common.OPENFACE_PATH
    modelDir = os.path.join(openface_path, "models")
    dlibModelDir = os.path.join(modelDir, "dlib")
    openfaceModelDir = os.path.join(modelDir, "openface")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dlibFacePredictor", type=str, help="Path to dlib's face predictor.",
                        default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
    parser.add_argument("--networkModel", type=str, help="Path to Torch network model.",
                        default=os.path.join(openfaceModelDir, "nn4.small2.v1.t7"))
    parser.add_argument("--imgDim", type=int,
                        help="Default image dimension.", default=96)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    align = openface.AlignDlib(args.dlibFacePredictor)
    net = openface.TorchNeuralNet(args.networkModel, args.imgDim, cuda=True)

def retrieve_facial_image_by_open_face(full_image_path, force_continue=True):
    """Retrieve the facial image by using open face.
    
    :param full_image_path: the path of the full image
    :type full_image_path: string
    :param force_continue: whether crop facial images by using bbox coordinates
    :type force_continue: boolean
    :return: the facial image
    :rtype: numpy array
    """

    try:
        full_image_in_BGR = cv2.imread(full_image_path)
        full_image_in_RGB = cv2.cvtColor(full_image_in_BGR, cv2.COLOR_BGR2RGB)
        bounding_box = align.getLargestFaceBoundingBox(full_image_in_RGB)
        facial_image_in_RGB = align.align(common.FACIAL_IMAGE_SIZE, full_image_in_RGB, bounding_box, \
                                          landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

        # Successful case
        assert facial_image_in_RGB is not None
        return cv2.cvtColor(facial_image_in_RGB, cv2.COLOR_RGB2BGR)
    except:
        # Failure case
        if force_continue:
            return retrieve_facial_image_by_bbox(full_image_path)
        else:
            return None

def retrieve_feature_by_open_face(facial_image_path, feature_file_path):
    """Retrieve the deep feature by using open face.
    
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
        facial_image_in_BGR = cv2.imread(facial_image_path)
        facial_image_in_BGR = cv2.resize(facial_image_in_BGR, dsize=(args.imgDim, args.imgDim))
        facial_image_in_RGB = cv2.cvtColor(facial_image_in_BGR, cv2.COLOR_BGR2RGB)
        feature = net.forward(facial_image_in_RGB)

        # Successful case. Save feature to file.
        assert feature is not None
        common.write_to_file(feature_file_path, feature)
        return feature
    except:
        # Failure case
        return None
