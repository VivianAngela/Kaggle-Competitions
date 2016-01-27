import common
import cv2

def retrieve_facial_image_by_bbox(full_image_path, force_continue=True):
    """Retrieve the facial image by using bbox coordinates.
    
    :param full_image_path: the path of the full image
    :type full_image_path: string
    :param force_continue: unused argument, for consistency with other functions
    :type force_continue: boolean
    :rtype: numpy array, the facial image
    """

    try:
        # Read the coordinates of facial image from the bbox file
        bbox_file_path = full_image_path + common.BBOX_EXTENSION
        y, x, w, h = common.read_from_file(bbox_file_path)
        x_start = int(x)
        x_end = int(x + h)
        y_start = int(y)
        y_end = int(y + w)

        # Generate the resized facial image
        full_image = cv2.imread(full_image_path)
        facial_image = full_image[x_start:x_end, y_start:y_end, :]
        facial_image = cv2.resize(facial_image, dsize=(common.FACIAL_IMAGE_SIZE, common.FACIAL_IMAGE_SIZE))

        # Successful case
        assert facial_image is not None
        return facial_image
    except:
        # Failure case
        return None
