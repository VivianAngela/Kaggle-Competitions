from landmark import retrieve_facial_image_by_bbox
import common
import cv2
import numpy as np
import os
import subprocess

def retrieve_facial_image_by_congealingcomplex(full_image_path, force_continue=True):
    """Retrieve the facial image by using congealingcomplex.
    
    :param full_image_path: the path of the full image
    :type full_image_path: string
    :param force_continue: whether crop facial images by using bbox coordinates
    :type force_continue: boolean
    :return: the facial image
    :rtype: numpy array
    """

    def call_congealingcomplex(facial_image):
        """Call congealingcomplex to perform face frontalization.
        
        :param facial_image: the facial image
        :type facial_image: numpy array
        :return: the processed facial image
        :rtype: numpy array
        """

        input_image_path = os.path.join("/tmp", "input_image.jpg")
        output_image_path = os.path.join("/tmp", "output_image.jpg")
        cv2.imwrite(input_image_path, facial_image)

        input_image_info_path = os.path.join("/tmp", "input_image.txt")
        output_image_info_path = os.path.join("/tmp", "output_image.txt")
        with open(input_image_info_path, "w") as text_file:
            text_file.write("{}\n".format(input_image_path))
        with open(output_image_info_path, "w") as text_file:
            text_file.write("{}\n".format(output_image_path))

        subprocess.call([os.path.join(common.CONGEALINGCOMPLEX_PATH, "funnelReal"), \
                         input_image_info_path, \
                         os.path.join(common.CONGEALINGCOMPLEX_PATH, "people.train"), \
                         output_image_info_path])

        # Read the processed facial image
        processed_facial_image = cv2.imread(output_image_path)

        # Omit the totally black rows and columns
        gray_processed_facial_image = cv2.cvtColor(processed_facial_image, cv2.COLOR_BGR2GRAY)
        cumsum_in_row = np.cumsum(gray_processed_facial_image, axis=1)
        valid_row_indexes = cumsum_in_row[:, -1] > 0
        cumsum_in_column = np.cumsum(gray_processed_facial_image, axis=0)
        valid_column_indexes = cumsum_in_column[-1, :] > 0

        return processed_facial_image[valid_row_indexes, :, :][:, valid_column_indexes, :]

    try:
        # Read the coordinates of facial image from the bbox file
        bbox_file_path = full_image_path + common.BBOX_EXTENSION
        y, x, w, h = common.read_from_file(bbox_file_path)

        # Find the middle point of the bounding rectangle
        x_middle = x + 0.5 * h
        y_middle = y + 0.5 * w

        # Make the bouding square a little bit larger
        x_start = int(x_middle - 0.8 * h)
        x_end = int(x_middle + 0.8 * h)
        y_start = int(y_middle - 0.8 * w)
        y_end = int(y_middle + 0.8 * w)

        # Retrieve the original facial image
        full_image = cv2.imread(full_image_path)
        facial_image = full_image[max(x_start, 0):min(x_end, full_image.shape[0]), max(y_start, 0):min(y_end, full_image.shape[1]), :]

        # Call congealingcomplex and resize it
        facial_image = call_congealingcomplex(facial_image)
        facial_image = cv2.resize(facial_image, dsize=(common.FACIAL_IMAGE_SIZE, common.FACIAL_IMAGE_SIZE))

        # Successful case
        assert facial_image is not None
        return facial_image
    except:
        # Failure case
        if force_continue:
            return retrieve_facial_image_by_bbox(full_image_path)
        else:
            return None
