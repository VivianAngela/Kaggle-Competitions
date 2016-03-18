import os
import prepare_data
import solution_basic
import cv2

facial_image_extension = "_open_face.jpg"
feature_extension = "_open_face.csv"

# Get image paths in the training and testing datasets
image_paths_in_training_dataset, training_image_index_list = prepare_data.get_image_paths_in_training_dataset()

# Load feature from file
training_image_feature_list = solution_basic.load_feature_from_file(image_paths_in_training_dataset, \
                                                    facial_image_extension, feature_extension)

feature_file_paths = [image_path + facial_image_extension + feature_extension \
                      for image_path in image_paths_in_training_dataset]

# Omit possible None element in training image feature list
invalid_feature_file_path_list = []
for training_image_feature, feature_file_path in zip(training_image_feature_list, feature_file_paths):
    if training_image_feature is None:
        invalid_feature_file_path_list.append(feature_file_path)

for invalid_feature_file_path in sorted(invalid_feature_file_path_list):
    found_index = invalid_feature_file_path.find(facial_image_extension)
    invalid_image_file_path = invalid_feature_file_path[0:found_index]

    invalid_image_file_name = os.path.basename(invalid_image_file_path)
    image_content = cv2.imread(invalid_image_file_path)
    print(invalid_image_file_name)

    cv2.imshow(invalid_image_file_name, image_content)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
