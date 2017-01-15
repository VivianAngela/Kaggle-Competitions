import os
import glob
import shutil
import json
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from keras.applications.imagenet_utils import preprocess_input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from SSD_related.ssd import SSD300
from SSD_related.ssd_training import MultiboxLoss
from SSD_related.ssd_utils import BBoxUtility

# Use tensorflow backend and ordering explicitly
assert K.backend() == "tensorflow" and K.image_dim_ordering() == "tf"

# Dataset
DATASET_FOLDER_PATH = os.path.join(os.path.expanduser("~"), "Documents/Dataset/The Nature Conservancy Fisheries Monitoring")
TRAIN_FOLDER_PATH = os.path.join(DATASET_FOLDER_PATH, "train")
TEST_FOLDER_PATH = os.path.join(DATASET_FOLDER_PATH, "test_stg1")
ANNOTATION_FOLDER_PATH = os.path.join(DATASET_FOLDER_PATH, "annotations")
RESOLUTION_RESULT_FILE_PATH = os.path.join(DATASET_FOLDER_PATH, "resolution_result.npy")
LABEL_WITHOUT_OBJECT = "NoF"

# SSD: Single Shot MultiBox Detector
SSD_FOLDER_PATH = os.path.join(os.path.expanduser("~"), "Documents/Dataset/SSD")
PRETRAINED_WEIGHTS_FILE_PATH = os.path.join(SSD_FOLDER_PATH, "weights_SSD300.hdf5")
DEFAULT_BOXES_FILE_PATH = os.path.join(SSD_FOLDER_PATH, "prior_boxes_ssd300.pkl")

# Workspace
WORKSPACE_FOLDER_PATH = os.path.join("/tmp", os.path.basename(DATASET_FOLDER_PATH))
CLUSTERING_FOLDER_PATH = os.path.join(WORKSPACE_FOLDER_PATH, "clustering")

# Output
OUTPUT_FOLDER_PATH = os.path.join(DATASET_FOLDER_PATH, "{}_output".format(os.path.basename(__file__).split(".")[0]))
OPTIMAL_WEIGHTS_FILE_PATH = os.path.join(OUTPUT_FOLDER_PATH, "optimal_weights.h5")
SUBMISSION_FILE_PATH = os.path.join(OUTPUT_FOLDER_PATH, "submission.csv")

# Image processing
IMAGE_ROW_SIZE = 300
IMAGE_COLUMN_SIZE = 300
IMAGE_CHANNEL_SIZE = 3

# Training and Testing procedure
MAXIMUM_EPOCH_NUM = 1000000
PATIENCE = 5
BATCH_SIZE = 32

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

class DataGenerator(object):
    def __init__(self, gt, bbox_util,
                 image_size,
                 train_image_path_list, valid_image_path_list, test_image_path_list,
                 saturation_var=0.5,
                 brightness_var=0.5,
                 contrast_var=0.5,
                 lighting_std=0.5,
                 hflip_prob=0.5,
                 vflip_prob=0.5,
                 do_crop=False,
                 crop_area_range=[0.75, 1.0],
                 aspect_ratio_range=[3. / 4., 4. / 3.]):
        self.gt = gt
        self.bbox_util = bbox_util
        self.image_size = image_size
        self.train_image_path_list = train_image_path_list
        self.valid_image_path_list = valid_image_path_list
        self.test_image_path_list = test_image_path_list
        self.color_jitter = []
        if saturation_var:
            self.saturation_var = saturation_var
            self.color_jitter.append(self.saturation)
        if brightness_var:
            self.brightness_var = brightness_var
            self.color_jitter.append(self.brightness)
        if contrast_var:
            self.contrast_var = contrast_var
            self.color_jitter.append(self.contrast)
        self.lighting_std = lighting_std
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.do_crop = do_crop
        self.crop_area_range = crop_area_range
        self.aspect_ratio_range = aspect_ratio_range

    def grayscale(self, rgb):
        return rgb.dot([0.299, 0.587, 0.114])

    def saturation(self, rgb):
        gs = self.grayscale(rgb)
        alpha = 2 * np.random.random() * self.saturation_var
        alpha += 1 - self.saturation_var
        rgb = rgb * alpha + (1 - alpha) * gs[:, :, None]
        return np.clip(rgb, 0, 255)

    def brightness(self, rgb):
        alpha = 2 * np.random.random() * self.brightness_var
        alpha += 1 - self.saturation_var
        rgb = rgb * alpha
        return np.clip(rgb, 0, 255)

    def contrast(self, rgb):
        gs = self.grayscale(rgb).mean() * np.ones_like(rgb)
        alpha = 2 * np.random.random() * self.contrast_var
        alpha += 1 - self.contrast_var
        rgb = rgb * alpha + (1 - alpha) * gs
        return np.clip(rgb, 0, 255)

    def lighting(self, img):
        cov = np.cov(img.reshape(-1, 3) / 255.0, rowvar=False)
        eigval, eigvec = np.linalg.eig(cov)
        noise = np.random.randn(3) * self.lighting_std
        noise = eigvec.dot(eigval * noise) * 255
        img += noise
        return np.clip(img, 0, 255)

    def horizontal_flip(self, img, y):
        if np.random.random() < self.hflip_prob:
            img = img[:, ::-1]
            y[:, [0, 2]] = 1 - y[:, [2, 0]]
        return img, y

    def vertical_flip(self, img, y):
        if np.random.random() < self.vflip_prob:
            img = img[::-1]
            y[:, [1, 3]] = 1 - y[:, [3, 1]]
        return img, y

    def random_sized_crop(self, img, targets):
        img_w = img.shape[1]
        img_h = img.shape[0]
        img_area = img_w * img_h
        for _ in range(self.crop_attempts):
            random_scale = np.random.random()
            random_scale *= (self.crop_area_range[1] -
                             self.crop_area_range[0])
            random_scale += self.crop_area_range[0]
            target_area = random_scale * img_area
            random_ratio = np.random.random()
            random_ratio *= (self.aspect_ratio_range[1] -
                             self.aspect_ratio_range[0])
            random_ratio += self.aspect_ratio_range[0]
            w = np.round(np.sqrt(target_area * random_ratio))
            h = np.round(np.sqrt(target_area / random_ratio))
            if np.random.random() < 0.5:
                w, h = h, w
            w = min(w, img_w)
            w_rel = w / img_w
            w = int(w)
            h = min(h, img_w)
            h_rel = h / img_h
            h = int(h)
            x = np.random.random() * (img_w - w)
            x_rel = x / img_w
            x = int(x)
            y = np.random.random() * (img_h - h)
            y_rel = y / img_h
            y = int(y)
            img = img[y:y + h, x:x + w]
            new_targets = []
            for box in targets:
                cx = 0.5 * (box[0] + box[2])
                cy = 0.5 * (box[1] + box[3])
                if (x_rel < cx < x_rel + w_rel and
                    y_rel < cy < y_rel + h_rel):
                    xmin = (box[0] - x) / w_rel
                    ymin = (box[1] - y) / h_rel
                    xmax = (box[2] - x) / w_rel
                    ymax = (box[3] - y) / h_rel
                    xmin = max(0, xmin)
                    ymin = max(0, ymin)
                    xmax = min(1, xmax)
                    ymax = min(1, ymax)
                    box[:4] = [xmin, ymin, xmax, ymax]
                    new_targets.append(box)
            new_targets = np.asarray(new_targets).reshape(-1, targets.shape[1])
            return img, new_targets

    def generate(self, dataset_name="train", batch_size=BATCH_SIZE):
        np.random.seed(0)
        while True:
            if dataset_name == "train":
                image_path_list = np.random.permutation(self.train_image_path_list)
            elif dataset_name == "valid":
                image_path_list = self.valid_image_path_list
            elif dataset_name == "test":
                image_path_list = self.test_image_path_list
            else:
                assert False
            inputs = []
            targets = []
            for (image_index, image_path) in enumerate(image_path_list, start=1):
                img = imread(image_path)
                if dataset_name != "test":
                    y = np.array(self.gt[image_path].copy())
                if dataset_name == "train" and self.do_crop:
                    img, y = self.random_sized_crop(img, y)
                img = resize(img, self.image_size, preserve_range=True).astype("float32")
                if dataset_name == "train":
                    for jitter in np.random.permutation(self.color_jitter):
                        img = jitter(img)
                    if self.lighting_std:
                        img = self.lighting(img)
                    if self.hflip_prob > 0:
                        img, y = self.horizontal_flip(img, y)
                    if self.vflip_prob > 0:
                        img, y = self.vertical_flip(img, y)
                inputs.append(img)
                if dataset_name != "test":
                    targets.append(self.bbox_util.assign_boxes(y))
                if len(inputs) == batch_size or image_index == len(image_path_list):
                    tmp_inp = np.array(inputs)
                    tmp_targets = np.array(targets)
                    inputs = []
                    targets = []
                    if dataset_name != "test":
                        yield preprocess_input(tmp_inp), tmp_targets
                    else:
                        yield preprocess_input(tmp_inp)

def load_dataset():
    # Get the labels
    unique_label_list = os.listdir(TRAIN_FOLDER_PATH)
    unique_label_list = sorted([unique_label for unique_label in unique_label_list if os.path.isdir(os.path.join(TRAIN_FOLDER_PATH, unique_label))])
    unique_label_with_object_list = [unique_label for unique_label in unique_label_list if unique_label != LABEL_WITHOUT_OBJECT]

    # Cross validation
    whole_train_image_path_list = glob.glob(os.path.join(TRAIN_FOLDER_PATH, "*/*.jpg"))
    train_index_array, valid_index_array = perform_CV(whole_train_image_path_list)
    train_image_path_array = np.array(whole_train_image_path_list)[train_index_array]
    valid_image_path_array = np.array(whole_train_image_path_list)[valid_index_array]
    test_image_path_list = glob.glob(os.path.join(TEST_FOLDER_PATH, "*.jpg"))

    # Load resolution result
    assert os.path.isfile(RESOLUTION_RESULT_FILE_PATH)
    image_name_with_image_shape_array = np.load(RESOLUTION_RESULT_FILE_PATH)
    image_name_to_image_shape_dict = dict([(image_name_with_image_shape[0], image_name_with_image_shape[1:].astype(np.int)) for image_name_with_image_shape in image_name_with_image_shape_array])

    # Load annotation
    annotation_dict = {}
    annotation_file_path_list = glob.glob(os.path.join(ANNOTATION_FOLDER_PATH, "*.json"))
    unique_label_with_object_in_lowercase_list = [unique_label_with_object.lower() for unique_label_with_object in unique_label_with_object_list]
    for annotation_file_path in annotation_file_path_list:
        unique_label_in_lowercase = os.path.basename(annotation_file_path).split("_")[0].lower()
        unique_label_index = unique_label_with_object_in_lowercase_list.index(unique_label_in_lowercase)
        with open(annotation_file_path) as annotation_file:
            annotation_file_content = json.load(annotation_file)
            for item in annotation_file_content:
                key = os.path.basename(item["filename"])
                value = []
                height, width, _ = image_name_to_image_shape_dict[key]
                for annotation in item["annotations"]:
                    # TODO: Something is wrong!
                    xmin = annotation["x"] / width
                    xmax = (annotation["x"] + annotation["width"]) / width
                    ymin = 1 - (annotation["y"] + annotation["height"]) / height
                    ymax = 1 - annotation["y"] / height
                    coordinate_array = np.clip([xmin, ymin, xmax, ymax], 0, 1)
                    prob_array = np.zeros(len(unique_label_with_object_list))
                    prob_array[unique_label_index] = 1
                    value.append(coordinate_array.tolist() + prob_array.tolist())
                annotation_dict[key] = value

    # Append annotation of images without object
    image_without_object_path_list = glob.glob(os.path.join(TRAIN_FOLDER_PATH, "{}/*.jpg".format(LABEL_WITHOUT_OBJECT)))
    image_without_object_name_list = [os.path.basename(image_path) for image_path in image_without_object_path_list]
    for image_name in image_without_object_name_list:
        annotation_dict[image_name] = []

    # Initialize bounding boxes utility
    bbox_utility = BBoxUtility(num_classes=len(unique_label_list), priors=np.load(DEFAULT_BOXES_FILE_PATH))

    # Construct generator
    data_generator = DataGenerator(annotation_dict, bbox_utility, (IMAGE_ROW_SIZE, IMAGE_COLUMN_SIZE),
                                train_image_path_array.tolist(), valid_image_path_array.tolist(), test_image_path_list)

    return data_generator, unique_label_list, unique_label_with_object_list, bbox_utility

def init_model(unique_label_num, learning_rate=0.0003):
    # Init model
    model = SSD300(input_shape=(IMAGE_ROW_SIZE, IMAGE_COLUMN_SIZE, IMAGE_CHANNEL_SIZE), num_classes=unique_label_num)

    # Load pretrained weights
    model.load_weights(PRETRAINED_WEIGHTS_FILE_PATH, by_name=True)

    # Freeze certain layers
    fixed_layer_name_list = ["input_1", "conv1_1", "conv1_2", "pool1", "conv2_1", "conv2_2", "pool2", "conv3_1", "conv3_2", "conv3_3", "pool3"]
    for layer in model.layers:
        if layer.name in fixed_layer_name_list:
            layer.trainable = False

    # Compile the model
    model.compile(optimizer=Adam(lr=learning_rate), loss=MultiboxLoss(unique_label_num, neg_pos_ratio=2.0).compute_loss)

    return model

def illustrate(image_path_list, model, bbox_utility, unique_label_with_object_list, confidence_threshold=0.6):
    # Use different colors for different labels
    color_array = plt.cm.hsv(np.linspace(0, 1, len(unique_label_with_object_list) + 1))

    # Construct input
    input_array = []
    vanilla_image_list = []
    for image_path in image_path_list:
        image = imread(image_path)
        vanilla_image_list.append(image)
        image = resize(image, (IMAGE_ROW_SIZE, IMAGE_COLUMN_SIZE), preserve_range=True).astype("float32")
        input_array.append(image)

    # Generate output
    input_array = preprocess_input(np.array(input_array))
    output_array = model.predict(input_array, batch_size=BATCH_SIZE, verbose=2)
    converted_output_list = bbox_utility.detection_out(output_array)

    # Create figures
    plt.figure()
    for (image_index, (image_path, image, converted_output)) in enumerate(zip(image_path_list, vanilla_image_list, converted_output_list), start=1):
        plt.imshow(image)
        plt.title("{}/{} {}".format(image_index, len(image_path_list), os.path.basename(image_path)))
        current_axis = plt.gca()

        height, width, _ = image.shape
        for (label, confidence, xmin, ymin, xmax, ymax) in converted_output:
            # Neglect low confidence detections
            if confidence < confidence_threshold:
                continue

            label = int(label)
            xmin_index, xmax_index, ymin_index, ymax_index = np.multiply((xmin, xmax, ymin, ymax), (width, width, height, height)).astype(np.int)

            # Get color for current detection
            color = color_array[label]

            # Add bounding box
            coordinate = ((xmin_index, ymin_index), xmax_index - xmin_index + 1, ymax_index - ymin_index + 1)
            current_axis.add_patch(plt.Rectangle(*coordinate, fill=False, edgecolor=color, linewidth=2))

            # Add text
            display_text = "{:0.2f}, {}".format(confidence, unique_label_with_object_list[label - 1])
            current_axis.text(xmin_index, ymin_index, display_text, bbox={"facecolor":color, "alpha":0.5})

        plt.show()

def run():
    print("Loading dataset ...")
    data_generator, unique_label_list, unique_label_with_object_list, bbox_utility = load_dataset()

    print("Initializing model ...")
    model = init_model(unique_label_num=len(unique_label_list))

    if not os.path.isdir(OUTPUT_FOLDER_PATH):
        print("Creating the output folder ...")
        os.makedirs(OUTPUT_FOLDER_PATH)

    if not os.path.isfile(OPTIMAL_WEIGHTS_FILE_PATH):
        print("Performing the training procedure ...")
        earlystopping_callback = EarlyStopping(monitor="val_loss", patience=PATIENCE)
        modelcheckpoint_callback = ModelCheckpoint(OPTIMAL_WEIGHTS_FILE_PATH, monitor="val_loss", save_best_only=True)
        model.fit_generator(generator=data_generator.generate(dataset_name="train"),
                            samples_per_epoch=len(data_generator.train_image_path_list),
                            validation_data=data_generator.generate(dataset_name="valid"),
                            nb_val_samples=len(data_generator.valid_image_path_list),
                            callbacks=[earlystopping_callback, modelcheckpoint_callback],
                            nb_epoch=MAXIMUM_EPOCH_NUM, verbose=2)

    print("Loading weights at {} ...".format(OPTIMAL_WEIGHTS_FILE_PATH))
    model.load_weights(OPTIMAL_WEIGHTS_FILE_PATH)

    print("Illustrating predictions ...")
    selected_image_path_list = np.random.choice(data_generator.valid_image_path_list, BATCH_SIZE, replace=False).tolist()
    illustrate(selected_image_path_list, model, bbox_utility, unique_label_with_object_list)

    if not os.path.isfile(SUBMISSION_FILE_PATH):
        print("Performing the testing procedure ...")
        prediction_array = model.predict_generator(generator=data_generator.generate(dataset_name="test"), val_samples=len(data_generator.test_image_path_list))
        converted_output_list = bbox_utility.detection_out(prediction_array)
        # TODO: Generate submission file

    print("All done!")

if __name__ == "__main__":
    run()
