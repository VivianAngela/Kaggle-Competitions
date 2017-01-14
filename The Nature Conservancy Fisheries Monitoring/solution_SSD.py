import os
import glob
import shutil
import json
import numpy as np
from random import shuffle
from keras import backend as K
from keras.applications.imagenet_utils import preprocess_input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from scipy.misc import imread, imresize
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
ACTUAL_DATASET_FOLDER_PATH = os.path.join(WORKSPACE_FOLDER_PATH, "actual_dataset")
ACTUAL_TRAIN_AND_VALID_FOLDER_PATH = os.path.join(ACTUAL_DATASET_FOLDER_PATH, "train_and_valid")
ACTUAL_TEST_FOLDER_PATH = os.path.join(ACTUAL_DATASET_FOLDER_PATH, "test")

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

class Generator(object):
    def __init__(self, gt, bbox_util,
                 batch_size, path_prefix,
                 train_keys, val_keys, image_size,
                 saturation_var=0.5,
                 brightness_var=0.5,
                 contrast_var=0.5,
                 lighting_std=0.5,
                 hflip_prob=0.5,
                 vflip_prob=0.5,
                 do_crop=True,
                 crop_area_range=[0.75, 1.0],
                 aspect_ratio_range=[3. / 4., 4. / 3.]):
        self.gt = gt
        self.bbox_util = bbox_util
        self.batch_size = batch_size
        self.path_prefix = path_prefix
        self.train_keys = train_keys
        self.val_keys = val_keys
        self.train_batches = len(train_keys)
        self.val_batches = len(val_keys)
        self.image_size = image_size
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

    def generate(self, train=True):
        while True:
            if train:
                shuffle(self.train_keys)
                keys = self.train_keys
            else:
                shuffle(self.val_keys)
                keys = self.val_keys
            inputs = []
            targets = []
            for key in keys:
                img_path = self.path_prefix + key
                img = imread(img_path).astype("float32")
                y = self.gt[key].copy()
                if train and self.do_crop:
                    img, y = self.random_sized_crop(img, y)
                img = imresize(img, self.image_size).astype("float32")
                if train:
                    shuffle(self.color_jitter)
                    for jitter in self.color_jitter:
                        img = jitter(img)
                    if self.lighting_std:
                        img = self.lighting(img)
                    if self.hflip_prob > 0:
                        img, y = self.horizontal_flip(img, y)
                    if self.vflip_prob > 0:
                        img, y = self.vertical_flip(img, y)
                y = self.bbox_util.assign_boxes(y)
                inputs.append(img)
                targets.append(y)
                if len(targets) == self.batch_size:
                    tmp_inp = np.array(inputs)
                    tmp_targets = np.array(targets)
                    inputs = []
                    targets = []
                    yield preprocess_input(tmp_inp), tmp_targets

def load_dataset():
    # Get the labels
    unique_label_list = os.listdir(TRAIN_FOLDER_PATH)
    unique_label_list = sorted([unique_label for unique_label in unique_label_list if os.path.isdir(os.path.join(TRAIN_FOLDER_PATH, unique_label))])
    unique_label_with_object_list = [unique_label for unique_label in unique_label_list if unique_label != LABEL_WITHOUT_OBJECT]

    # Cross validation
    whole_train_image_path_list = glob.glob(os.path.join(TRAIN_FOLDER_PATH, "*/*.jpg"))
    whole_train_image_name_list = [os.path.basename(image_path) for image_path in whole_train_image_path_list]
    train_index_array, valid_index_array = perform_CV(whole_train_image_path_list)
    train_image_name_array = np.array(whole_train_image_name_list)[train_index_array]
    valid_image_name_array = np.array(whole_train_image_name_list)[valid_index_array]

    # Create symbolic links
    shutil.rmtree(ACTUAL_DATASET_FOLDER_PATH, ignore_errors=True)
    os.makedirs(ACTUAL_TRAIN_AND_VALID_FOLDER_PATH)
    for image_path in whole_train_image_path_list:
        os.symlink(image_path, os.path.join(ACTUAL_TRAIN_AND_VALID_FOLDER_PATH, os.path.basename(image_path)))
    os.makedirs(ACTUAL_TEST_FOLDER_PATH)
    os.symlink(TEST_FOLDER_PATH, os.path.join(ACTUAL_TEST_FOLDER_PATH, "dummy"))

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
                key = item["filename"]
                value = []
                height, width, _ = image_name_to_image_shape_dict[key]
                for annotation in item["annotations"]:
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
    generator = Generator(annotation_dict, bbox_utility, BATCH_SIZE, ACTUAL_TRAIN_AND_VALID_FOLDER_PATH, train_image_name_array, valid_image_name_array, (IMAGE_ROW_SIZE, IMAGE_COLUMN_SIZE), do_crop=False)

    return generator, unique_label_list, unique_label_with_object_list

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

def run():
    print("Loading dataset ...")
    generator, unique_label_list, unique_label_with_object_list = load_dataset()

    print("Initializing model ...")
    model = init_model(unique_label_num=len(unique_label_list))

    if not os.path.isdir(OUTPUT_FOLDER_PATH):
        print("Creating the output folder ...")
        os.makedirs(OUTPUT_FOLDER_PATH)

    if not os.path.isfile(OPTIMAL_WEIGHTS_FILE_PATH):
        print("Performing the training procedure ...")
        earlystopping_callback = EarlyStopping(monitor="val_loss", patience=PATIENCE)
        modelcheckpoint_callback = ModelCheckpoint(OPTIMAL_WEIGHTS_FILE_PATH, monitor="val_loss", save_best_only=True)
        model.fit_generator(generator=generator.generate(True),
                            samples_per_epoch=generator.train_batches,
                            validation_data=generator.generate(False),
                            nb_val_samples=generator.val_batches,
                            callbacks=[earlystopping_callback, modelcheckpoint_callback],
                            nb_epoch=MAXIMUM_EPOCH_NUM, verbose=2)

    print("All done!")

if __name__ == "__main__":
    run()
