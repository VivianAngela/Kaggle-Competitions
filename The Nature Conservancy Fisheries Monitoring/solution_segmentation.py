import os
import glob
import shutil
import json
import pylab
import numpy as np
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.layers import Input, merge
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Convolution2D, UpSampling2D
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from scipy.misc import imread, imresize
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder

# Dataset
DATASET_FOLDER_PATH = os.path.join(os.path.expanduser("~"), "Documents/Dataset/The Nature Conservancy Fisheries Monitoring")
TRAIN_FOLDER_PATH = os.path.join(DATASET_FOLDER_PATH, "train")
TEST_FOLDER_PATH = os.path.join(DATASET_FOLDER_PATH, "test_stg1")
ANNOTATION_FOLDER_PATH = os.path.join(DATASET_FOLDER_PATH, "annotations")
RESOLUTION_RESULT_FILE_PATH = os.path.join(DATASET_FOLDER_PATH, "resolution_result.npy")

# Workspace
WORKSPACE_FOLDER_PATH = os.path.join("/tmp", os.path.basename(DATASET_FOLDER_PATH))
CLUSTERING_FOLDER_PATH = os.path.join(WORKSPACE_FOLDER_PATH, "clustering")

# Output
OUTPUT_FOLDER_PATH = os.path.join(DATASET_FOLDER_PATH, "{}_output".format(os.path.basename(__file__).split(".")[0]))
VISUALIZATION_FOLDER_PATH = os.path.join(OUTPUT_FOLDER_PATH, "Visualization")
OPTIMAL_WEIGHTS_FOLDER_PATH = os.path.join(OUTPUT_FOLDER_PATH, "Optimal Weights")
OPTIMAL_WEIGHTS_FILE_RULE = os.path.join(OPTIMAL_WEIGHTS_FOLDER_PATH, "epoch_{epoch:03d}-val_loss_{val_loss:.5f}.h5")

# Image processing
IMAGE_CHANNEL_SIZE = 3
IMAGE_ROW_SIZE = 320
IMAGE_COLUMN_SIZE = 320

# Model definition
ENCODER_FILTER_NUM_ARRAY = (np.arange(5) + 1) * 32
DECODER_FILTER_NUM_ARRAY = np.hstack((np.flipud(ENCODER_FILTER_NUM_ARRAY)[1:], 1))

# Training and Testing procedure
MAXIMUM_EPOCH_NUM = 1000
PATIENCE = 10
BATCH_SIZE = 32
INSPECT_SIZE = 4

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

def get_index_generator(sample_num):
    current_seed = 0
    while True:
        np.random.seed(current_seed)
        for index in np.random.permutation(sample_num):
            yield index
        current_seed += 1

def get_sample_generator(index_generator, image_path_array, annotation_dict):
    for index in index_generator:
        image_path = image_path_array[index]
        X = imresize(imread(image_path), size=(IMAGE_ROW_SIZE, IMAGE_COLUMN_SIZE, IMAGE_CHANNEL_SIZE)) / 255.0

        coordinate_array_list = []
        image_name = os.path.basename(image_path)
        if image_name in annotation_dict:
            coordinate_array_list = annotation_dict[image_name]

        Y = np.zeros((IMAGE_ROW_SIZE, IMAGE_COLUMN_SIZE, 1))
        for coordinate_array in coordinate_array_list:
            xmin, ymin, xmax, ymax = coordinate_array
            column_min_index, column_max_index, row_min_index, row_max_index = np.multiply((xmin, xmax, ymin, ymax), (IMAGE_COLUMN_SIZE, IMAGE_COLUMN_SIZE, IMAGE_ROW_SIZE, IMAGE_ROW_SIZE)).astype(np.int)
            Y[row_min_index:row_max_index, column_min_index:column_max_index] = 1

        yield np.rollaxis(X, -1, 0), np.rollaxis(Y, -1, 0)

def get_data_generator(image_path_array, annotation_dict, sample_num_per_batch):
    index_generator = get_index_generator(sample_num=len(image_path_array))
    sample_generator = get_sample_generator(index_generator, image_path_array, annotation_dict)

    X_list = []
    Y_list = []
    for X, Y in sample_generator:
        if len(X_list) < sample_num_per_batch:
            X_list.append(X)
            Y_list.append(Y)
        if len(X_list) == sample_num_per_batch:
            yield np.array(X_list).astype(np.float32), np.array(Y_list).astype(np.float32)
            X_list = []
            Y_list = []

def load_dataset(sample_num_per_batch):
    # Get the labels
    unique_label_list = os.listdir(TRAIN_FOLDER_PATH)
    unique_label_list = sorted([unique_label for unique_label in unique_label_list if os.path.isdir(os.path.join(TRAIN_FOLDER_PATH, unique_label))])

    # Cross validation
    whole_train_image_path_list = sorted(glob.glob(os.path.join(TRAIN_FOLDER_PATH, "*/*.jpg")))
    train_index_array, valid_index_array = perform_CV(whole_train_image_path_list)
    train_image_path_array = np.array(whole_train_image_path_list)[train_index_array]
    valid_image_path_array = np.array(whole_train_image_path_list)[valid_index_array]
    test_image_path_array = np.array(glob.glob(os.path.join(TEST_FOLDER_PATH, "*.jpg")))

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
                height, width, _ = image_name_to_image_shape_dict[key]
                for annotation in item["annotations"]:
                    xmin = annotation["x"] / width
                    xmax = (annotation["x"] + annotation["width"]) / width
                    ymin = annotation["y"] / height
                    ymax = (annotation["y"] + annotation["height"]) / height
                    coordinate_array = np.clip([xmin, ymin, xmax, ymax], 0, 1)
                    value.append(coordinate_array)
                if key in annotation_dict:
                    assert False, "Found existing key {}!!!".format(key)
                annotation_dict[key] = value

    # Get data generator
    train_data_generator = get_data_generator(train_image_path_array, annotation_dict, sample_num_per_batch)
    valid_data_generator = get_data_generator(valid_image_path_array, annotation_dict, sample_num_per_batch)
    test_data_generator = get_data_generator(test_image_path_array, annotation_dict, sample_num_per_batch)

    return train_data_generator, len(train_image_path_array), \
        valid_data_generator, len(valid_image_path_array), \
        test_data_generator, len(test_image_path_array)

def init_model():
    # Vanilla input
    input_image_tensor = Input(shape=(IMAGE_CHANNEL_SIZE, IMAGE_ROW_SIZE, IMAGE_COLUMN_SIZE))

    # Encoder
    current_input_tensor = input_image_tensor
    encoder_output_tensor_list = []
    for encoder_filter_num in ENCODER_FILTER_NUM_ARRAY:
        current_output_tensor = Convolution2D(encoder_filter_num, 3, 3, subsample=(2, 2), activation="linear", border_mode="same")(current_input_tensor)
        current_output_tensor = BatchNormalization(mode=0, axis=1)(current_output_tensor)
        current_output_tensor = LeakyReLU()(current_output_tensor)
        encoder_output_tensor_list.append(current_output_tensor)
        current_input_tensor = current_output_tensor

    # Decoder
    current_input_tensor = None
    for layer_index, (encoder_output_tensor, decoder_filter_num) in enumerate(zip(np.flipud(encoder_output_tensor_list), DECODER_FILTER_NUM_ARRAY)):
        if current_input_tensor is not None:
            current_input_tensor = merge([current_input_tensor, encoder_output_tensor], mode="concat", concat_axis=1)
        else:
            current_input_tensor = encoder_output_tensor
        current_output_tensor = UpSampling2D(size=(2, 2))(current_input_tensor)
        current_output_tensor = Convolution2D(decoder_filter_num, 3, 3, subsample=(1, 1), activation="linear", border_mode="same")(current_output_tensor)
        if layer_index != len(ENCODER_FILTER_NUM_ARRAY) - 1:
            current_output_tensor = BatchNormalization(mode=0, axis=1)(current_output_tensor)
            current_output_tensor = LeakyReLU()(current_output_tensor)
        else:
            current_output_tensor = Activation("sigmoid")(current_output_tensor)
        current_input_tensor = current_output_tensor

    # Construct the model
    model = Model(input_image_tensor, current_output_tensor)
    model.summary()

    # Compile the model
    model.compile(optimizer="adam", loss="binary_crossentropy")

    return model

def get_optimal_weights_file_path():
    optimal_weights_file_path_list = sorted(glob.glob(os.path.join(OPTIMAL_WEIGHTS_FOLDER_PATH, "*.h5")))
    if len(optimal_weights_file_path_list) > 0:
        return optimal_weights_file_path_list[-1]
    else:
        return None

class InspectPrediction(Callback):
    def __init__(self, data_generator_list):
        super(InspectPrediction, self).__init__()

        self.data_generator_list = data_generator_list

    def on_epoch_end(self, epoch, logs=None):
        for data_generator_index, data_generator in enumerate(self.data_generator_list, start=1):
            X_array, GT_Y_array = next(data_generator)
            P_Y_array = self.model.predict_on_batch(X_array)

            for sample_index, (X, GT_Y, P_Y) in enumerate(zip(X_array, GT_Y_array, P_Y_array), start=1):
                pylab.figure()
                pylab.subplot(1, 3, 1)
                pylab.imshow(np.rollaxis(X, 0, 3))
                pylab.title("X")
                pylab.axis("off")
                pylab.subplot(1 , 3, 2)
                pylab.imshow(GT_Y[0], cmap="gray")
                pylab.title("GT_Y")
                pylab.axis("off")
                pylab.subplot(1 , 3, 3)
                pylab.imshow(P_Y[0], cmap="gray")
                pylab.title("P_Y")
                pylab.axis("off")
                pylab.savefig(os.path.join(VISUALIZATION_FOLDER_PATH, "Epoch_{}_Split_{}_Sample_{}.jpg".format(epoch + 1, data_generator_index, sample_index)))
                pylab.close()

class InspectLoss(Callback):
    def __init__(self):
        super(InspectLoss, self).__init__()

        self.train_loss_list = []
        self.valid_loss_list = []

    def on_epoch_end(self, epoch, logs=None):
        train_loss = logs.get("loss")
        valid_loss = logs.get("val_loss")
        self.train_loss_list.append(train_loss)
        self.valid_loss_list.append(valid_loss)
        epoch_index_array = np.arange(len(self.train_loss_list)) + 1

        pylab.figure()
        pylab.plot(epoch_index_array, self.train_loss_list, "yellowgreen", label="train_loss")
        pylab.plot(epoch_index_array, self.valid_loss_list, "lightskyblue", label="valid_loss")
        pylab.grid()
        pylab.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2, ncol=2, mode="expand", borderaxespad=0.)
        pylab.savefig(os.path.join(OUTPUT_FOLDER_PATH, "Loss Curve.jpg"))
        pylab.close()

def run():
    print("Loading dataset ...")
    train_data_generator, train_sample_num, valid_data_generator, valid_sample_num, _, _ = load_dataset(BATCH_SIZE)
    inspectprediction_train_data_generator, _, inspectprediction_valid_data_generator, _, inspectprediction_test_data_generator, _ = load_dataset(INSPECT_SIZE)

    print("Initializing model ...")
    model = init_model()

    print("Creating folders ...")
    os.makedirs(VISUALIZATION_FOLDER_PATH, exist_ok=True)
    os.makedirs(OPTIMAL_WEIGHTS_FOLDER_PATH, exist_ok=True)

    optimal_weights_file_path = get_optimal_weights_file_path()
    if optimal_weights_file_path is None:
        print("Performing the training procedure ...")
        earlystopping_callback = EarlyStopping(monitor="val_loss", patience=PATIENCE)
        modelcheckpoint_callback = ModelCheckpoint(OPTIMAL_WEIGHTS_FILE_RULE, monitor="val_loss", save_weights_only=True, save_best_only=True)
        inspectprediction_callback = InspectPrediction([inspectprediction_train_data_generator, inspectprediction_valid_data_generator, inspectprediction_test_data_generator])
        inspectloss_callback = InspectLoss()
        model.fit_generator(generator=train_data_generator,
                            samples_per_epoch=train_sample_num,
                            validation_data=valid_data_generator,
                            nb_val_samples=valid_sample_num,
                            callbacks=[earlystopping_callback, modelcheckpoint_callback, inspectprediction_callback, inspectloss_callback],
                            nb_epoch=MAXIMUM_EPOCH_NUM, verbose=2)
        optimal_weights_file_path = get_optimal_weights_file_path()

    print("Loading weights at {} ...".format(optimal_weights_file_path))
    model.load_weights(optimal_weights_file_path)

    print("All done!")

if __name__ == "__main__":
    run()
