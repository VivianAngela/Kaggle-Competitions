from __future__ import absolute_import, division, print_function

import matplotlib
matplotlib.use("Agg")

import os
import pylab
import numpy as np
from keras import backend as K
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Input, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Nadam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.visualize_util import plot
from data_preprocessing import PROJECT_FOLDER_PATH
from data_preprocessing import PROCESSED_DATASET_FOLDER_PATH as DATASET_FOLDER_PATH
from data_preprocessing import PROCESSED_IMAGE_HEIGHT as IMAGE_HEIGHT
from data_preprocessing import PROCESSED_IMAGE_WIDTH as IMAGE_WIDTH

# Choose ResNet50 or InceptionV3 or VGG16
MODEL_NAME = "ResNet50"  # "ResNet50" or "InceptionV3" or "VGG16"
if MODEL_NAME == "ResNet50":
    from keras.applications.resnet50 import preprocess_input as PREPROCESS_INPUT
    from keras.applications.resnet50 import ResNet50 as INIT_FUNC
    BOTTLENECK_LAYER_NAME = "activation_40"
    DROPOUT_RATIO = 0.5
    LEARNING_RATE = 0.00001
elif MODEL_NAME == "InceptionV3":
    from keras.applications.inception_v3 import preprocess_input as PREPROCESS_INPUT
    from keras.applications.inception_v3 import InceptionV3 as INIT_FUNC
    BOTTLENECK_LAYER_NAME = "mixed8"
    DROPOUT_RATIO = 0.5
    LEARNING_RATE = 0.00001
elif MODEL_NAME == "VGG16":
    from keras.applications.vgg16 import preprocess_input as PREPROCESS_INPUT
    from keras.applications.vgg16 import VGG16 as INIT_FUNC
    BOTTLENECK_LAYER_NAME = "block4_pool"
    DROPOUT_RATIO = 0.5
    LEARNING_RATE = 0.00005
else:
    assert False

# Dataset
TRAIN_FOLDER_PATH = os.path.join(DATASET_FOLDER_PATH, "additional")

# Workspace
ACTUAL_TRAIN_FOLDER_PATH = os.path.join(DATASET_FOLDER_PATH, "additional")
ACTUAL_VALID_FOLDER_PATH = os.path.join(DATASET_FOLDER_PATH, "train")

# Output
OUTPUT_FOLDER_PATH = os.path.join(PROJECT_FOLDER_PATH, "phase_1")
OPTIMAL_WEIGHTS_FOLDER_PATH = os.path.join(OUTPUT_FOLDER_PATH, "optimal weights")
OPTIMAL_WEIGHTS_FILE_PATH = os.path.join(OPTIMAL_WEIGHTS_FOLDER_PATH, "{}.h5".format(MODEL_NAME))

# Training procedure
WEIGHTS_FILE_PATH = None
MAXIMUM_EPOCH_NUM = 1000
PATIENCE = 4
BATCH_SIZE = 32
SEED = 0

def init_model(image_height, image_width, unique_label_num, init_func=INIT_FUNC, bottleneck_layer_name=BOTTLENECK_LAYER_NAME, dropout_ratio=DROPOUT_RATIO, learning_rate=LEARNING_RATE):
    def set_model_trainable_properties(model, trainable, bottleneck_layer_name):
        for layer in model.layers:
            layer.trainable = trainable
            if layer.name == bottleneck_layer_name:
                break

    def get_feature_extractor(input_shape):
        feature_extractor = init_func(include_top=False, weights="imagenet", input_shape=input_shape)
        set_model_trainable_properties(model=feature_extractor, trainable=False, bottleneck_layer_name=bottleneck_layer_name)
        return feature_extractor

    def get_dense_classifier(input_shape, unique_label_num):
        input_tensor = Input(shape=input_shape)
        output_tensor = GlobalAveragePooling2D()(input_tensor)
        output_tensor = Dropout(dropout_ratio)(output_tensor)
        output_tensor = Dense(unique_label_num, activation="softmax")(output_tensor)
        model = Model(input_tensor, output_tensor)
        return model

    # Initiate the input tensor
    if K.image_dim_ordering() == "tf":
        input_tensor = Input(shape=(image_height, image_width, 3))
    else:
        input_tensor = Input(shape=(3, image_height, image_width))

    # Define the feature extractor
    feature_extractor = get_feature_extractor(input_shape=K.int_shape(input_tensor)[1:])
    output_tensor = feature_extractor(input_tensor)

    # Define the dense classifier
    dense_classifier = get_dense_classifier(input_shape=feature_extractor.output_shape[1:], unique_label_num=unique_label_num)
    output_tensor = dense_classifier(output_tensor)

    # Define the overall model
    model = Model(input_tensor, output_tensor)
    model.compile(optimizer=Nadam(lr=learning_rate), loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    # Plot the model structures
    plot(feature_extractor, to_file=os.path.join(OPTIMAL_WEIGHTS_FOLDER_PATH, "{}_feature_extractor.png".format(MODEL_NAME)), show_shapes=True, show_layer_names=True)
    plot(dense_classifier, to_file=os.path.join(OPTIMAL_WEIGHTS_FOLDER_PATH, "{}_dense_classifier.png".format(MODEL_NAME)), show_shapes=True, show_layer_names=True)
    plot(model, to_file=os.path.join(OPTIMAL_WEIGHTS_FOLDER_PATH, "{}_model.png".format(MODEL_NAME)), show_shapes=True, show_layer_names=True)

    # Load weights if applicable
    if WEIGHTS_FILE_PATH is not None:
        assert os.path.isfile(WEIGHTS_FILE_PATH), "Could not find file {}!".format(WEIGHTS_FILE_PATH)
        print("Loading weights from {} ...".format(WEIGHTS_FILE_PATH))
        model.load_weights(WEIGHTS_FILE_PATH)

    return model

def load_dataset(folder_path, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH), classes=None, class_mode=None, batch_size=BATCH_SIZE, shuffle=True, seed=None, preprocess_input=PREPROCESS_INPUT):
    # Get the generator of the dataset
    data_generator_object = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        zoom_range=0.05,
        horizontal_flip=True,
        vertical_flip=True,
        preprocessing_function=lambda sample: preprocess_input(np.array([sample]))[0])
    data_generator = data_generator_object.flow_from_directory(
        directory=folder_path,
        target_size=target_size,
        color_mode="rgb",
        classes=classes,
        class_mode=class_mode,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed)

    return data_generator

class InspectLossAccuracy(Callback):
    def __init__(self):
        super(InspectLossAccuracy, self).__init__()

        self.train_loss_list = []
        self.valid_loss_list = []

        self.train_acc_list = []
        self.valid_acc_list = []

    def on_epoch_end(self, epoch, logs=None):
        # Loss
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
        pylab.savefig(os.path.join(OUTPUT_FOLDER_PATH, "Loss Curve.png"))
        pylab.close()

        # Accuracy
        train_acc = logs.get("acc")
        valid_acc = logs.get("val_acc")
        self.train_acc_list.append(train_acc)
        self.valid_acc_list.append(valid_acc)
        epoch_index_array = np.arange(len(self.train_acc_list)) + 1

        pylab.figure()
        pylab.plot(epoch_index_array, self.train_acc_list, "yellowgreen", label="train_acc")
        pylab.plot(epoch_index_array, self.valid_acc_list, "lightskyblue", label="valid_acc")
        pylab.grid()
        pylab.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2, ncol=2, mode="expand", borderaxespad=0.)
        pylab.savefig(os.path.join(OUTPUT_FOLDER_PATH, "Accuracy Curve.png"))
        pylab.close()

def run():
    print("Creating folders ...")
    os.makedirs(OPTIMAL_WEIGHTS_FOLDER_PATH, exist_ok=True)

    print("Getting the labels ...")
    unique_label_list = sorted([folder_name for folder_name in os.listdir(TRAIN_FOLDER_PATH) if os.path.isdir(os.path.join(TRAIN_FOLDER_PATH, folder_name))])

    print("Initializing model ...")
    model = init_model(image_height=IMAGE_HEIGHT, image_width=IMAGE_WIDTH, unique_label_num=len(unique_label_list))

    print("Performing the training procedure ...")
    train_generator = load_dataset(ACTUAL_TRAIN_FOLDER_PATH, classes=unique_label_list, class_mode="categorical", shuffle=True, seed=SEED)
    valid_generator = load_dataset(ACTUAL_VALID_FOLDER_PATH, classes=unique_label_list, class_mode="categorical", shuffle=True, seed=SEED)
    train_sample_num = len(train_generator.filenames)
    valid_sample_num = len(valid_generator.filenames)
    earlystopping_callback = EarlyStopping(monitor="val_loss", patience=PATIENCE)
    modelcheckpoint_callback = ModelCheckpoint(OPTIMAL_WEIGHTS_FILE_PATH, monitor="val_loss", save_best_only=True, save_weights_only=True)
    inspectlossaccuracy_callback = InspectLossAccuracy()
    model.fit_generator(generator=train_generator,
                        samples_per_epoch=train_sample_num,
                        validation_data=valid_generator,
                        nb_val_samples=valid_sample_num,
                        callbacks=[earlystopping_callback, modelcheckpoint_callback, inspectlossaccuracy_callback],
                        nb_epoch=MAXIMUM_EPOCH_NUM, verbose=2)

    print("All done!")

if __name__ == "__main__":
    run()
