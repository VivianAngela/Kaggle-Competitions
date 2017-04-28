from __future__ import absolute_import, division, print_function

import matplotlib
matplotlib.use("Agg")

import os
import pylab
import numpy as np
from keras.applications.resnet50 import conv_block, identity_block, preprocess_input, ResNet50, TH_WEIGHTS_PATH_NO_TOP
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Input, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.visualize_util import plot
from data_preprocessing import PROCESSED_DATASET_FOLDER_PATH as DATASET_FOLDER_PATH
from data_preprocessing import PROCESSED_IMAGE_HEIGHT as IMAGE_HEIGHT
from data_preprocessing import PROCESSED_IMAGE_WIDTH as IMAGE_WIDTH

# Dataset
TRAIN_FOLDER_PATH = os.path.join(DATASET_FOLDER_PATH, "additional")

# Workspace
ACTUAL_TRAIN_FOLDER_PATH = os.path.join(DATASET_FOLDER_PATH, "additional")
ACTUAL_VALID_FOLDER_PATH = os.path.join(DATASET_FOLDER_PATH, "train")

# Output
OUTPUT_FOLDER_PATH = os.path.join(DATASET_FOLDER_PATH, "{}_output".format(os.path.basename(__file__).split(".")[0]))
OPTIMAL_WEIGHTS_FOLDER_PATH = os.path.join(OUTPUT_FOLDER_PATH, "Optimal Weights")
OPTIMAL_WEIGHTS_FILE_RULE = os.path.join(OPTIMAL_WEIGHTS_FOLDER_PATH, "epoch_{epoch:03d}-loss_{loss:.5f}-val_loss_{val_loss:.5f}.h5")

# Training procedure
WEIGHTS_FILE_PATH = None
MAXIMUM_EPOCH_NUM = 1000
PATIENCE = 100
BATCH_SIZE = 32
SEED = 0

def init_model(image_height=224, image_width=224, unique_label_num=1000, learning_rate=0.00001):
    def set_model_trainable_properties(model, trainable):
        for layer in model.layers:
            layer.trainable = trainable
        model.trainable = trainable

    def get_feature_extractor(input_shape):
        feature_extractor = ResNet50(include_top=False, weights="imagenet", input_shape=input_shape)
        feature_extractor = Model(input=feature_extractor.input, output=feature_extractor.get_layer("activation_40").output)
        set_model_trainable_properties(feature_extractor, False)
        return feature_extractor

    def get_trainable_classifier(input_shape, unique_label_num):
        input_tensor = Input(shape=input_shape)

        x = conv_block(input_tensor, 3, [512, 512, 2048], stage=5, block="a")
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block="b")
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block="c")

        x = GlobalAveragePooling2D(name="global_avg_pool")(x)
        x = Dense(unique_label_num, activation="softmax", name="fc{}".format(unique_label_num))(x)

        model = Model(input_tensor, x)
        return model

    # Initiate the input tensor
    input_tensor = Input(shape=(3, image_height, image_width))

    # Define the feature extractor
    feature_extractor = get_feature_extractor(input_shape=input_tensor._keras_shape[1:])  # pylint: disable=W0212
    output_tensor = feature_extractor(input_tensor)

    # Define the trainable classifier
    trainable_classifier = get_trainable_classifier(input_shape=feature_extractor.output_shape[1:], unique_label_num=unique_label_num)
    output_tensor = trainable_classifier(output_tensor)

    # Define the overall model
    model = Model(input_tensor, output_tensor)
    model.compile(optimizer=Adam(lr=learning_rate), loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    # Plot the model structures
    plot(feature_extractor, to_file=os.path.join(OPTIMAL_WEIGHTS_FOLDER_PATH, "feature_extractor.png"), show_shapes=True, show_layer_names=True)
    plot(trainable_classifier, to_file=os.path.join(OPTIMAL_WEIGHTS_FOLDER_PATH, "trainable_classifier.png"), show_shapes=True, show_layer_names=True)
    plot(model, to_file=os.path.join(OPTIMAL_WEIGHTS_FOLDER_PATH, "model.png"), show_shapes=True, show_layer_names=True)

    # Load weights for the trainable classifier
    trainable_classifier.load_weights(os.path.expanduser(os.path.join("~", ".keras/models", TH_WEIGHTS_PATH_NO_TOP.split("/")[-1])), by_name=True)

    # Load weights if applicable
    if WEIGHTS_FILE_PATH is not None:
        assert os.path.isfile(WEIGHTS_FILE_PATH), "Could not find file {}!".format(WEIGHTS_FILE_PATH)
        print("Loading weights from {} ...".format(WEIGHTS_FILE_PATH))
        model.load_weights(WEIGHTS_FILE_PATH)

    return model

def load_dataset(folder_path, classes=None, class_mode=None, batch_size=BATCH_SIZE, shuffle=True, seed=None):
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
        target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
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
    modelcheckpoint_callback = ModelCheckpoint(OPTIMAL_WEIGHTS_FILE_RULE, monitor="val_loss", save_best_only=True, save_weights_only=True)
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
