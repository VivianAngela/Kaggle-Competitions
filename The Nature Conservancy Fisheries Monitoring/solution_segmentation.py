import numpy as np
from keras.layers import Input, merge
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import UpSampling2D
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model

# Image processing
IMAGE_CHANNEL_SIZE = 3
IMAGE_ROW_SIZE = 320
IMAGE_COLUMN_SIZE = 320

# Model definition
ENCODER_FILTER_NUM_ARRAY = (np.arange(5) + 1) * 32
DECODER_FILTER_NUM_ARRAY = np.hstack((np.flipud(ENCODER_FILTER_NUM_ARRAY)[1:], 1))

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

def run():
    print("Initializing model ...")
    model = init_model()

    print("All done!")

if __name__ == "__main__":
    run()
