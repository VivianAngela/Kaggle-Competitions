# Adapted from https://www.kaggle.com/c/cdiscount-image-classification-challenge/discussion/41021
import os
import cv2
import time
import glob
import numpy as np
import pandas as pd

# Import torch-related functions
import torch
import torch.nn.functional as F
from torch.autograd import Variable

# Dataset
PROJECT_NAME = "Cdiscount Image Classification"
PROJECT_FOLDER_PATH = os.path.join(os.path.expanduser("~"), "Documents/Dataset", PROJECT_NAME)
HENGCHERKENG_FOLDER_PATH = os.path.join(PROJECT_FOLDER_PATH, "HengCherKeng")
EXTRACTED_DATASET_FOLDER_PATH = os.path.join(PROJECT_FOLDER_PATH, "extracted")
TEST_FOLDER_PATH = os.path.join(EXTRACTED_DATASET_FOLDER_PATH, "test")
SUBMISSION_FOLDER_PATH = os.path.join(PROJECT_FOLDER_PATH, "submission")

# Add the HengCherKeng folder to the path
import sys
sys.path.append(HENGCHERKENG_FOLDER_PATH)
from inception_v3 import Inception3  # @UnresolvedImport pylint: disable=import-error
from excited_inception_v3 import SEInception3  # @UnresolvedImport pylint: disable=import-error
MODEL_NAME_TO_MODEL_DETAILS_DICT = {"Inception3": (Inception3, "LB=0.69565_inc3_00075000_model.pth"),
                                    "SEInception3": (SEInception3, "LB=0.69673_se-inc3_00026000_model.pth")}
MODEL_NAME = "SEInception3"
MODEL_FUNCTION, MODEL_FILE_NAME = MODEL_NAME_TO_MODEL_DETAILS_DICT[MODEL_NAME]

# Hyperparameters for the neural network
HEIGHT, WIDTH = 180, 180
NUM_CLASSES = 5270

# Save top N predictions to disk
TOP_N_PREDICTIONS = 5

# Save predictions to disk when there are N entries
SAVE_EVERY_N_ENTRIES = 1000

def pytorch_image_to_tensor_transform(image):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.transpose((2, 0, 1))
    tensor = torch.from_numpy(image).float().div(255)  # @UndefinedVariable
    tensor[0] = (tensor[0] - mean[0]) / std[0]
    tensor[1] = (tensor[1] - mean[1]) / std[1]
    tensor[2] = (tensor[2] - mean[2]) / std[2]
    return tensor

def image_to_tensor_transform(image):
    tensor = pytorch_image_to_tensor_transform(image)
    tensor[0] = tensor[0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
    tensor[1] = tensor[1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
    tensor[2] = tensor[2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
    return tensor

def append_entries_to_file(entry_list, file_path):
    file_content = pd.DataFrame(entry_list)
    file_content.to_csv(file_path, header=None, index=False, mode="a", float_format="%.2f", encoding="utf-8")

def run():
    print("Creating folders ...")
    os.makedirs(SUBMISSION_FOLDER_PATH, exist_ok=True)

    print("Loading {} ...".format(MODEL_NAME))
    net = MODEL_FUNCTION(in_shape=(3, HEIGHT, WIDTH), num_classes=NUM_CLASSES)
    net.load_state_dict(torch.load(os.path.join(HENGCHERKENG_FOLDER_PATH, MODEL_FILE_NAME)))
    net.cuda().eval()

    prediction_file_path = os.path.join(SUBMISSION_FOLDER_PATH, "{}_prediction_{}.csv".format(MODEL_NAME, time.strftime("%c")).replace(" ", "_").replace(":", "_"))
    open(prediction_file_path, "w").close()
    print("Prediction will be saved to {}".format(prediction_file_path))

    entry_list = []
    image_file_path_list = sorted(glob.glob(os.path.join(TEST_FOLDER_PATH, "*/*.jpg")))
    for image_file_path in image_file_path_list:
        # Read image
        image = cv2.imread(image_file_path)
        x = image_to_tensor_transform(image)
        x = Variable(x.unsqueeze(0), volatile=True).cuda()

        # Inference
        logits = net(x)
        probs = F.softmax(logits)
        probs = probs.cpu().data.numpy().reshape(-1)

        # Get the top N predictions
        top_n_index_array = probs.argsort()[-TOP_N_PREDICTIONS:][::-1]
        top_n_prob_array = probs[top_n_index_array]

        # Append the results
        entry = [tuple(os.path.basename(image_file_path).split(".")[0].split("_"))] + [*zip(top_n_index_array, top_n_prob_array)]
        entry_list.append([np.int64(item) if isinstance(item, str) else item for item_list in entry for item in item_list])

        # Save predictions to disk
        if len(entry_list) >= SAVE_EVERY_N_ENTRIES:
            append_entries_to_file(entry_list, prediction_file_path)
            entry_list = []

    # Save predictions to disk
    if len(entry_list) > 0:
        append_entries_to_file(entry_list, prediction_file_path)
        entry_list = []

    print("All done!")

if __name__ == "__main__":
    run()
