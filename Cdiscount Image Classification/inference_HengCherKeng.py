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

def load_text_file(file_path, sep=",", header="infer", usecols=None, quoting=0, chunksize=1e4, encoding="utf-8"):
    file_content = pd.read_csv(file_path, sep=sep, header=header, usecols=usecols, quoting=quoting, chunksize=chunksize, encoding=encoding)
    for chunk in file_content:
        for data in chunk.itertuples(index=False):
            yield data

def get_predictions_for_each_product(prediction_file_path):
    accumulated_product_id = None
    accumulated_label_index_and_prob_value_list = []
    for data in load_text_file(prediction_file_path, header=None):
        # Unpack the values
        product_id = data[0]
        label_index_and_prob_value_list = list(data[2:])

        # Append the record if product_id is the same
        if product_id == accumulated_product_id:
            accumulated_label_index_and_prob_value_list += label_index_and_prob_value_list
            continue

        # Yield the accumulated records
        if len(accumulated_label_index_and_prob_value_list) > 0:
            yield accumulated_product_id, accumulated_label_index_and_prob_value_list

        # Update the accumulated records
        accumulated_product_id = product_id
        accumulated_label_index_and_prob_value_list = label_index_and_prob_value_list

    # Yield the accumulated records
    if len(accumulated_label_index_and_prob_value_list) > 0:
        yield accumulated_product_id, accumulated_label_index_and_prob_value_list

def get_submission_from_prediction(prediction_file_path, label_index_to_category_id_dict, ensemble_func):
    for product_id, label_index_and_prob_value_list in get_predictions_for_each_product(prediction_file_path):
        label_index_to_prob_value_list_dict = {}
        label_index_list = label_index_and_prob_value_list[0::2]
        prob_value_list = label_index_and_prob_value_list[1::2]
        for label_index, prob_value in zip(label_index_list, prob_value_list):
            if label_index not in label_index_to_prob_value_list_dict:
                label_index_to_prob_value_list_dict[label_index] = []
            label_index_to_prob_value_list_dict[label_index].append(prob_value)

        label_index_and_chosen_prob_value_array = np.array([(label_index, ensemble_func(prob_value_list)) for label_index, prob_value_list in label_index_to_prob_value_list_dict.items()])
        chosen_label_index = label_index_and_chosen_prob_value_array[np.argmax(label_index_and_chosen_prob_value_array[:, 1]), 0].astype(np.int)
        chosen_category_id = label_index_to_category_id_dict[chosen_label_index]

        yield product_id, chosen_category_id

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

    print("Loading label_index_to_category_id_dict ...")
    label_index_to_category_id_dict = dict(pd.read_csv(os.path.join(HENGCHERKENG_FOLDER_PATH, "label_index_to_category_id.csv"), header=None).itertuples(index=False))

    print("Generating submission files from prediction files ...")
    for ensemble_name, ensemble_func in zip(["min", "max", "mean", "median"], [np.min, np.max, np.mean, np.median]):
        submission_file_path = os.path.join(SUBMISSION_FOLDER_PATH, "ensembling_{}_{}.csv".format(ensemble_name, time.strftime("%c")).replace(" ", "_").replace(":", "_"))
        with open(submission_file_path, "w") as submission_file_object:
            submission_file_object.write("_id,category_id\n")
        print("Submission will be saved to {}".format(submission_file_path))

        entry_list = []
        for entry in get_submission_from_prediction(prediction_file_path, label_index_to_category_id_dict, ensemble_func):
            # Append the results
            entry_list.append(entry)

            # Save submissions to disk
            if len(entry_list) >= SAVE_EVERY_N_ENTRIES:
                append_entries_to_file(entry_list, submission_file_path)
                entry_list = []

        # Save submissions to disk
        if len(entry_list) > 0:
            append_entries_to_file(entry_list, submission_file_path)
            entry_list = []

    print("All done!")

if __name__ == "__main__":
    run()
