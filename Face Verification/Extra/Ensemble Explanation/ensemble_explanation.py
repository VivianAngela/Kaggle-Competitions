from sklearn.metrics.pairwise import pairwise_distances
import glob
import numpy as np
import os
import pandas as pd
import sys
script_folder_path = os.path.abspath(os.path.join("../.."))
sys.path.append(script_folder_path)
import common

prediction_list = []
for current_file_path in sorted(glob.glob(os.path.join(common.SUBMISSIONS_FOLDER_PATH, "Aurora_bbox_open_face_keras_Model_*.csv"))):
# for current_file_path in sorted(glob.glob(os.path.join(common.SUBMISSIONS_FOLDER_PATH, "Aurora_bbox_open_face_sklearn_Model_*.csv"))):
    current_file_path = pd.read_csv(current_file_path)
    prediction_list.append(current_file_path["Prediction"].as_matrix())

prediction_array = np.array(prediction_list)
distances = pairwise_distances(prediction_array, metric="cosine", n_jobs=-1)
distances = np.abs(distances)

for row_index in np.arange(distances.shape[0]):
    for column_index in np.arange(distances.shape[1] - 1):
        print("{:.5f} & ".format(distances[row_index, column_index]), end="")
    print("{:.5f} \\\\ \\hline".format(distances[row_index, -1]))

values = np.reshape(distances, distances.size)
is_zero = values < (0.1) ** 8
print("{:.5f}".format(np.mean(values[np.logical_not(is_zero)])))
