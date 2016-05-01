import file_operations
import glob
import numpy as np
import os
import pandas as pd
import solution
import time

OLD_SUBMISSION_FOLDER_PATH = solution.SUBMISSION_FOLDER_PATH
NEW_SUBMISSION_FOLDER_PATH = "./"

def perform_ensembling(low_threshold, high_threshold):
    print("Reading the submission files from disk ...")
    prediction_list = []
    for submission_file_path in glob.glob(os.path.join(OLD_SUBMISSION_FOLDER_PATH, "*.csv")):
        if os.path.basename(submission_file_path) < "Aurora_{:.4f}".format(low_threshold) or \
            os.path.basename(submission_file_path) > "Aurora_{:.4f}".format(high_threshold):
            continue
        submission_file_content = pd.read_csv(submission_file_path)
        prediction = submission_file_content[file_operations.LABEL_COLUMN_NAME_IN_SUBMISSION].as_matrix()
        prediction_list.append(prediction)

    print("Writing the submission files to disk ...")
    mean_prediction = np.mean(prediction_list, axis=0)
    median_prediction = np.median(prediction_list, axis=0)
    for bias, prediction in enumerate([mean_prediction, median_prediction]):
        submission_file_name = "Ensemble_{:.4f}_to_{:.4f}_{:d}.csv".format(low_threshold, high_threshold, int(time.time()) + bias)
        submission_file_path = os.path.join(NEW_SUBMISSION_FOLDER_PATH, submission_file_name)
        submission_file_content[file_operations.LABEL_COLUMN_NAME_IN_SUBMISSION] = (prediction > 0.5).astype(np.int)
        submission_file_content.to_csv(submission_file_path, index=False)

perform_ensembling(0, 1)

print("All done!")
