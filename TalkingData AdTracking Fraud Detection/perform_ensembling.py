import os
import glob
import shutil
import datetime
import numpy as np
import pandas as pd

# Dataset
PROJECT_NAME = "TalkingData AdTracking Fraud Detection"
PROJECT_FOLDER_PATH = os.path.join(os.path.expanduser("~"), "Documents/Dataset", PROJECT_NAME)

# Submission
TEAM_NAME = "Aurora"
SUBMISSION_FOLDER_PATH = os.path.join(PROJECT_FOLDER_PATH, "submission")
os.makedirs(SUBMISSION_FOLDER_PATH, exist_ok=True)

# Ensembling
WORKSPACE_FOLDER_PATH = os.path.join(PROJECT_FOLDER_PATH, "script/Mar_25_3")
KEYWORD = "DL"

# Generate a zip archive for a file
create_zip_archive = lambda file_path: shutil.make_archive(file_path[:file_path.rindex(".")], "zip", os.path.abspath(os.path.join(file_path, "..")), os.path.basename(file_path))

def run():
    print("Searching for submissions with keyword {} at {} ...".format(KEYWORD, WORKSPACE_FOLDER_PATH))
    submission_file_path_list = sorted(glob.glob(os.path.join(WORKSPACE_FOLDER_PATH, "*{}*".format(KEYWORD))))
    assert len(submission_file_path_list) != 0

    ranking_array_list = []
    for submission_file_path in submission_file_path_list:
        print("Loading {} ...".format(submission_file_path))
        submission_df = pd.read_csv(submission_file_path)

        print("Ranking the entries ...")
        index_series = submission_df["is_attributed"].argsort()
        ranking_array = np.zeros(index_series.shape, dtype=np.uint32)
        ranking_array[index_series] = np.arange(len(index_series))
        ranking_array_list.append(ranking_array)

    ensemble_df = submission_df.copy()
    ensemble_prediction_array = np.mean(ranking_array_list, axis=0)
    apply_normalization = lambda data_array: 1.0 * (data_array - np.min(data_array)) / (np.max(data_array) - np.min(data_array))
    ensemble_df["is_attributed"] = apply_normalization(ensemble_prediction_array)
    ensemble_file_path = os.path.join(SUBMISSION_FOLDER_PATH, "{} {} {}.csv".format(TEAM_NAME, KEYWORD, str(datetime.datetime.now()).split(".")[0]).replace(" ", "_"))
    print("Saving submission to {} ...".format(ensemble_file_path))
    ensemble_df.to_csv(ensemble_file_path, index=False)
    compressed_ensemble_file_path = create_zip_archive(ensemble_file_path)
    print("Saving compressed submission to {} ...".format(compressed_ensemble_file_path))

    print("All done!")

if __name__ == "__main__":
    run()
