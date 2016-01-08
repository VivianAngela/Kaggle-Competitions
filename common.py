import os

# The path of the folder where the scripts are saved
SCRIPTS_FOLDER_PATH = os.path.dirname(os.path.realpath(__file__))

# The path of the folder where the submission files are saved
SUBMISSIONS_FOLDER_NAME = "Submissions"
SUBMISSIONS_FOLDER_PATH = os.path.join(SCRIPTS_FOLDER_PATH, SUBMISSIONS_FOLDER_NAME)

# The file name of the GroundTruth. This file is saved at SUBMISSIONS_FOLDER_PATH.
GROUNDTRUTH_FILE_NAME = "GroundTruth.csv"
