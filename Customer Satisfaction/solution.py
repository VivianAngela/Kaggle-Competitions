import XGBoost
import file_operations
import os
import shutil

SUBMISSION_FOLDER_PATH = "/tmp/submissions"

def run():
    print("Resetting the submission folder {:s} ...".format(os.path.basename(SUBMISSION_FOLDER_PATH)))
    shutil.rmtree(SUBMISSION_FOLDER_PATH, ignore_errors=True)
    os.makedirs(SUBMISSION_FOLDER_PATH)

    print("Loading data ...")
    X_train, Y_train, X_test, ID_test = file_operations.load_data()

    print("Performing tuning ...")
    optimal_parameters = XGBoost.perform_tuning(X_train, Y_train)

    print("Generating predictions ...")
    prediction_num = 100
    for prediction_index in range(1, prediction_num + 1):
        print("Working on prediction {:d}/{:d} ...".format(prediction_index, prediction_num))
        score, prediction = XGBoost.generate_prediction(X_train, Y_train, X_test, optimal_parameters, random_state=prediction_index)

        submission_file_name = "Aurora_{:.4f}_{:d}.csv".format(score, prediction_index)
        submission_file_path = os.path.join(SUBMISSION_FOLDER_PATH, submission_file_name)
        file_operations.write_submission(ID_test, prediction, submission_file_path)

    print("All done!")

if __name__ == "__main__":
    run()
