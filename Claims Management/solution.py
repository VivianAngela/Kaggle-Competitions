from fine_tune import *
import copy
import os
import preprocessing
import shutil
import time

submission_folder_path = "/tmp/submissions"

def generate_prediction(estimator, X_train, Y_train, X_test, submission_file_content, early_stopping_rounds=100, cv_num=1):
    best_iteration_list = []
    best_score_list = []

    for cv_index, _ in enumerate(range(cv_num), start=1):
        print("Working on CV {:d}/{:d} ...".format(cv_index, cv_num))

        cv_object = StratifiedKFold(Y_train, n_folds=CV_FOLD_NUM, shuffle=True)
        for cv_fold_index, (train_indexes, validate_indexes) in enumerate(cv_object, start=1):
            print("Working on fold {:d}/{:d} ...".format(cv_fold_index, CV_FOLD_NUM))

            cv_X_train, cv_X_validate = X_train[train_indexes], X_train[validate_indexes]
            cv_Y_train, cv_Y_validate = Y_train[train_indexes], Y_train[validate_indexes]
            temp_estimator = copy.deepcopy(estimator)
            temp_estimator.fit(cv_X_train, cv_Y_train, eval_set=[(cv_X_validate, cv_Y_validate)], eval_metric=EVAL_METRIC,
                          early_stopping_rounds=early_stopping_rounds, verbose=False)
            best_score = temp_estimator.best_score
            best_iteration = temp_estimator.best_iteration

            best_param = {"n_estimators":best_iteration}
            temp_estimator = copy.deepcopy(estimator)
            temp_estimator.set_params(**best_param)
            temp_estimator.fit(cv_X_train, cv_Y_train, verbose=False)

            proba = temp_estimator.predict_proba(X_test)
            prediction = proba[:, 1]

            submission_file_name = "Aurora_{:.4f}_{:d}.csv".format(best_score, int(time.time()))
            submission_file_content[preprocessing.LABEL_COLUMN_NAME_IN_SUBMISSION] = prediction
            submission_file_content.to_csv(os.path.join(submission_folder_path, submission_file_name), index=False)

            best_iteration_list.append(best_iteration)
            best_score_list.append(best_score)
            print("The best score {:.4f} is obtained at iteration {:d}.".format(best_score, best_iteration))

    print("The median best_iteration={:d} and best_score={:.4f}.".format(np.int(np.median(best_iteration_list)), np.median(best_score_list)))
    return np.int(np.median(best_iteration_list)), np.median(best_score_list)

def run():
    print("Resetting the submission folder {:s} ...".format(os.path.basename(submission_folder_path)))
    shutil.rmtree(submission_folder_path, ignore_errors=True)
    os.makedirs(submission_folder_path)

    print("Loading data ...")
    X_train, Y_train, X_test, submission_file_content = preprocessing.load_data()

    print("Tuning parameters ...")
    optimal_max_depth, optimal_min_child_weight, optimal_subsample, optimal_colsample_bytree = perform_tuning(X_train, Y_train)

    print("Training ...")
    optimal_learning_rate = 0.05
    estimator = XGBClassifier(max_depth=optimal_max_depth, learning_rate=optimal_learning_rate, n_estimators=1000000,
                              min_child_weight=optimal_min_child_weight, subsample=optimal_subsample,
                              colsample_bytree=optimal_colsample_bytree, objective=OBJECTIVE)
    generate_prediction(estimator, X_train, Y_train, X_test, submission_file_content, early_stopping_rounds=200, cv_num=20)

    print("All done!")

if __name__ == "__main__":
    run()
