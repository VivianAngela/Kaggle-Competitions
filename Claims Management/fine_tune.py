from sklearn.cross_validation import StratifiedKFold
from xgboost.sklearn import XGBClassifier
import numpy as np

np.random.seed(666)

OBJECTIVE = "binary:logistic"
EVAL_METRIC = "logloss"
SCORING = "log_loss"
GET_BEST_SCORE_INDEX = np.argmin
CV_FOLD_NUM = 5

def evaluate_estimator(estimator, X_train, Y_train, early_stopping_rounds=100, cv_num=1):
    best_iteration_list = []
    best_score_list = []

    for cv_index, _ in enumerate(range(cv_num), start=1):
        print("Working on CV {:d}/{:d} ...".format(cv_index, cv_num))

        cv_object = StratifiedKFold(Y_train, n_folds=CV_FOLD_NUM, shuffle=True)
        for cv_fold_index, (train_indexes, validate_indexes) in enumerate(cv_object, start=1):
            print("Working on fold {:d}/{:d} ...".format(cv_fold_index, CV_FOLD_NUM))
            cv_X_train, cv_X_validate = X_train[train_indexes], X_train[validate_indexes]
            cv_Y_train, cv_Y_validate = Y_train[train_indexes], Y_train[validate_indexes]
            estimator.fit(cv_X_train, cv_Y_train, eval_set=[(cv_X_validate, cv_Y_validate)], eval_metric=EVAL_METRIC,
                          early_stopping_rounds=early_stopping_rounds, verbose=False)
            best_iteration_list.append(estimator.best_iteration)
            best_score_list.append(estimator.best_score)
            print("The best score {:.4f} is obtained at iteration {:d}.".format(estimator.best_score, estimator.best_iteration))

    print("The median best_iteration={:d} and best_score={:.4f}.".format(np.int(np.median(best_iteration_list)), np.median(best_score_list)))
    return np.int(np.median(best_iteration_list)), np.median(best_score_list)

def perform_tuning(X_train, Y_train):
    print("Tuning max_depth ...")
    max_depth_search_space = np.linspace(4, 14, num=6, dtype=np.int)
    best_score_list = []
    for max_depth in max_depth_search_space:
        estimator = XGBClassifier(max_depth=max_depth, learning_rate=0.1, n_estimators=1000000,
                                  min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                                  objective=OBJECTIVE)
        _, best_score = evaluate_estimator(estimator, X_train, Y_train)
        best_score_list.append(best_score)
    best_score_index = GET_BEST_SCORE_INDEX(best_score_list)
    optimal_max_depth = max_depth_search_space[best_score_index]
    print("The optimal max_depth is {:d}.".format(optimal_max_depth))

    print("Tuning subsample ...")
    subsample_search_space = np.linspace(0.6, 1, num=5)
    best_score_list = []
    for subsample in subsample_search_space:
        estimator = XGBClassifier(max_depth=optimal_max_depth, learning_rate=0.1, n_estimators=1000000,
                                  min_child_weight=5, subsample=subsample, colsample_bytree=0.8,
                                  objective=OBJECTIVE)
        _, best_score = evaluate_estimator(estimator, X_train, Y_train)
        best_score_list.append(best_score)
    best_score_index = GET_BEST_SCORE_INDEX(best_score_list)
    optimal_subsample = subsample_search_space[best_score_index]
    print("The optimal subsample is {:f}.".format(optimal_subsample))

    print("Tuning min_child_weight ...")
    min_child_weight_search_space = np.linspace(1, 9, num=5)
    best_score_list = []
    for min_child_weight in min_child_weight_search_space:
        estimator = XGBClassifier(max_depth=optimal_max_depth, learning_rate=0.1, n_estimators=1000000,
                                  min_child_weight=min_child_weight, subsample=optimal_subsample,
                                  colsample_bytree=0.8, objective=OBJECTIVE)
        _, best_score = evaluate_estimator(estimator, X_train, Y_train)
        best_score_list.append(best_score)
    best_score_index = GET_BEST_SCORE_INDEX(best_score_list)
    optimal_min_child_weight = min_child_weight_search_space[best_score_index]
    print("The optimal min_child_weight is {:f}.".format(optimal_min_child_weight))

    print("Tuning colsample_bytree ...")
    colsample_bytree_search_space = np.linspace(0.6, 1, num=5)
    best_score_list = []
    for colsample_bytree in colsample_bytree_search_space:
        estimator = XGBClassifier(max_depth=optimal_max_depth, learning_rate=0.1, n_estimators=1000000,
                                  min_child_weight=optimal_min_child_weight, subsample=optimal_subsample,
                                  colsample_bytree=colsample_bytree, objective=OBJECTIVE)
        _, best_score = evaluate_estimator(estimator, X_train, Y_train)
        best_score_list.append(best_score)
    best_score_index = GET_BEST_SCORE_INDEX(best_score_list)
    optimal_colsample_bytree = colsample_bytree_search_space[best_score_index]
    print("The optimal colsample_bytree is {:f}.".format(optimal_colsample_bytree))

    optimal_parameters = [optimal_max_depth, optimal_min_child_weight, optimal_subsample, optimal_colsample_bytree]
    print("The optimal parameters are as follows:")
    print(optimal_parameters)
    return optimal_parameters
