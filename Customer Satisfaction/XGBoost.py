from copy import deepcopy
from sklearn.cross_validation import StratifiedKFold, StratifiedShuffleSplit
from xgboost.sklearn import XGBClassifier
import numpy as np

N_FOLDS = 5
EVAL_METRIC = "auc"
EARLY_STOPPING_ROUNDS = 100
N_ESTIMATORS = 1000000
OBJECTIVE = "binary:logistic"
GET_BEST_SCORE_INDEX = np.argmax
OPTIMAL_LEARNING_RATE = 0.01

def evaluate_estimator(estimator, X_train, Y_train):
    best_score_list = []
    cv_object = StratifiedKFold(Y_train, n_folds=N_FOLDS, shuffle=True, random_state=0)
    for train_indexes, validate_indexes in cv_object:
        cv_X_train, cv_X_validate = X_train[train_indexes], X_train[validate_indexes]
        cv_Y_train, cv_Y_validate = Y_train[train_indexes], Y_train[validate_indexes]
        estimator.fit(cv_X_train, cv_Y_train, eval_set=[(cv_X_validate, cv_Y_validate)], eval_metric=EVAL_METRIC,
                      early_stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)
        best_score_list.append(estimator.best_score)

    print("The median best_score is {:.4f}.".format(np.median(best_score_list)))
    return np.median(best_score_list)

def perform_tuning(X_train, Y_train):
    print("Tuning max_depth ...")
    max_depth_search_space = np.linspace(4, 14, num=6, dtype=np.int)
    best_score_list = []
    for max_depth in max_depth_search_space:
        estimator = XGBClassifier(max_depth=max_depth, learning_rate=0.1, n_estimators=N_ESTIMATORS,
                                  min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                                  objective=OBJECTIVE)
        best_score = evaluate_estimator(estimator, X_train, Y_train)
        best_score_list.append(best_score)
    best_score_index = GET_BEST_SCORE_INDEX(best_score_list)
    optimal_max_depth = max_depth_search_space[best_score_index]
    print("The optimal max_depth is {:d}.".format(optimal_max_depth))

    print("Tuning subsample ...")
    subsample_search_space = np.linspace(0.6, 1, num=5)
    best_score_list = []
    for subsample in subsample_search_space:
        estimator = XGBClassifier(max_depth=optimal_max_depth, learning_rate=0.1, n_estimators=N_ESTIMATORS,
                                  min_child_weight=5, subsample=subsample, colsample_bytree=0.8,
                                  objective=OBJECTIVE)
        best_score = evaluate_estimator(estimator, X_train, Y_train)
        best_score_list.append(best_score)
    best_score_index = GET_BEST_SCORE_INDEX(best_score_list)
    optimal_subsample = subsample_search_space[best_score_index]
    print("The optimal subsample is {:f}.".format(optimal_subsample))

    print("Tuning min_child_weight ...")
    min_child_weight_search_space = np.linspace(1, 9, num=5)
    best_score_list = []
    for min_child_weight in min_child_weight_search_space:
        estimator = XGBClassifier(max_depth=optimal_max_depth, learning_rate=0.1, n_estimators=N_ESTIMATORS,
                                  min_child_weight=min_child_weight, subsample=optimal_subsample,
                                  colsample_bytree=0.8, objective=OBJECTIVE)
        best_score = evaluate_estimator(estimator, X_train, Y_train)
        best_score_list.append(best_score)
    best_score_index = GET_BEST_SCORE_INDEX(best_score_list)
    optimal_min_child_weight = min_child_weight_search_space[best_score_index]
    print("The optimal min_child_weight is {:f}.".format(optimal_min_child_weight))

    print("Tuning colsample_bytree ...")
    colsample_bytree_search_space = np.linspace(0.6, 1, num=5)
    best_score_list = []
    for colsample_bytree in colsample_bytree_search_space:
        estimator = XGBClassifier(max_depth=optimal_max_depth, learning_rate=0.1, n_estimators=N_ESTIMATORS,
                                  min_child_weight=optimal_min_child_weight, subsample=optimal_subsample,
                                  colsample_bytree=colsample_bytree, objective=OBJECTIVE)
        best_score = evaluate_estimator(estimator, X_train, Y_train)
        best_score_list.append(best_score)
    best_score_index = GET_BEST_SCORE_INDEX(best_score_list)
    optimal_colsample_bytree = colsample_bytree_search_space[best_score_index]
    print("The optimal colsample_bytree is {:f}.".format(optimal_colsample_bytree))

    optimal_parameters = [optimal_max_depth, optimal_min_child_weight, optimal_subsample, optimal_colsample_bytree]
    print("The optimal parameters are as follows:")
    print(optimal_parameters)
    return optimal_parameters

def generate_prediction(X_train, Y_train, X_test, optimal_parameters, train_size=0.8, random_state=0):
    optimal_max_depth, optimal_min_child_weight, optimal_subsample, optimal_colsample_bytree = optimal_parameters
    optimal_estimator = XGBClassifier(max_depth=optimal_max_depth, learning_rate=OPTIMAL_LEARNING_RATE, n_estimators=N_ESTIMATORS,
                                      min_child_weight=optimal_min_child_weight, subsample=optimal_subsample,
                                      colsample_bytree=optimal_colsample_bytree, objective=OBJECTIVE)

    cv_object = StratifiedShuffleSplit(Y_train, n_iter=1, train_size=train_size, random_state=random_state)
    for train_indexes, validate_indexes in cv_object:
        cv_X_train, cv_X_validate = X_train[train_indexes], X_train[validate_indexes]
        cv_Y_train, cv_Y_validate = Y_train[train_indexes], Y_train[validate_indexes]

        estimator = deepcopy(optimal_estimator)
        estimator.fit(cv_X_train, cv_Y_train, eval_set=[(cv_X_validate, cv_Y_validate)], eval_metric=EVAL_METRIC,
                      early_stopping_rounds=2 * EARLY_STOPPING_ROUNDS, verbose=False)
        best_score = estimator.best_score
        best_iteration = estimator.best_iteration

        best_param = {"n_estimators":best_iteration}
        estimator = deepcopy(optimal_estimator)
        estimator.set_params(**best_param)
        estimator.fit(cv_X_train, cv_Y_train)

        proba = estimator.predict_proba(X_test)
        return best_score, proba[:, 1]
