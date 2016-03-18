from sklearn.externals import joblib
from sklearn.grid_search import ParameterGrid
from sklearn.svm import SVC
import evaluation
import numpy as np
import os

def train_model(X_train, Y_train, X_test, Y_test, model_path):
    """Training phase.
    
    :param X_train: the training attributes
    :type X_train: numpy array
    :param Y_train: the training labels
    :type Y_train: numpy array
    :param X_test: the testing attributes
    :type X_test: numpy array
    :param Y_test: the testing labels
    :type Y_test: numpy array
    :param model_path: the path of the model file
    :type model_path: string
    :return: best_score refers to the highest score
    :rtype: float
    """

    # Set the parameters for SVC
    param_grid = [{"C": [1, 10, 100, 1000], "gamma": ["auto"], "kernel": ["linear"]}, \
                  {"C": [1, 10, 100, 1000], "gamma": [0.001, 0.0001], "kernel": ["rbf"]}]
    parameters_combinations = ParameterGrid(param_grid)

    # Get a list of different classifiers
    unique_classifier_list = []
    for current_parameters in parameters_combinations:
        current_classifier = SVC(C=current_parameters["C"],
                                 kernel=current_parameters["kernel"],
                                 gamma=current_parameters["gamma"],
                                 probability=True)
        unique_classifier_list.append(current_classifier)

    # Loop through the classifiers
    best_score = -np.Inf
    for classifier_index, classifier in enumerate(unique_classifier_list):
        classifier.fit(X_train, Y_train)
        probability_estimates = classifier.predict_proba(X_test)
        prediction = probability_estimates[:, 1]
        score = evaluation.compute_Weighted_AUC(Y_test, prediction)
        print("Classifier {:d} achieved {:.4f}.".format(classifier_index, score))

        if best_score < 0 or score > best_score:
            print("Score improved from {:.4f} to {:.4f}, saving model to {}.".format(\
                    best_score, score, os.path.basename(model_path)))
            best_score = score
            joblib.dump(classifier, model_path)

    return best_score
