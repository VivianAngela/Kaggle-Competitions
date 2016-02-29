from keras.callbacks import Callback
from keras.layers.advanced_activations import PReLU
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.utils import np_utils
import evaluation
import numpy as np
import os

def init_model(dimension, unique_label_num=2):
    """Init a keras model which could be found in 
        https://github.com/fchollet/keras/blob/master/examples/kaggle_otto_nn.py.
    
    :param dimension: the dimension of the features
    :type dimension: int
    :param unique_label_num: the number of unique labels
    :type unique_label_num: int
    :return: the keras model
    :rtype: object
    """

    model = Sequential()

    model.add(Dense(32, input_shape=(dimension,)))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(32))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(32))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(unique_label_num))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer="adam")
    return model

class Customized_Callback(Callback):
    """Customized Callback. The code is inspired by the definition of ModelCheckpoint.
    The model file will be updated only if the new coefficients achieve higher score.
    """

    def __init__(self, model_path, X_test, Y_test, monitor="score"):
        """Init function.
        
        :param model_path: the path of the model file
        :type model_path: string
        :param X_test: the testing attributes
        :type X_test: numpy array
        :param Y_test: the testing labels
        :type Y_test: numpy array
        :param monitor: the name of the error metric
        :type monitor: string
        :return: the class object will be initiated based on the arguments
        :rtype: None
        """

        super(Callback, self).__init__()
        self.monitor = monitor
        self.model_path = model_path
        self.best_score_index = None
        self.best_score = -np.Inf
        self.X_test = X_test
        self.Y_test = Y_test

    def on_epoch_end(self, epoch, logs={}):
        """This function will be called after each epoch.
        
        :param epoch: the index of the epoch
        :type epoch: int
        :param logs: contain keys for quantities relevant to the current batch or epoch
        :type logs: dict
        :return: the model file will be updated if necessary
        :rtype: None
        """

        model_path = self.model_path.format(epoch=epoch, **logs)

        probability_estimates = self.model.predict_proba(self.X_test, verbose=0)
        prediction = probability_estimates[:, 1]
        score = evaluation.compute_Weighted_AUC(self.Y_test, prediction)

        if self.best_score < 0 or score > self.best_score:
            print("In epoch {:05d}: {} improved from {:.4f} to {:.4f}, saving model to {}.".format(\
                    epoch + 1, self.monitor, self.best_score, score, os.path.basename(model_path)))
            self.best_score_index = epoch + 1
            self.best_score = score
            self.model.save_weights(model_path, overwrite=True)
        else:
            pass

    def inspect_details(self):
        """Inspect the details of the training phase.
        
        :return: best_score_index refers to the index of the epoch, 
            while best_score refers to the highest score
        :rtype: tuple
        """

        return (self.best_score_index, self.best_score)

def train_model(X_train, Y_train, X_test, Y_test, model_path, nb_epoch):
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
    :param nb_epoch: the maximum number of epochs
    :type nb_epoch: int
    :return: best_score_index refers to the index of the epoch, 
        while best_score refers to the highest score
    :rtype: tuple
    """

    # Init a keras model
    dimension = X_train.shape[1]
    unique_label_num = np.size(np.unique(Y_train))
    model = init_model(dimension, unique_label_num)

    # Start the training phase
    customized_callback = Customized_Callback(model_path=model_path, X_test=X_test, Y_test=Y_test)
    categorical_Y_train = np_utils.to_categorical(Y_train, unique_label_num)
    model.fit(X_train, categorical_Y_train, batch_size=32, nb_epoch=nb_epoch, verbose=0, callbacks=[customized_callback])

    return customized_callback.inspect_details()
