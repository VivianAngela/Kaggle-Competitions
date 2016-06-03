import keras_NN
import numpy as np
import pandas as pd
import time

# Read data set from file
X_train = pd.read_csv("./input/X_train.csv", skiprows=0).as_matrix()[:, 1:]
multi_Y_train = pd.read_csv("./input/y_train.csv", skiprows=0).as_matrix()[:, 1:]
X_test = pd.read_csv("./input/X_test.csv", skiprows=0).as_matrix()[:, 1:]
multi_Y_test = []

# Generate prediction for each angle
for current_column in range(multi_Y_train.shape[1]):
    Y_train = multi_Y_train[:, current_column]
    prediction = keras_NN.generate_prediction(X_train, Y_train, X_test, True, layer_size=512, layer_num=5, nb_epoch=100)
    multi_Y_test.append(np.reshape(prediction, (-1, 1)))

# Create submission file
multi_Y_test = np.hstack(multi_Y_test)
ID = np.arange(X_train.shape[0] + 1, X_train.shape[0] + multi_Y_test.shape[0] + 1)
submission_file_name = "Aurora_" + str(int(time.time())) + ".csv"
submission_file_DataFrame = pd.DataFrame({"Id": ID, "Angle1": multi_Y_test[:, 0], "Angle2": multi_Y_test[:, 1]})
submission_file_DataFrame.to_csv(submission_file_name, index=False, header=True)

print("All done!")
