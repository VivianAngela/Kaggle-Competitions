import numpy as np
import pylab

def weighted_AUC(weight_distribution):
    weight_distribution = weight_distribution / np.mean(weight_distribution)
    threshold_array, step = np.linspace(0, 1, num=weight_distribution.size + 1, retstep=True)
    threshold_array = threshold_array[1:-1]

    x = np.linspace(0, 1, num=200)
    y = np.zeros(x.shape)
    for index, single_x in enumerate(x):
        y[index] = -(single_x - 1) ** 2 + 1

    pylab.figure()
    pylab.plot(x, y, "yellowgreen", label="ROC Curve")

    for threshold in threshold_array:
        pylab.plot(x, np.ones(x.size) * threshold, "lightskyblue")

    text_y_loc_array = threshold_array - step / 1.2
    text_y_loc_array = np.hstack([text_y_loc_array, text_y_loc_array[-1] + step])
    for text_y_loc, weight in zip(text_y_loc_array, weight_distribution):
        pylab.text(.7, text_y_loc, "Weight = {:.1f}".format(weight))

    pylab.xlim([0, 1])
    pylab.ylim([0, 1])
    pylab.gca().set_aspect("equal", adjustable="box")
    pylab.xlabel("False Positive Rate", fontsize="large")
    pylab.ylabel("True Positive Rate", fontsize="large")
    pylab.title("ROC Curve with Specific Weight Distribution")
    pylab.show()

weighted_AUC(np.arange(0, 5, 1))
weighted_AUC(np.arange(4, -1, -1))
