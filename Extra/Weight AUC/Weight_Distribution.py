import numpy as np
import pylab

def illustrate_weight_distribution():
    x = np.linspace(0, 1, num=1000, endpoint=False)

    weight_distribution = np.arange(0, 4 + 1, 1.0)
    y_1 = np.zeros(x.shape)
    for index, single_x in enumerate(x):
        y_1[index] = weight_distribution[int(single_x / (1.0 / weight_distribution.size))]

    weight_distribution = np.arange(4, -1, -1.0)
    y_2 = np.zeros(x.shape)
    for index, single_x in enumerate(x):
        y_2[index] = weight_distribution[int(single_x / (1.0 / weight_distribution.size))]

    pylab.figure()
    pylab.plot(x, y_1, "yellowgreen", label="Weight Distribution A")
    pylab.plot(x, y_2, "lightskyblue", label="Weight Distribution B")
    pylab.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    pylab.xlabel("True Positive Rate", fontsize="large")
    pylab.ylabel("Weight", fontsize="large")
    pylab.show()

illustrate_weight_distribution()
