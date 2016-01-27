import numpy as np
import pylab

def conventional_AUC():
    x = np.linspace(0, 1, num=200)
    y = np.zeros(x.shape)
    for index, single_x in enumerate(x):
        y[index] = -(single_x - 1) ** 2 + 1

    pylab.figure()
    pylab.plot(x, y, "b", label="Curve A")
    pylab.plot(1 - y, 1 - x, "r", label="Curve B")
    pylab.legend(loc="lower right")
    pylab.xlabel("False Positive Rate")
    pylab.ylabel("True Positive Rate")
    pylab.title("ROC Curve")
    pylab.show()

conventional_AUC()
