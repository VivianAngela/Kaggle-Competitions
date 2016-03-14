import numpy as np
import pylab

def conventional_AUC():
    x = np.linspace(0, 1, num=200)
    y = np.zeros(x.shape)
    for index, single_x in enumerate(x):
        y[index] = -(single_x - 1) ** 2 + 1

    pylab.figure()
    pylab.plot(x, y, "yellowgreen", label="Curve A")
    pylab.plot(1 - y, 1 - x, "lightskyblue", label="Curve B")
    pylab.gca().set_aspect("equal", adjustable="box")
    pylab.legend(loc="lower right")
    pylab.xlabel("False Positive Rate", fontsize="large")
    pylab.ylabel("True Positive Rate", fontsize="large")
    pylab.title("ROC Curve")
    pylab.show()

conventional_AUC()
