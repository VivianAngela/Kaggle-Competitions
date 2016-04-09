import numpy as np
import pylab

def plot_all():
    sigmoid_x = np.linspace(-10, 10, num=200)
    sigmoid_y = [1 / (1 + np.exp(-1 * value)) for value in sigmoid_x]

    tanh_x = np.linspace(-10, 10, num=200)
    tanh_y = [2 * (1 / (1 + np.exp(-1 * 2 * value))) - 1 for value in tanh_x]

    ReLU_x = np.linspace(-10, 10, num=200)
    ReLU_y = []
    for value in ReLU_x:
        if value < 0:
            ReLU_y.append(0)
        else:
            ReLU_y.append(value)

    PReLU_x = np.linspace(-10, 10, num=200)
    PReLU_y = []
    for value in PReLU_x:
        if value < 0:
            PReLU_y.append(0.3 * value)
        else:
            PReLU_y.append(value)

    pylab.figure()

    pylab.subplot(2, 2, 1)
    pylab.plot(sigmoid_x, sigmoid_y, 'yellowgreen')
    pylab.grid()
    pylab.title("Sigmoid")

    pylab.subplot(2, 2, 2)
    pylab.plot(tanh_x, tanh_y, 'yellowgreen')
    pylab.grid()
    pylab.title("Tanh")

    pylab.subplot(2, 2, 3)
    pylab.plot(ReLU_x, ReLU_y, 'yellowgreen')
    pylab.grid()
    pylab.title("ReLU")

    pylab.subplot(2, 2, 4)
    pylab.plot(PReLU_x, PReLU_y, 'yellowgreen')
    pylab.grid()
    pylab.title("PReLU")

    pylab.show()

plot_all()
