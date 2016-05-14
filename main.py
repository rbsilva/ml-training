import numpy as np
import pylab
import matplotlib.pyplot as plt
from numpy import genfromtxt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def cost(theta, x, y, lamb):
    m = x.shape[0] #number of samples
    J = 0 # history
    theta1 = np.copy(theta)
    theta1[0] = np.zeros(theta.shape[1])
    grad = np.zeros(theta.shape)
    h = sigmoid(np.dot(x, theta))
    p = lamb * np.dot(theta1.T, theta1) / 2 * m
    J = (np.dot((-y).T, np.log(h)) - np.dot((1-y).T, np.log(1-h))) / m + p

    grad = (np.dot(x.T, (h - y)) + np.dot(lamb, theta1)) / m

    return J, grad

def gradient_descent(alpha, x, y, numIterations):
    m = x.shape[0] # number of samples
    n = x.shape[1] # number of features
    theta = np.ones((n, 1))

    for iter in range(0, numIterations):
        J, grad = cost(theta, x, y, 400)  # cost

        print "iter %s | J: %.3f" % (iter, J)      

        theta = theta - np.dot(alpha, grad)  # update

    return theta

def predict(a, all_theta):
    print all_theta
    matrix = np.dot(a, all_theta.T)
    results = sigmoid(matrix)
    return results

if __name__ == '__main__':

    csv = genfromtxt('iris.data.txt', delimiter=',')
    x = csv[:, [0,1,2,3]]
    y = csv[:, [-1]]
    y = np.resize(y, (y.shape[0], 3))
    for value in np.nditer(y, op_flags=['readwrite']):
        if value == 0:
            y[...] = [1,0,0]
        if value == 1:
            y[...] = [0,1,0]
        if value == 2:
            y[...] = [0,0,1]

    print y
    all_theta = np.zeros((y.shape[1], x.shape[1]))
    alpha = 0.01 # learning rate
    np.insert(x, 0, np.ones(x.shape[1]), axis=0)

    for i in range(0, y.shape[1]):
        all_theta[i] = gradient_descent(alpha, x, y[:,[i]], 1).T

    print predict([5.9,3.0,5.1,1.8], all_theta)

    # plot
    plot_x = [np.min(x[1,:])-2,  np.max(x[1,:])+2]
    plot_y = np.dot(np.divide(-1, all_theta[0][2]), (np.dot(all_theta[0][1], plot_x) + all_theta[0][0]))
    pylab.plot(plot_x, plot_y)
    pylab.plot(x, y, 'o')
    pylab.show()
    labels ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    print "Done!"