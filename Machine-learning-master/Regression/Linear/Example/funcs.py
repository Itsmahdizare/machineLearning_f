#packages
import pandas as pd
from matplotlib import pyplot as plt,numpy as np


#algorithms
def standardize_X(x):
    '''
    To standardize input to a computable matrix by checking and adding bias coefs.

    Parameters
    ----------
    x : ndarray
        Input matrix.

    Returns
    -------
    ndarray
        An standard array to compute.
    '''
    if type(x)!=np.ndarray:
        raise TypeError('x must be a numpy 2D array')
    if x.ndim == 1:
        x = x[:, np.newaxis]

    if not(np.all(x[:, 0] == 1)):
        tempArray = np.ones([x.shape[0], x.shape[1]+1])
        tempArray[:, 1:] = x
        x = tempArray
    return x


def Hyp(x, w0, degree=1, activationFunction='identity'):
    '''
    The hypothesis function for linear regression.

    Parameters
    ----------
    x : ndarray
        Input matrix.
    w0 : ndarray
        weights matrix.
    degree : int, optional
        the degree of weights.
        use degree 2 or more for polynomial regression. (the default is 1 )
    activationFunction : string, optional
        order : {'identity', 'sigmoid'}
        The identity function is the default function as linear regression models. however, it can be some others for classification problems.

    Returns
    -------
    ndarray
        hypothesis for each data.
    '''
    # set activation function
    if activationFunction == 'identity':
        def activeF(x): return x
    elif activationFunction == 'sigmoid':
        def activeF(x): return 1 / (1 + np.exp(-x))
    else:
        raise ValueError(
            'The activation function must be one of these : \nidentity\nsigmoid')

    x = standardize_X(x)
    hyps = np.zeros((x.shape[0], 1))
    for deg in range(1, degree+1):
        hyp = activeF(np.power(x, deg) @ w0)
        hyps += hyp
    return hyps


def loss_MSE(x, y, w0, m='mean'):
    '''
    Mean sqaured errors approach to compute loss

    Parameters
    ----------
    x : ndarray
        Input matrix.
    y : ndarray
        results matrix.
    w0 : ndarray
        weights matrix
    m : int, optional
        Usually number of the data. However, you may enter your custom number. 

    Returns
    -------
    ndarray
    '''
    if m == 'mean':
        m = len(y)
    if type(m) != int:
        raise TypeError(''' m must be an integer.''')
    x = standardize_X(x)
    return 1/m * np.sum(np.square((Hyp(x, w0) - y)))


def loss_vec(x, y, w0, m='mean'):
    '''
    Mean sqaured errors approach to compute loss

    Parameters
    ----------
    x : ndarray
        Input matrix.
    y : ndarray
        results matrix.
    w0 : ndarray
        weights matrix
    m : int, optional
        Usually number of the data. However, you may enter your custom number. 

    Returns
    -------
    ndarray
    '''
    if m == 'mean':
        m = len(y)
    if type(m) != int:
        raise TypeError(''' m must be an integer.''')
    x = standardize_X(x)
    return 1/m * (((Hyp(x, w0) - y).T @ (Hyp(x, w0) - y))[0, 0])


def gradient_descent(x, y, w0, m='mean', learning_rate=.00001, max_iter=100, convergenceLimit=0.001, verbose=False, plot=False):
    '''
    Gradient descent algorithm for linear regression.

    Parameters
    ----------
    x : ndarray
        Input matrix.
    y : ndarray
        results matrix.
    w0 : ndarray
        weights matrix
    learning_rate : float
        alpha hyperparameter.
    max_iter : int, optional
        maximum number of iterations.
    convergenceLimit : float, optional
        set where to stop the algorithm by convergence limitation
    bias : bool, optional
        if not true, a column of bias coefficients will be added to x. (default is false)
    verbose: bool, optional
        if true, each iteration details will be shown.
    plot: bool, optional
        to plot cost function of each iteration

    Returns
    -------
    ndarray, int
    optimum weights matrix, loss of new weights.
    '''

    # variables

    costs = [9999, 99999]
    iterNum = 0
    # standardize input
    x = standardize_X(x)
    # other conditions:
    if m == 'mean':
        m = len(y)
    elif type(m) != int:
        raise TypeError(''' m must be an integer.''')
    if learning_rate < 0:
        raise ValueError('Learning rate must be greater than 0')
    # starting the algorithm
    if not verbose:
        while iterNum != max_iter:
            errors = (Hyp(x, w0) - y)
            grad = (1/m) * (np.dot(x.T, errors))
            w0 = w0 - (learning_rate * grad)
            cost = 1/m * np.sum(np.square(errors))
            costs.append(cost)
            iterNum += 1
            if abs(costs[-1] - costs[-2]) < convergenceLimit:
                print('End of the algorithm at the iteration number {}.\nThe differences in costs was less than {}'.format(
                    iterNum, convergenceLimit))
                break
        if plot:
            plt.plot(costs[2:], color='red')
            plt.xlabel('iteration number')
            plt.ylabel('cost')
            plt.show()
        return w0, costs[-1]
    else:
        for iter in range(max_iter):
            errors = (Hyp(x, w0) - y)
            grad = np.dot(x.T, errors)
            w0 = w0 - (1/m * learning_rate * grad)
            cost = 1/m * np.sum(np.square(errors))
            costs.append(cost)
            iterNum += 1
            print('\n\niteration no. {}\nbias = {}\ncoefficients = {}\nloss = {}\n'.format(
                iterNum, w0[0], w0[1:], float(cost)))
            if abs(costs[-1] - costs[-2]) < convergenceLimit:
                print('\nEnd of the algorithm.\nThe differences in costs was less than {}\n'.format(
                    convergenceLimit))
                break
        if plot:
            plt.plot(costs[2:], color='red')
            plt.xlabel('iteration number')
            plt.ylabel('cost')
            plt.show()
        return w0, costs[-1]


def normal_equation(x, y, w0):
    '''
    normal equation for linear models.

    Parameters
    ----------
    x : ndarray
        Input matrix.
    y : ndarray
        results matrix.
    w0 : ndarray
        weights matrix
    Returns
    -------
    ndarray
        optimum weiths matrix
    '''
    x = standardize_X(x)
    return (np.linalg.pinv(x.T @ x) @ x.T @ y)