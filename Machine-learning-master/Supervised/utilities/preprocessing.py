# preprocessing

import numpy as np
import errors as e
#########################################################################################################


def add_bias_coef(x):
    '''
    standardize input to a computable matrix by checking and adding bias coefs.

    Parameters
    ----------
    x : ndarray
        features matrix.

    Returns
    -------
    ndarray
        An standard array to compute.
    '''

    if x.ndim == 1:
        x = x[:, np.newaxis]

    if (x.ndim == 2) and (not(np.all(x[:, 0] == 1))):
        tempArray = np.ones([x.shape[0], x.shape[-1]+1])
        tempArray[:, 1:] = x
        x = tempArray
    return x


def standardization(x,just_mean = False):
    '''
    converts input to to a new array with these features : {mean = 0 and standard deviation = 1}

    Parameters
    ----------
    x : ndarray
        features matrix.
    just_mean : bool , False
        if True , it will only reduce the meaning of the data.
    
    Returns
    -------
    ndarray
    '''
    e.NumpyErrorCheck(x)
    if just_mean:
        return  (x - np.mean(x))
    return (x - np.mean(x)) / (np.std(x))

def normalization(x,mean_norm=False):
    '''
    normalization thechnique for feature scaling. contains both min/max scaling and mean norm scaling

    parameters
    ----------
    x : ndarray
        matrix to ba scaled
    mean_norm : bool , optional
        default = False
        if True , mean norm will be performed to x.
    
    returns
    --------
    ndarray
    '''
    if mean_norm:
        return ((x - np.mean(x)) / (np.max(x) - np.min(x)))
    return (x - np.min(x)) / (np.max(x) - np.min(x))



def split_train_test(x, y, train_rate=.8,shuffle=False):
    '''
        Split arrays or matrices into random train and test subsets

    Parameters
    ----------
    x : ndarray
        features matrix.
    y : ndarray
        target matrix.
    train_rate : float , optional
        ratio of the train to the test
    shuffle : bool , optional
        default = False
        if True , data will shuffled.

    Returns
    -------
    ndarray
        four arrays in order : {x-train , x-test, y-train, y-test}

    '''
    if shuffle:
        _t = np.concatenate((x,y),axis=1)
        np.random.shuffle(_t)
        x = _t[:,:-1]
        y = _t[:,-1].reshape(-1,1)
        del _t
    if train_rate < 0 or train_rate > 1:
        raise ValueError('train rate must be between 0 and 1')
    e.NumpyErrorCheck(x, y)
    idx = int(train_rate * len(x))
    def train(A): return A[0: idx]
    def test(B): return B[idx:]
    return (train(x), test(x), train(y), test(y))