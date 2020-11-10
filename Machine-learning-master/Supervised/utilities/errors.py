# errors

import numpy as np
##########################################################################################################



class CustomErrors(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return repr(self.message)


class NoObjectError(CustomErrors):
    pass


class ActiveFuncError(CustomErrors):
    pass


class DimentionError(CustomErrors):
    pass

class ShapeError(CustomErrors):
    pass


class CustomWarnings(Warning):
    def __init__(self,message):
        self.message = message
    def __str__(self):
        return repr(self.message)

class LearningRateWarning(CustomWarnings):
    pass


def NumpyErrorCheck(*args):
    for arg in args:
        if type(arg) != np.ndarray:
            raise TypeError('just numpy arrays can be passed as input')
    if args[0].ndim >= 3:
        raise DimentionError('only 2-D arrays supported')




