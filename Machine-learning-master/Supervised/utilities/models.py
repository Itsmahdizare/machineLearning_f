# algorithms
import warnings
import errors as e
import preprocessing as prc
import numpy as np
import matplotlib.pyplot as plt
#########################################################################################################


class SupervisedModel():
    '''
    the class "supervised model" supports some supervised learning model algorithms, such as
    hypothesis,cost and optimzation.
    
    parameters
    ----------
    x : ndarray
        feature matrix
    y : ndarray
        target matrix
    w0 : ndarray
        initial weight matrix
    m : int
        an integer number which is usually length of the given data
    tag : str , optional
        order : {LinearRegression , LogisticRegression}
        to determine the problem type
    standardize : boolean , optional
        if true, the parameter "x" will get these new features : (mean = 0 , std = 1)
    '''
    def __init__(self, x, y, w0,m=None, tag='LinearRegression',standard=False,bias = True):

        e.NumpyErrorCheck(x, y, w0)
        ##########################
        (self.m , self.n) = x.shape
        if m is None:
            m = self.m
        if type(m) != int:
            if type(m)==float:
                self.m = int(m)
            else:
                raise TypeError(''' m must be an integer.''')
        ##########################
        if tag != 'LinearRegression' :
            if tag != 'LogisticRegression':
                raise ValueError(tag)
        ##########################    
        self.x = x
        if standard:
            self.x = prc.standardization(x)
        if bias:
            self.x = prc.add_bias_coef(self.x)
        self.y = y
        self.w0 = w0
        self.m = m
        self.tag = tag
        ##########################
    
    def __algs_init__(self,xyw):
        if len(xyw)>3:
            raise ValueError('At most 3 values to unpack')
        if 'x' in xyw:
            x = xyw['x']
        else:
            x = self.x
        if 'y' in xyw:
            y = xyw['y']
        else:
            y = self.y
        if 'w0' in xyw:
            w0  = xyw['w0']
        else:
            w0  = self.w0
        return x,y,w0
    
    def Hyp(self, degree = 1 , **kwargs):
        '''
        hypothesis function for supervised models.

        Parameters
        ----------
        degree : int, optional
            the degree of weights.
            use degree 2 or more for polynomial regression. (the default is 1 )
        kwargs : dictionary, optional
            it is for passing other arguments, beside self arguments.
            names {x , y , w0} are only accepted.


        Returns
        -------
        ndarray
            estimated value for each data.
        '''
        # set conditions

        #######################    
        x,_,w0 = self.__algs_init__( kwargs)

        if self.tag == 'LinearRegression':
            def activeF(x): return x
        elif self.tag == 'LogisticRegression':
            def activeF(x): return 1 / (1 + np.exp(-x))
        else:
            raise e.ActiveFuncError(
                'Ativation function can not be activated for this problem')
        # starting the algorithm
        hyps = np.zeros((x.shape[0], w0.shape[1]))
        for deg in range(1, degree+1):
            hyp = activeF(np.dot(np.power(x, deg) , w0))
            hyps += hyp
        return hyps

    def loss(self, A='MSE',**kwargs):
        '''
        Computes cost for a given hypothesis.

        Parameters
        ----------
        A : str , optional
            order: {MSE,VEC,MLE}
            to select the solving method
        kwargs : dictionary, optional
            it is for passing other arguments, beside self arguments.
            names {x , y , w0} are only accepted.

        Returns
        -------
        float
        '''
        # starting the algorithm
        
        x,y,w0 = self.__algs_init__(kwargs)

            
        if A == 'MSE':
            def loss(H, m, y): return 1/m * np.sum(np.square((H - y)))
        elif A == 'MLE':
            def loss(H, m, y): return 1/m * np.sum(  (-y * np.log(H))    -    ((1-y)*np.log(1-H))  )
        elif A == 'VEC':
            def loss(H, m, y): return 1/m * (((H - y).T @ (H - y))[0, 0])
        else:
            raise ValueError(
                        'invalid value for parameter %s' % 'A')
        return loss(self.Hyp(x=x,w0=w0), m= self.m,y = y)



    def gradient_descent(self,los='MSE', lr=.001, maxIter=100, converLim=0.001,verbose=0 ,Type='batch', plot_j=False,inplace=False,just_fit = False,**kwargs):
        '''
        Gradient descent algorithm .

        Parameters
        ----------
        los  : str, optional
            loss function to be optimized.
            order : {'MSE','MLE','VEC'}            
        lr : float
            alpha hyperparameter.
        maxIter : int, optional
            maximum number of iterations.
        converLim : float, optional
            set where to stop the algorithm by convergence limitation.
        verbose : int , optional
            default = o
            will show iteration details based on the number is passed to it.
            for example , verbose 2 , will show first two iteration details.
        Type : string, optional
            {'batch','stochastic','mini-batch'}
            for each iteration, if batch, all of smaples pass to algorithm for; if stochastic, just one smaple passes; if
            mini-batch, a random sample collection pass.
        plot_j : bool, optional
            to plot cost function of each iteration
        inplace : bool , optional
            default = False
            if True , self weights will be updated.
        just_fit = bool , optional
            default = False
            if True , weights will be shown in output, but the model still learns.
            recommended (to be more like sikit learn fit function) :
            just_fit = True
            inplace = True
        kwargs : dictionary, optional
            it is for passing other arguments, beside self arguments.
            names {x , y , w0} are only accepted.

        Returns
        -------
        ndarray, int
        optimum weights matrix, loss of new weights.
        '''
        # variables
        costs = []
        iterNum = 0
        x,y,w0 = self.__algs_init__(kwargs)
        ################
        
        # conditions
        if Type != 'batch':
            if Type == 'stochastic':
                idx = np.random.randint(len(x), size=1)
            elif Type == 'mini-batch':
                idx = np.random.randint(len(x), size=int(.30 * len(x)))
            else:
                raise ValueError('invalid value for parameter Type')

        # other conditions:
        if lr < 0:
            raise ValueError('Learning rate must be greater than 0')
        if verbose > maxIter:
            raise ValueError('verbose must be smaller than maxIter')

        # static codes
        def grad(Ob,x,y): return (1/Ob.m) * (np.dot(x.T, (Ob.Hyp(x =x ,w0 = w0) - y)))
        def choose_random_values(x, y, idx): return (x[idx], y[idx])

        # starting the algorithm
        while iterNum != maxIter:

            if Type != 'batch':
                random_x, random_y = choose_random_values(x= x, y=y, idx=idx)
                w0 = w0 - (lr * grad(self,random_x,random_y))
                costs.append(cost := self.loss(A=los,x=random_x,y=random_y,w0=w0))
            else:
                w0 = w0 - (lr * grad(self,x,y))
                costs.append(cost := self.loss(A=los,x=x,y=y,w0=w0))
            iterNum += 1
            
            for verb in range(verbose):
                print('\n\niteration no. {}\nbias : {}\ncoefficients : {}\ncost : {}\n'.format(
                    iterNum, w0[0], w0[1:], float(cost)))
                break
            ##########################################################
            try:
                if abs(costs[-1] - costs[-2]) < converLim:
                    print('End of the algorithm at the iteration number {}.\nThe differences in costs was less than {}'.format(
                        iterNum, converLim))
                    break
            except IndexError:
                pass
            ##########################################################
        if plot_j:
            plt.scatter(range(1,iterNum+1),costs, color='red')
            plt.xlabel('iteration number')
            plt.ylabel('cost')
            plt.show()

        self.intercept_ = w0[0].reshape(1,-1)
        self.coef_ = w0[1:].reshape(-1, 1)
        if inplace:
            self.w0 = w0
        if not just_fit:
            return w0, costs[-1]

    def normal_equation(self,x=None,y=None,w0 = None):
        '''
        normal equation for linear models.
        '''
        if x is not None:
            x = prc.add_bias_coef(x)
        else:
            x = self.x
        ################
        if y is not None:
            y = y
        else:
            y = self.y
        #################
        if w0 is not None:
            w0 = w0
        else:
            w0 = self.w0
        return (np.linalg.pinv(x.T @ x) @ x.T @ y)

    def scores(self,pred , y,*args):
        '''
        docstring will be added later
        '''
        import sklearn.metrics as slm
        if 'r2_score' in args:
            print(f' r2_score is : {slm.r2_score(pred,y)}')
        if 'accuracy_score' in args:
            print(f' accuracy_score is : {slm.accuracy_score(pred,y)}')
        if 'precision_score' in args:
            print(f' precision_score is : {slm.precision_score(pred,y)}')
        if 'recall_score' in args:
            print(f' recall_score is : {slm.recall_score(pred,y)}')
        if 'f1_score' in args:
            print(f' r2_score is : {slm.f1_score(pred,y)}')