# algorithms
import warnings
import errors as e
import preprocessing as prc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
#########################################################################################################


class LinearLogisticRegression():
    '''
    the class "LinearLogisticRegression" provides some algorithms for both linear and logistic regression.
    
    parameters
    ----------
    x : ndarray
        feature matrix
    y : ndarray
        target matrix
    w : ndarray
        initial weight matrix
    m : int
        an integer number which is usually length of the given data
    kind : str , optional
        order : {LinearRegression , LogisticRegression}
        to determine the problem type
    standard : boolean , optional
        if true, the parameter "x" will get these new features : (mean = 0 , std = 1)
    penalty : str , optional
        order : {'L1','L2'}
        regularization technique.
    alpha : float , optional
        regularization hyper parameter.
    '''
    def __init__(self, x, y, w=None, bias = 0, m=None, kind='LinearRegression',standard=False,bias_included=False,penalty='L2',alpha = 0):

        e.NumpyErrorCheck(x, y, w)
        ##########################
        if x.ndim == 1:
            x = x[:,np.newaxis]
        (self.m,self.n)=x.shape

        if m is None:
            m = self.m
        if type(m) != int:
            if type(m)==float:
                self.m = int(m)
            else:
                raise TypeError(''' m must be an integer.''')
        ##########################
        if kind != 'LinearRegression' :
            if kind != 'LogisticRegression':
                raise ValueError(kind)
        ########################### 
    
        self.x = x
        self.y = y
        self.m = m
        self.bias = bias
        self.penalty = penalty
        self.alpha = alpha
        self.kind_ = kind

        if w is not None:
            self.w = w
        else:
            self.w = np.ones((self.n,1))        
        if bias_included:
            self.bias = w[0]
            self.w = w[1:]
        if standard:
            self.x = prc.standardization(x)
        ##########################
    
    def __init_algs__(self,kwg):
        if len(kwg)>4:
            raise ValueError('At most 4 values to unpack')
        if 'x' in kwg:
            x = kwg['x']
        else:
            x = self.x

        if 'y' in kwg:
            y = kwg['y']
        else:
            y = self.y

        if 'w' in kwg:
            w  = kwg['w']
        else:
            w  = self.w

        if 'bias' in kwg:
            bias = kwg['bias']
        else:
            bias = self.bias

        return x,y,w,bias

    def _penalty(self,w):

        if self.penalty =='L2':
            #return self.alpha * np.sum(np.square(w))
            return self.alpha * (w.T @ w)[0,0]
        elif self.penalty =='L1':
            return self.alpha * np.sum(np.abs(w))
        else:
            raise ValueError(f'''invalid value "{self.penalty}"''')

    def predict(self, degree = 1, interaction=True, random_weight = False, **kwargs):
        '''
        hypothesis function for supervised models.

        Parameters
        ----------
        degree : int, optional
            the degree of weights.
            use degree 2 or more for polynomial regression. (the default is 1 )
        kwargs : dictionary, optional
            it is for passing other arguments, beside self arguments.
            names {x , y , w} are only accepted.

        Returns
        -------
        ndarray
            estimated value for each data.
        '''
        # set conditions

        #######################    
        x,_,w,bias = self.__init_algs__( kwargs)
        poly = PolynomialFeatures(degree=degree,include_bias=False)
        if self.kind_ == 'LinearRegression':
            def activeF(x): return x
        elif self.kind_ == 'LogisticRegression':
            def activeF(x): return 1 / (1 + np.exp(-x))
        else:
            raise e.ActiveFuncError(
                'Ativation function can not be activated for this problem')
        # starting the algorithm
        if interaction:
            x = poly.fit_transform(x)
            if random_weight:
                w = np.random.randn(x.shape[-1],1)
            return activeF(np.dot(x,w)+bias)
        else:
            hyps = np.zeros((x.shape[0], w.shape[1]))
            for deg in range(1, degree+1):
                hyp = activeF(np.dot(np.power(x, deg) , w)+bias)
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
            names {x , y , w} are only accepted.

        Returns
        -------
        float
        '''
        # starting the algorithm
        
        x,y,w,bias = self.__init_algs__(kwargs)

            
        if A == 'MSE':
            def loss(p, m, y): return 1/m * (np.sum(np.square(p - y)) + self._penalty(w))
        elif A == 'VEC_MSE':
            def loss(H, m, y): return 1/m * ( ((H - y).T @ (H - y))[0, 0] + self._penalty(w) )
        elif A == 'MLE':
            def loss(H, m, y): return 1/m * (np.sum( (-y * np.log(H)) - ((1-y)*np.log(1-H)) ) + self._penalty(w)) 
        else:
            raise ValueError(
                        'invalid value for parameter %s' % 'A')
        return loss(self.predict(x=x,w=w,bias = bias), m= self.m,y = y)



    def gradient_descent(self,los='MSE', lr=.001, maxIter=100, converLim=0.001,verbose=0 ,Type='batch', plot_j=False,inplace=False,**kwargs):
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
            names {x , y , w} are only accepted.

        Returns
        -------
        ndarray, int
        optimum weights matrix, loss of new weights.
        '''
        # variables
        costs = []
        iterNum = 0
        x,y,w,bias = self.__init_algs__(kwargs)
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
        def grad(Ob,x,y,W,B): return (1/Ob.m) * (np.dot(x.T, (Ob.predict(x =x ,w = W,bias = B) - y)))
        def bias_grad(Ob,x,y,W,B) : return (1/Ob.m) * np.sum(Ob.predict(x=x,w=W,bias = B) - y) 
        def choose_random_values(x, y, idx): return (x[idx], y[idx])

        # starting the algorithm
        while iterNum != maxIter:

            if Type != 'batch':
                random_x, random_y = choose_random_values(x= x, y=y, idx=idx)
                G = grad(self,random_x,random_y,w,bias)
                B_G = bias_grad(self,random_x,random_y,w,bias)
                bias = bias - (lr * B_G)
                #w = w * (1 - (self.alpha * lr))  - (lr * grad(self,random_x,random_y))
                w = w - (lr * (G + (self.alpha * w)))
                costs.append(cost := self.loss(A=los,x=random_x,y=random_y,w=w,bias = bias))

            else:
                G = grad(self, x, y, w, bias)
                B_G = bias_grad(self,x,y,w,bias)
                bias = bias - (lr * B_G)
                #w = w * (1 - (self.alpha * lr)) - (lr * grad(self,x,y))
                w = w - lr * (G + (self.alpha * w))
                costs.append(cost := self.loss(A=los,x=x,y=y,w=w,bias = bias ))
            
            iterNum += 1
            
            if verbose>0:
                print('\n\niteration no. {}\nbias : {}\ncoefficients : {}\ncost : {}\n'.format(
                    iterNum, bias, w, float(cost)))
                verbose -=1
                print(len(costs))
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

        if inplace:
            self.bias = w[0].reshape(1,-1)
            self.w = w[1:].reshape(-1, 1)

        return bias,w,costs[-1]

    def normal_equation(self,**kwargs):
        '''
        normal equation for linear models.
        note = this implentation is an old version and not updated yet.
        '''
        x,y,_,_ = self.__init_algs__(kwargs)
        return (np.linalg.pinv(x.T @ x) @ x.T @ y)