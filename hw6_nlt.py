import numpy as np
from linreg import LinReg

# want: 1, x1, x2, x1^2, x2^2, x1*x2, abs(x1-x2), abs(x1+x2)
class LinRegNLT2(LinReg):
    def __init__(self, dim):
        #want squares of both elements, both elts multiplied, then abs sub and add
        # = 2*dim + 3 
        self.dim = (2*dim + 3)
        #adding the x0 bit
        self.weights = np.zeros((self.dim + 1, 1))

    def X_reshape(self,X):
        #do the nonlinear transform here
        num_ex = X.shape[0] #number of examples
        X_mult = np.prod(X, axis=1)
        X_sub_mtx = np.c_[ X[:,0], np.multiply(-1, X[:,1:])] #subtraction matrix
        X_res = np.c_[np.ones(num_ex), X, np.square(X), np.abs(np.sum(X_sub_mtx, axis=1)), np.abs(np.sum(X, axis=1))]
        return X_res

    def calc_error(self, X,Y):
        num_ex = X.shape[0]
        predicted = np.sign(self.predict(X))
        num_incorrect = np.sum(np.not_equal(predicted, Y))
        prop_incorrect = float(num_incorrect)/float(num_ex)
        return prop_incorrect
        
        
