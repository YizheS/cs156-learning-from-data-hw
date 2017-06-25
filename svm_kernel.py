import numpy as np
import cvxopt as cvo
from svm import SVM

#using cvxopt notation, it takes minimizes x in the following equation:
# 0.5 * xT * P * x + qT * x with constrants G*x <= h, Ax = b

#in the case of our lagrangian, P(i,j) = yi*yj*xi.T*xj
#given that y is a Nx1 matrix, the y components are essentially the outer product y*y.T
# the x compnents are just in the matrix product x*x.T

#our constraints xare y.T*alpha = 0 and alpha => 0
# w Ax=b is the equality constraint, we must make our first constraint fit it
# b = 0, A = y where y should be a row vector
# since we have a greater than 0 constrant and cvxopt takes a less than constraint, we must make our constraint negative
# also we want EACH alpha to be greater than 0, thus dot alpha and identity
# h = 0 vector
# G = -1 * NxN identity times alpha where alpha should be a 1XN matrix


class SVM_Poly(SVM):
    def __init__(self, exponent = 1, upper_limit = 0.01):
        self.thresh = 1.0e-5
        self.exponent = exponent
        self.upper_limit = max(0, upper_limit)
        #suppress output
        cvo.solvers.options['show_progress'] = False

    def set_exponent(self, Q):
        self.exponent = Q

    def set_upper_limit(self, C):
        self.upper_limit = max(0, C)

    def kernel_calc(self, X):
        #polynomial kernel (1+xnT*xm)^Q
        kernel = np.power(np.add(1, np.dot(X,X.T)), self.exponent)
        return kernel

    def get_constraints(self, num_ex):
        #make constraints matrix G, h being passed number of examples
        #-alphas <= 0
        G1 = np.multiply(-1, np.eye(num_ex))
        #alphas <= c
        G2 = np.eye(num_ex)
        G = np.vstack((G1, G2))
        h1 = np.zeros(num_ex)
        h2 = np.ones(num_ex)*self.upper_limit
        h = np.hstack((h1, h2))
        return cvo.matrix(G), cvo.matrix(h)
