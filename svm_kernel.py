import numpy as np
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
    def __init__(self, exponent upper_limit):
        self.thresh = 1.0e-5
        self.exponent = exponent
        self.upper_limit = max(0, upper_limit)
        #suppress output
        cvo.solvers.options['show_progress'] = False

    def kernel_calc(self, X):
        #polynomial kernel (1+xnT*xm)^Q
        return np.power(np.add(1, np.dot(X,X.T)), self.exponent) 

    def get_constraints(self, num_ex):
        #make constraints matrix G, h being passed number of examples
        #-alphas <= 0
        G1 = np.multiply(-1, np.eye(num_ex))
        #alphas <= c
        G2 = np.eye(num_ex)
        G = cvo.matrix(np.vstack((G1, G2)))
        h = cvo.matrix(np.r_[np.zeros(num_ex), np.full(num_ex, self.upper_limit)])
        return G, h
