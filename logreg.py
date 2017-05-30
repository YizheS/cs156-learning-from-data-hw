import math
import numpy as np

class LogReg:
    def __init__(self,dim, l_rate):
        dim = max(1, dim)
        self.dim = dim
        self.l_rate = l_rate # learning rate
        self.weights = np.zeros(dim+1)

    def init_weights(self):
        self.weights = np.zeros(self.dim + 1)
        
    def reshape_X(self, X):
        #number of examples
        if len(X.shape) > 1:
            num_ex = X.shape[0]
            return np.c_[ np.ones(num_ex), X]
        else:
            return np.r_[1, X]

    def risk_score(self, X):
        #should return (n, 1)
        res_X = self.reshape_X(X)
        return np.dot(X,self.weights)

    def sigmoid(self, X):
        #theta(s) = e^s/(1+e^s)
        cur_es = np.exp(risk_score(X))
        return np.divide(cur_es, np.add(1, cur_es))

    def gradient(self, X, y):
        #grad(E_in) = (-1/N)*sum(n=1;N){(y_n*x_n)/(1+e^(y_n*wT(t)*x_n))}
        res_X = self.reshape_X(X)
        cur_N = X.shape[0]
        cur_numer = np.multiply(y,res_X) #y_n*x_n by row, should be (n,dim+1)
        #should return (n,1)
        cur_denom = np.add(1, np.exp(np.multiply(y, self.risk_score(X))))
        #divide cur_numer row wise by cur_denom, should still be (n, dim+1)
        presum = np.divide(cur_numer, cur_denom)
        #sum by column
        cur_sum = np.sum(presum, axis = 0)
        #now normalize by (-1/N) and return
        cur_sum = np.divide(cur_sum, -1*cur_N)
        return cur_sum

    def update_weights(self, X, y):
        #w(t+1) = w(t) - l_rate * gradient
        cur_grad = self.gradient(X,y)
        self.weights = np.subtract(self.weights, np.multiply(self.l_rate, cur_grad))
    
        

    
