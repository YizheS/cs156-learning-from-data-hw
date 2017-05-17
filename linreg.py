import numpy as np
            
class LinReg:
    def __init__(self, dim):
        self.dim = max(1, dim)
        self.weights = np.zeros((1+dim,1)) #adding one for offset

    def predict(self,x):
        real_x = np.append([1], x[:self.dim])
        cur_h = np.matmul(real_x, self.weights)
        return cur_h

    def train(self,X,Y):
        #for the sake of programming ease, let's just assume inputs are numpy ndarrays
        #and are the proper shapes (X = (n, dim), y = (n,1))
        num_examples = np.shape[0]
        real_X = np.c_[np.ones(num_examples), X]
        pinv_X = np.linalg.pinv(real_X)
        self.weights = np.matmul(pinv_X,Y)
