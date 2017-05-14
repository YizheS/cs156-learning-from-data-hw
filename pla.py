import numpy as np

class PLA:
    def __init__(self, dim):
        self.dim = dim
        self.weights = np.zeros(1+dim) #adding one for threshold

    def predict(self,x):
        realx = np.append([1],x[:self.dim])
        cur_h = np.dot(self.weights, realx)
        return np.sign(cur_h)

    def train(self,x,y):
        guess = self.predict(x)
        realx = np.append([1],x[:self.dim])
        agreed = guess == y
        if not agreed:
            self.weights = self.weights + np.multiply(y,realx)
        return agreed
            
