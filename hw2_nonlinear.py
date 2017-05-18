from hw2_linreg import Line, LRtest

class NLTtest(LRtest):
    #adding noise to labels
    def add_noise(self, amt):
        amt = max(min(1, amt), 0)
        # number of indices to flip labels
        n_flip = int(self.n * amt)
        # label-flipping array consisting of
        # 1: don't flip
        # -1: flip
        flip_arr = np.random.shuffle(np.r_[np.ones(self.n - n_flip), np.zeros(n_flip)])
        self.noisy_labels = np.multiply(flip_arr, self.labels)
        
    def __init__(self, numpoints, noise):
        LRtest.__init__(self, numpoints)
        self.noise = max(min(1, noise), 0)
        self.add_noise(self.noise)
        
