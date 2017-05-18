import pla
import numpy as np

class Line:
    def __init__(self, p1, p2):
        #input: 2 2-dim numpy arrays
        self.p1 = p1
        self.p2 = p2
        diff = np.subtract(p2, p1)
        if diff[0] <= 0.001:
            #if vertical (or close to it), just set slope to none
            self.slope = None
            self.is_vert = True
        else:
            self.slope = diff[1]/diff[0]
            self.is_vert = False
        #point slope form = y - y1 = m(x - x1) 
        #y = 0 -> -y1 = m(x - x1) -> -y1/m = x - x1 -> (-y1/m) + x1 = x
        if not self.is_vert:
            self.y_int = ((-1 * p1[1])/self.slope) + p1[0]

        
    def calc(self,testpt):
        #input: numpy array with 2 dim
        #goal: test against equation of line, if above, then return +1
        #if on line 0, else -1
        #to check:
        #if vertical, check against x, else check against y

        #slope-intercept: y = mx + b or (y-b)/m = x
        if self.is_vert == False:
            line_y = self.slope*testpt[0] + self.y_int
            diff = testpt[1] - line_y
        else:
            line_x = self.p1[0]
            diff = testpt[0] - line_x
        return np.sign(diff)
        
class PLAtest:        
    def __init__(self, numpoints):
        self.n = numpoints
        self.points = [np.random.uniform(-1.0,1.0,2) for x in range(numpoints)]
        p = [np.random.uniform(-1.0,1.0,2) for x in range(2)]
        while p[0][0] == p[1][0] and p[0][1] == p[1][1]:
            p = [np.random.uniform(-1.0,1.0,2) for x in range(2)]
        self.line = Line(p[0],p[1])
        self.PLA = pla.PLA(2)

    def test_agreement(self,point):
        cur_p = self.PLA.predict(point)
        actval = self.line.calc(point)
        return cur_p == actval

    def test_convergence(self):
        iters = 0
        testidx = 0
        while True:
            actval = self.line.calc(self.points[testidx])
            success = self.PLA.train(self.points[testidx],actval)
            if success:
                allgood = True
                for i in range(self.n):
                    agree = self.test_agreement(self.points[i])
                    if not agree:
                        allgood = False
                        break
                if allgood:
                    break
            else:
                iters = iters + 1
            testidx = int(np.random.uniform(0, self.n-0.5))
        return iters

    def disagree_probability(self):
        n_disagree = 0
        for x in range(1000):
            agree = self.test_agreement(np.random.uniform(-1.0,1.0,2))
            if not agree:
                n_disagree = n_disagree + 1
        prob = float(n_disagree)/1000.0
        return prob
            
        
                    
        
    
        

def test_run(test_dim, numtests):
    converge_iters = []
    probs = []
    for a in range(numtests):
        cur_test = PLAtest(test_dim)
        cur_conv = cur_test.test_convergence()
        converge_iters.append(cur_conv)
        cur_prob = cur_test.disagree_probability()
        probs.append(cur_prob)
    avg_iters = np.average(converge_iters)
    avg_probs = np.average(probs)
    prefix_str = "For N = " + str(test_dim) + ", "
    print(prefix_str + "convergence took on average " + str(avg_iters) + " iterations")
    print(prefix_str + "probability of disagreeing with target function on average is " + str(avg_probs))
