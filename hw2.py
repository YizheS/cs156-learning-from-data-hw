import hw2_hoeffding
import hw2_linreg

import sys

n_hoeffding = 10000
n_linreg = 1000

if len(sys.argv) > 1:
    probstr = sys.argv[1].lower()
    if probstr == "hoeffding":
        hw2_hoeffding.prob12(n_hoeffding)
    elif probstr == "linreg":
        hw2_linreg.prob567(n_linreg)

"""

HW2_HOEFFDING:

assignment asks for 10 times the experiments, but it was running a little long so this should be fine

EXAMPLE OUTPUT
Average number of heads:
first coins of sims: 0.498810
coin w/ min freq of heads of sims: 0.038080
rand selection of coin of sims: 0.501150

HW2_LINREG:
e_in average: 0.045974
e_out average: 0.015013
perceptron convergence average: 3.995000

"""
