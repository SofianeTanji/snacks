# Author : Sofiane Tanji
# License : GNU GPL V3
import numpy as np
from numba.experimental import jitclass
from numba import float32
from optimizer import Optimizer


class Snacks:
    """Documentation"""

    def __init__(self, nb_iterations, lambda_reg, stepsize, verbose=False):
        self.verbose = verbose
        self.nb_iterations = nb_iterations
        self.lambda_reg = lambda_reg
        self.stepsize = stepsize

    def fit(self, Xtr, Ytr):
        self.weights = Optimizer(nb_iterations = self.nb_iterations, lambda_reg = self.lambda_reg, eta0 = self.stepsize).run(Xtr.T, Ytr.T)
        return self

    def decision_function(self, Xts):
        return np.dot(self.weights, Xts.T)

    def predict(self, Xts):
        d = self.decision_function(Xts)
        d[d > 0] = 1
        d[d <= 0] = -1
        return d.astype(np.int32)

    def score(self, Xts, Yts):
        Ypred = self.predict(Xts)
        score = (Ypred == Yts).sum() / len(Yts)
        return score