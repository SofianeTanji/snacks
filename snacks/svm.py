# Author : Sofiane Tanji
# License : GNU GPL V3
import numpy as np
from sklearn.base import BaseEstimator
from optimizer import Optimizer


class Snacks:
    """Documentation"""

    def __init__(self, nb_iterations, lambda_reg, stepsize, verbose=False):
        self.verbose = verbose
        self.nb_iterations = nb_iterations
        self.lambda_reg = lambda_reg
        self.stepsize = stepsize
        self.objs = None
        super().__init__()

    def fit(self, Xtr, Ytr):
        self.weights, self.objs = Optimizer(
            nb_iterations=self.nb_iterations,
            lambda_reg=self.lambda_reg,
            eta0=self.stepsize,
            verbose=self.verbose,
        ).run(Xtr.T, Ytr.T)
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


class MyEstimator(BaseEstimator):
    def __init__(self, subestimator):
        self.subestimator = subestimator

    def fit(self, Xtr, Ytr):
        self.subestimator.fit(Xtr, Ytr)
        return self

    def predict(self, Xts):
        return self.subestimator.predict(Xts)

    def score(self, Xts, Yts):
        return 100 - 100 * self.subestimator.score(Xts, Yts)
