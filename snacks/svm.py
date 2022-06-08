# Author : Sofiane Tanji
# License : GNU GPL V3
import numpy as np
from optimizer import Optimizer


class Snacks:
    """Documentation"""

    def __init__(self, lambda_reg, stepsize = 10, n_iter = 85000, D = 1100, K = 3, verbose=False):
        self.verbose = verbose
        self.n_iter = n_iter
        self.lambda_reg = lambda_reg
        self.stepsize = stepsize
        self.D = D
        self.K = K
        self.objs = None
        self.weights = None

    def fit(self, X, Y):
        self.weights, self.objs = Optimizer(
            n_iter=self.n_iter,
            K = self.K,
            D0 = self.D,
            lambda_reg=self.lambda_reg,
            eta0=self.stepsize,
            verbose=self.verbose,
        ).run(X.T, Y.T)
        self.classes_, Y = np.unique(Y, return_inverse=True)
        return self

    def decision_function(self, X):
        return np.dot(self.weights, X.T)

    def predict(self, X):
        d = self.decision_function(X)
        d[d > 0] = 1
        d[d <= 0] = -1
        return d.astype(np.int32)

    def score(self, X, Y):
        Ypred = self.predict(X)
        score = (Ypred == Y).sum() / len(Y)
        return score
    
    def get_params(self, deep = True):
        return {
            "n_iter": self.n_iter,
            "lambda_reg": self.lambda_reg,
            "stepsize": self.stepsize,
            "D": self.D,
            "K": self.K
        }
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self