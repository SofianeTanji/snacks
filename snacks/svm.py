# Author : Sofiane Tanji
# License : GNU GPL V3
import numpy as np
from optimizer import Optimizer
from sklearn.metrics import f1_score

class Snacks:
    """Documentation"""

    def __init__(self, lambda_reg, stepsize = 1.5, n_iter = 120000, D = 1100, K = 10, verbose=False):
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

    def score(self, X, Y, metric = "accuracy"):
        Ypred = self.predict(X)
        if metric == "accuracy":
            score = ((Ypred == Y).sum() / len(Y))
        elif metric == "f1":
            score = f1_score(Y, Ypred, average = 'binary')
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
