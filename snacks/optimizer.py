# Author : Sofiane Tanji
# License : GNU GPL V3

import numpy as np
import utils
from numba import njit

@njit(fastmath=True)
def _run(X, Y, D0, eta0, lambda_reg, K, n_iter, verbose):
    weights = np.zeros((X.shape[0],), dtype=X.dtype)
    eps = utils.objective_func(X, Y, lambda_reg, weights)
    objs = []
    D = D0 * eps
    eta = eta0 * eps / 3

    for k in range(K):
        center = np.copy(weights)
        avgw = np.zeros_like(weights)
        for _ in range(n_iter):
            grad = utils.objective_grad(X, Y, lambda_reg, weights)
            weights -= eta * grad
            weights = utils.project(center, D, weights)
            avgw += weights / n_iter
            if verbose and _ % 500 == 0:
                objs.append(
                    (
                        n_iter * k + _,
                        utils.objective_func(X, Y, lambda_reg, weights),
                    )
                )
        weights = np.copy(avgw)
        eta, D = eta / 2, D / 2

    return weights, objs


class Optimizer:
    """Documentation
    The user should provide the objective function obj and its derivate grad.
    This class solves the minimization problem
    """

    def __init__(
        self, n_iter=30000, K=3, D0=15, eta0=1.0, lambda_reg=1e-5, verbose=False
    ):
        self.n_iter = n_iter
        self.K = K
        self.D0 = D0
        self.eta0 = eta0
        self.lambda_reg = lambda_reg
        self.verbose = verbose

    def run(self, X, Y):
        weights, objs = _run(
            X,
            Y,
            self.D0,
            self.eta0,
            self.lambda_reg,
            self.K,
            self.n_iter,
            self.verbose,
        )

        return weights, objs
