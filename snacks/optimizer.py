# Author : Sofiane Tanji
# License : GNU GPL V3

from profilehooks import profile
import numpy as np
import utils
from numba import njit

@njit
def _run(X, Y, D0, eta0, lambda_reg, K, nb_iterations):
    weights = np.zeros((X.shape[0],), dtype=X.dtype)
    eps = utils.objective_func(X, Y, lambda_reg, weights)
    D = D0 * eps
    eta = eta0 * eps / 3

    for k in range(K):
        center = np.copy(weights)
        avgw = np.zeros_like(weights)
        for _ in range(nb_iterations):
            grad = utils.objective_grad(X, Y, lambda_reg, weights)
            weights -= eta * grad
            weights = utils.project(center, D, weights)
            avgw += weights / nb_iterations
        weights = np.copy(avgw)
        eta, D = eta / 2, D / 2

    return weights

class Optimizer:
    """Documentation
    The user should provide the objective function obj and its derivate grad.
    This class solves the minimization problem
    """

    def __init__(
        self, nb_iterations=30000, K=3, D0=15, eta0=1.0, lambda_reg=1e-5, verbose=False
    ):
        self.nb_iterations = nb_iterations
        self.K = K
        self.D0 = D0
        self.eta0 = eta0
        self.lambda_reg = lambda_reg
        self.verbose = verbose
    
    # @profile(immediate = True)
    def run(self, X, Y):
        weights = _run(X, Y, self.D0, self.eta0, self.lambda_reg, self.K, self.nb_iterations)
        return weights

    def runnn(self, X, Y):
        """Runs the optimization process
        :type verbose: bool
        :param verbose: If true, prints advance of the process in the console
        :return: solution of the problem
        """
        weights = np.zeros((X.shape[0],), dtype=X.dtype)

        eps = utils.objective_func(X, Y, self.lambda_reg, weights)
        D = self.D0 * eps
        eta = self.eta0 * eps / 3

        for k in range(self.K):
            center = np.copy(weights)
            avgw = np.zeros_like(weights)
            for _ in range(self.nb_iterations):
                grad = utils.objective_grad(X, Y, self.lambda_reg, weights)
                weights -= eta * grad
                weights = utils.project(center, D, weights)
                avgw += weights / self.nb_iterations
            weights = np.copy(avgw)
            eta, D = eta / 2, D / 2

        return weights
