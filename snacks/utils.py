# Author : Sofiane Tanji
# License : GNU GPL V3

import random

SEED = 1999
random.seed(SEED)
import sys
import numpy as np
np.random.seed(SEED)
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file
from sklearn.metrics.pairwise import pairwise_kernels
from scipy.linalg import sqrtm, inv
import numba


@numba.njit
def objective_func(X, Y, pen, w):
    pred = w @ X
    objres = np.maximum(0, 1 - pred * Y)
    reg = pen * np.dot(w, w)
    return np.mean(objres) + reg


@numba.njit
def objective_grad(X, Y, gamma, w):
    _, n = X.shape
    data_idx = random.randint(0, n - 1)
    x, y = X[:, data_idx], Y[data_idx]
    pred = np.dot(w, x)
    subg = gamma * w
    if y * pred < 1:
        subg -= y * x
    return subg


@numba.njit
def project(center, radius, weights):
    """Projection method
    Project "weights" onto ball of radius "radius" around "center"
    """
    dist = np.linalg.norm(weights - center)
    if dist > radius:
        weights = center + np.float32(radius / dist) * (weights - center)
    return weights


def dataloader(datafile, train_size):
    datafile = "../datasets/" + datafile
    data = load_svmlight_file(datafile)
    X, y = data[0], data[1]
    X, y = X.toarray(), y
    Xtr, Xts, Ytr, Yts = train_test_split(
        X, y, train_size=train_size, random_state=SEED
    )
    Xtr, Xts, Ytr, Yts = Xtr.astype('float32'), Xts.astype('float32'), Ytr.astype('float32'), Yts.astype('float32')
    return Xtr, Xts, Ytr, Yts


def kernel_embedding(Xtr, Ytr, Xts, Yts, gpu, num_centers, **kernel_params):
    """Documentation"""

    centers_idx = np.random.choice(Xtr.shape[0], size=num_centers, replace=False)
    centers = Xtr[centers_idx].astype('float32')

    Kmm = pairwise_kernels(centers, centers, metric="rbf", **kernel_params).astype('float32')
    Kmm_sqrt_inv = inv(
        np.real(
            sqrtm(Kmm + 1e-6 * centers.shape[0] * np.eye(Kmm.shape[0], dtype=Kmm.dtype))
        )
    ).astype('float32')

    Knm = pairwise_kernels(Xtr, centers, metric="rbf", **kernel_params).astype('float32')

    Xtr_emb = Knm @ Kmm_sqrt_inv

    Knm = pairwise_kernels(Xts, centers, metric="rbf", **kernel_params)

    Xts_emb = Knm @ Kmm_sqrt_inv

    return Xtr_emb, Ytr, Xts_emb, Yts
