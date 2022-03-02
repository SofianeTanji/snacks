# Author : Sofiane Tanji
# License : GNU GPL V3

import random

import sys
import numpy as np

from sklearn.model_selection import train_test_split
from svmloader import load_svmfile
from sklearn.metrics.pairwise import pairwise_kernels
from scipy.linalg import sqrtm, inv
import numba


@numba.njit(fastmath=True)
def objective_func(X, Y, pen, w):
    pred = w @ X
    objres = np.maximum(0, 1 - pred * Y)
    reg = pen * np.dot(w, w)
    return np.mean(objres) + reg


@numba.njit(fastmath=True)
def objective_grad(X, Y, gamma, w):
    _, n = X.shape
    data_idx = random.randint(0, n - 1)
    x, y = X[:, data_idx], Y[data_idx]
    pred = np.dot(w, x)
    subg = gamma * w
    if y * pred < 1:
        subg -= y * x
    return subg


@numba.njit(fastmath=True)
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
    data = load_svmfile(datafile)
    X, y = data[0], data[1]
    X, y = X.toarray(), y
    Xtr, Xts, Ytr, Yts = train_test_split(X, y, train_size=train_size)
    return X, y, Xtr, Xts, Ytr, Yts


def kernel_embedding(Xtr, Ytr, Xts, Yts, num_centers, **kernel_params):
    """Documentation"""

    Ytr[Ytr == 0] = -1
    Yts[Yts == 0] = -1

    centers_idx = np.random.choice(Xtr.shape[0], size=num_centers, replace=False)
    centers = Xtr[centers_idx].astype("float32")

    Kmm = pairwise_kernels(centers, centers, metric="rbf", **kernel_params).astype(
        "float32"
    )
    Kmm_sqrt_inv = inv(
        np.real(
            sqrtm(Kmm + 1e-6 * centers.shape[0] * np.eye(Kmm.shape[0], dtype=Kmm.dtype))
        )
    ).astype("float32")

    del Kmm

    Knm = pairwise_kernels(Xtr, centers, metric="rbf", **kernel_params).astype(
        "float32"
    )

    Xtr_emb = Knm @ Kmm_sqrt_inv

    del Knm

    Knm = pairwise_kernels(Xts, centers, metric="rbf", **kernel_params)

    Xts_emb = Knm @ Kmm_sqrt_inv

    del Knm
    del Kmm_sqrt_inv

    return Xtr_emb, Ytr, Xts_emb, Yts
