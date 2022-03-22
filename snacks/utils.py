# Author : Sofiane Tanji
# License : GNU GPL V3

import random

random.seed(1999)

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from libsvmdata import fetch_libsvm
from sklearn.metrics.pairwise import pairwise_kernels
from scipy.linalg import sqrtm, inv
from collections import Counter
from sklearn.gaussian_process.kernels import PairwiseKernel
from memory_profiler import profile
import numba


@numba.njit(fastmath=True)
def objective_func(X, Y, pen, w):
    pred = w @ X
    objres = np.maximum(0, 1 - pred * Y)
    reg = pen * np.dot(w, w)
    return np.mean(objres) + reg

@numba.njit(fastmath=True)
def l1_func(X, Y, pen, w):
    pred = w @ X
    objres = np.maximum(0, 1 - pred * Y)
    reg = pen * np.linalg.norm(w, ord = 1)
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
def l1_grad(X, Y, _, w):
    _, n = X.shape
    data_idx = random.randint(0, n - 1)
    x, y = X[:, data_idx], Y[data_idx]
    pred = np.dot(w, x)
    subg = np.sign(w)
    subg[subg == 0] = 1
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


def dataloader(datafile, train_size = 0.8):
    # datafile = "../datasets/" + datafile + ".bz2"
    # data = load_svmfile(datafile, dtype="f")
    X, y = fetch_libsvm(datafile)
    # X, y = X.toarray(), y
    Xtr, Xts, Ytr, Yts = train_test_split(X, y, train_size=train_size)
    return X, y, Xtr, Xts, Ytr, Yts

def kernel_embedding(Xtr, Ytr, Xts, Yts, num_centers, **kernel_params):
    """Documentation"""
    
    Xtr = normalize(Xtr, axis=1, norm='l2').astype("float32")
    Xts = normalize(Xts, axis=1, norm='l2').astype("float32")
    
    Ytr[Ytr != 1] = -1
    Yts[Yts != 1] = -1

    centers_idx = np.random.choice(Xtr.shape[0], size=num_centers, replace=False)
    centers = Xtr[centers_idx].astype("float32")

    del centers_idx
    
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
