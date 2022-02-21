from svm import Snacks
from itertools import product
import utils
import numpy as np
import matplotlib.pyplot as plt


def cerr_vs_m(
    dataset, train_size, gamma, nb_iterations, nb_runs, lambda_reg, stepsize, arr_m
):
    oXtr, oXts, oYtr, oYts = utils.dataloader(dataset, train_size)
    cerrs_vs_m = []
    for m in arr_m:
        Xtr, Ytr, Xts, Yts = utils.kernel_embedding(
            oXtr, oYtr, oXts, oYts, m, gamma=gamma
        )
        scores = []
        for n in range(nb_runs):
            model = Snacks(
                nb_iterations=nb_iterations, lambda_reg=lambda_reg, stepsize=stepsize
            )
            model.fit(Xtr, Ytr)
            scores.append(1 - model.score(Xts, Yts))
            del model
        cerrs_vs_m.append((m, np.mean(scores)))
    cerrs_vs_m = np.array(cerrs_vs_m)
    Cerrs, Ms = cerrs_vs_m[:, 0], cerrs_vs_m[:, 1]
    plt.plot(Ms, Cerrs)
    return Ms, Cerrs


def cerr_vs_lmbd(
    dataset, train_size, gamma, nb_iterations, nb_runs, m, stepsize, arr_l
):
    oXtr, oXts, oYtr, oYts = utils.dataloader(dataset, train_size)
    Xtr, Ytr, Xts, Yts = utils.kernel_embedding(oXtr, oYtr, oXts, oYts, m, gamma=gamma)
    cerrs_vs_lmbd = []
    for lmbd in arr_l:
        scores = []
        for n in range(nb_runs):
            model = Snacks(
                nb_iterations=nb_iterations, lambda_reg=lmbd, stepsize=stepsize
            )
            model.fit(Xtr, Ytr)
            scores.append(1 - model.score(Xts, Yts))
            del model
        cerrs_vs_lmbd.append((lmbd, np.mean(scores)))
    cerrs_vs_lmbd = np.array(cerrs_vs_lmbd)
    Cerrs, Lmbds = cerrs_vs_lmbd[:, 0], cerrs_vs_lmbd[:, 1]
    plt.plot(Lmbds, Cerrs)
    return Lmbds, Cerrs


def cerr_m_lmbd(
    dataset,
    train_size,
    gamma,
    nb_iterations,
    nb_runs,
    stepsize,
    arr_m,
    arr_l,
):
    oXtr, oXts, oYtr, oYts = utils.dataloader(dataset, train_size)

    Cerr = []
    for m, lmbd in product(arr_m, arr_l):
        Xtr, Ytr, Xts, Yts = utils.kernel_embedding(
            oXtr, oYtr, oXts, oYts, m, gamma=gamma
        )
        scores = []
        for n in range(nb_runs):
            model = Snacks(
                nb_iterations=nb_iterations, lambda_reg=lmbd, stepsize=stepsize
            )
            model.fit(Xtr, Ytr)
            scores.append(1 - model.score(Xts, Yts))
            del model
        Cerr.append(np.mean(scores))

    Cerr = np.reshape(np.array(Cerr), (arr_m.size, arr_l.size))
    return Cerr, arr_m, arr_l


def grid_search_map():
    pass


def comparison():
    pass


def obj_vs_it(
    dataset,
    train_size,
    gamma,
    m,
    lmbd,
    stepsize,
    nb_iterations,
):
    oXtr, oXts, oYtr, oYts = utils.dataloader(dataset, train_size)
    Xtr, Ytr, _, _ = utils.kernel_embedding(oXtr, oYtr, oXts, oYts, m, gamma=gamma)
    model = Snacks(
        nb_iterations=nb_iterations, lambda_reg=lmbd, stepsize=stepsize, verbose=True
    )
    model.fit(Xtr, Ytr)
    return model.objs
