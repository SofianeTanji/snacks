from svm import Snacks
import seaborn as sns
from sklearn import svm
from sklearn.linear_model import RidgeClassifier
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
        m = int(m)
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
    sns.set_theme(style="white", palette = "colorblind")
    ax = sns.lineplot(x=Cerrs, y=Ms, legend="auto")
    ax.set(xlabel="m", ylabel="C-err", title=f"Dataset = {dataset}")
    fig = ax.get_figure()
    fig.savefig(f"../figures/c_err-vs-m-{dataset}.png", bbox_inches='tight')
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
    sns.set_theme(style="white", palette = "colorblind")
    ax = sns.lineplot(x=Cerrs, y=Lmbds, legend="auto")
    ax.set_xscale("log")
    ax.set(xlabel="Regularization parameter", ylabel="C-err", title=f"Dataset = {dataset}")
    fig = ax.get_figure()
    fig.savefig(f"../figures/c_err-vs-lmbd-{dataset}.png", bbox_inches='tight')
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
    sns.set_theme(style="white", palette = "colorblind")
    cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True)
    ax = sns.heatmap(Cerr.T, xticklabels=arr_m, yticklabels=arr_l, cmap=cmap, square = True)
    ax.set(xlabel="m", ylabel="Lambda", title=f"Dataset = {dataset}")
    fig = ax.get_figure()
    fig.savefig(f"../figures/c_err-m-lmbd-{dataset}.png", bbox_inches='tight')
    return Cerr, arr_m, arr_l

def heatmap(
    dataset,
    train_size,
    gamma,
    nb_runs,
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
            model = RidgeClassifier(alpha = lmbd, solver = "saga")
            model.fit(Xtr, Ytr)
            scores.append(1 - model.score(Xts, Yts))
            del model
        Cerr.append(np.mean(scores))

    Cerr = np.reshape(np.array(Cerr), (arr_m.size, arr_l.size))
    sns.set_theme(style="white", palette = "colorblind")
    cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True)
    ax = sns.heatmap(Cerr.T, xticklabels=arr_m, yticklabels=arr_l, cmap=cmap, square = True)
    ax.set(xlabel="m", ylabel="Lambda", title=f"Dataset = {dataset}")
    fig = ax.get_figure()
    fig.savefig(f"../figures/c_err-m-lmbd-{dataset}-krr.png", bbox_inches='tight')
    return Cerr, arr_m, arr_l

def grid_search_map(dataset, train_size, nb_iterations, m, nb_runs, arr_gamma, arr_lambda):
    oXtr, oXts, oYtr, oYts = utils.dataloader(dataset, train_size)

    Cerr = []
    for g, l in product(arr_gamma, arr_lambda):
        Xtr, Ytr, Xts, Yts = utils.kernel_embedding(
            oXtr, oYtr, oXts, oYts, m, gamma=g
        )
        scores = []
        for _ in range(nb_runs):
            model = Snacks(nb_iterations = nb_iterations, lambda_reg = l, stepsize = 1.)
            model.fit(Xtr, Ytr)
            scores.append(1 - model.score(Xts, Yts))
            del model
        Cerr.append(np.mean(scores))
    Cerr = np.reshape(np.array(Cerr), (arr_gamma.size, arr_lambda.size))
    sns.set_theme(style="white", palette = "colorblind")
    cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True)
    ax = sns.heatmap(Cerr.T, xticklabels=arr_gamma, yticklabels=arr_lambda, cmap=cmap, square = True)
    ax.set(xlabel="Gamma", ylabel="Lambda", title=f"Dataset = {dataset}")
    fig = ax.get_figure()
    fig.savefig(f"../figures/heatmap-gamma-lmbd-{dataset}.png", bbox_inches='tight')
    return Cerr, arr_gamma, arr_lambda

def grid_search_krr(dataset, train_size, m, nb_runs, arr_gamma, arr_lambda):
    oXtr, oXts, oYtr, oYts = utils.dataloader(dataset, train_size)

    Cerr = []
    for g, l in product(arr_gamma, arr_lambda):
        Xtr, Ytr, Xts, Yts = utils.kernel_embedding(
            oXtr, oYtr, oXts, oYts, m, gamma=g
        )
        scores = []
        for _ in range(nb_runs):
            model = RidgeClassifier(alpha = l, solver = "sparse_cg")
            model.fit(Xtr, Ytr)
            scores.append(1 - model.score(Xts, Yts))
            del model
        Cerr.append(np.mean(scores))
    Cerr = np.reshape(np.array(Cerr), (arr_gamma.size, arr_lambda.size))
    sns.set_theme(style="white", palette = "colorblind")
    cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True)
    ax = sns.heatmap(Cerr.T, xticklabels=arr_gamma, yticklabels=arr_lambda, cmap=cmap, square = True)
    ax.set(xlabel="Gamma", ylabel="Lambda", title=f"Dataset = {dataset}")
    fig = ax.get_figure()
    fig.savefig(f"../figures/heatmap-gamma-lmbd-{dataset}-krr.png", bbox_inches='tight')
    return Cerr, arr_gamma, arr_lambda

def comparison():
    # On fait les benchmarks sur SUSY ensuite on avise.
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
    objs = model.objs
    it, obj_vals = [x[0] for x in objs], [x[1] for x in objs]
    sns.set_theme(style="white", palette = "colorblind")
    ax = sns.lineplot(x=it, y=obj_vals, legend="auto")
    ax.set(xlabel="iterations", ylabel="Objective function value", title="Dataset = a9a")
    return model.objs
