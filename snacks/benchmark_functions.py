from svm import Snacks
import seaborn as sns
import time
import sys
from sklearn.linear_model import RidgeClassifier
from itertools import product
import utils
import numpy as np

SEABORN_STYLE = {
    "figure.facecolor": "white",
    "axes.labelcolor": ".15",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.color": ".15",
    "ytick.color": ".15",
    "axes.axisbelow": True,
    "grid.linestyle": "-",
    "text.color": ".15",
    "font.family": ["sans-serif"],
    "font.sans-serif": ["Helvetica", "DejaVu Sans"],
    "lines.solid_capstyle": "round",
    "patch.edgecolor": "w",
    "patch.force_edgecolor": True,
    "image.cmap": "rocket",
    "xtick.top": False,
    "ytick.right": False,
    "axes.grid": True,
    "axes.facecolor": "white",
    "axes.edgecolor": "white",
    "grid.color": "white",
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "axes.spines.right": True,
    "axes.spines.top": True,
    "xtick.bottom": False,
    "ytick.left": False,
}


def cerr_vs_m(
    dataset, train_size, gamma, nb_iterations, nb_runs, lambda_reg, stepsize, arr_m
):
    oXtr, oXts, oYtr, oYts = utils.dataloader(dataset, train_size)
    cerrs_vs_m = []
    for m in arr_m:
        m = int(m)
        print(f"m = {m}, let's go ! ")
        Xtr, Ytr, Xts, Yts = utils.kernel_embedding(
            oXtr, oYtr, oXts, oYts, m, gamma=gamma
        )
        empty = odel = Snacks(nb_iterations=3, lambda_reg=lambda_reg, stepsize=stepsize)
        empty.fit(Xtr, Ytr)
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
    sns.set_theme(style=SEABORN_STYLE, palette="colorblind")
    ax = sns.lineplot(x=Cerrs, y=Ms, legend="auto")
    ax.set(xlabel="m", ylabel="C-err", title=f"Dataset = {dataset}")
    fig = ax.get_figure()
    fig.savefig(f"../figures/c_err-vs-m-{dataset}.png", bbox_inches="tight")
    del fig
    del ax
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
    sns.set_theme(style=SEABORN_STYLE, palette="colorblind")
    ax = sns.lineplot(x=Cerrs, y=Lmbds, legend="auto")
    ax.set_xscale("log")
    ax.set(
        xlabel="Regularization parameter", ylabel="C-err", title=f"Dataset = {dataset}"
    )
    fig = ax.get_figure()
    fig.savefig(f"../figures/c_err-vs-lmbd-{dataset}.png", bbox_inches="tight")
    del fig
    del ax
    return Lmbds, Cerrs


def cerr_m_lmbd(
    dataset, train_size, gamma, nb_iterations, nb_runs, stepsize, arr_m, arr_l,
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
    sns.set_theme(style=SEABORN_STYLE, palette="colorblind")
    cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True)
    ax = sns.heatmap(
        Cerr.T, xticklabels=arr_m, yticklabels=arr_l, cmap=cmap, square=True
    )
    ax.set(xlabel="m", ylabel="Lambda", title=f"Dataset = {dataset}")
    fig = ax.get_figure()
    fig.savefig(f"../figures/c_err-m-lmbd-{dataset}.png", bbox_inches="tight")
    del fig
    del ax
    return Cerr, arr_m, arr_l


def heatmap(
    dataset, train_size, gamma, nb_runs, arr_m, arr_l,
):
    oXtr, oXts, oYtr, oYts = utils.dataloader(dataset, train_size)

    Cerr = []
    for m, lmbd in product(arr_m, arr_l):
        Xtr, Ytr, Xts, Yts = utils.kernel_embedding(
            oXtr, oYtr, oXts, oYts, m, gamma=gamma
        )
        scores = []
        for n in range(nb_runs):
            model = RidgeClassifier(alpha=lmbd, solver="lsqr")
            model.fit(Xtr, Ytr)
            scores.append(1 - model.score(Xts, Yts))
            del model
        Cerr.append(np.mean(scores))

    Cerr = np.reshape(np.array(Cerr), (arr_m.size, arr_l.size))
    sns.set_theme(style=SEABORN_STYLE, palette="colorblind")
    cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True)
    ax = sns.heatmap(
        Cerr.T, xticklabels=arr_m, yticklabels=arr_l, cmap=cmap, square=True
    )
    ax.set(xlabel="m", ylabel="Lambda", title=f"Dataset = {dataset}")
    fig = ax.get_figure()
    fig.savefig(f"../figures/c_err-m-lmbd-{dataset}-krr.png", bbox_inches="tight")
    del fig
    del ax
    return Cerr, arr_m, arr_l


def grid_search_map(
    arr_lambda, dataset, train_size, nb_iterations, m, nb_runs, arr_gamma
):
    oXtr, oXts, oYtr, oYts = utils.dataloader(dataset, train_size)

    Cerr = []
    for g, l in product(arr_gamma, arr_lambda):
        Xtr, Ytr, Xts, Yts = utils.kernel_embedding(oXtr, oYtr, oXts, oYts, m, gamma=g)
        scores = []
        for _ in range(nb_runs):
            model = Snacks(nb_iterations=nb_iterations, lambda_reg=l, stepsize=1.0)
            model.fit(Xtr, Ytr)
            scores.append(1 - model.score(Xts, Yts))
            del model
        Cerr.append(np.mean(scores))
    Cerr = np.reshape(np.array(Cerr), (arr_gamma.size, arr_lambda.size))
    sns.set_theme(style=SEABORN_STYLE, palette="colorblind")
    cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True)
    arrg = np.array(
        [np.format_float_scientific(x, unique=False, precision=1) for x in arr_gamma]
    )
    arrg.astype(np.float32)
    ax = sns.heatmap(
        Cerr.T, xticklabels=arrg, yticklabels=arr_lambda, cmap=cmap, square=True
    )
    ax.set(xlabel="Gamma", ylabel="Lambda", title=f"Dataset = {dataset}")
    fig = ax.get_figure()
    fig.savefig(f"../figures/heatmap-gamma-lmbd-{dataset}.png", bbox_inches="tight")
    del fig
    del ax
    return Cerr, arr_gamma, arr_lambda


def grid_search_krr(dataset, train_size, m, nb_runs, arr_gamma, arr_lambda):
    oXtr, oXts, oYtr, oYts = utils.dataloader(dataset, train_size)

    Cerr = []
    for g, l in product(arr_gamma, arr_lambda):
        Xtr, Ytr, Xts, Yts = utils.kernel_embedding(oXtr, oYtr, oXts, oYts, m, gamma=g)
        scores = []
        for _ in range(nb_runs):
            model = RidgeClassifier(alpha=l, solver="lsqr")
            model.fit(Xtr, Ytr)
            scores.append(1 - model.score(Xts, Yts))
            del model
        Cerr.append(np.mean(scores))
    Cerr = np.reshape(np.array(Cerr), (arr_gamma.size, arr_lambda.size))
    sns.set_theme(style=SEABORN_STYLE, palette="colorblind")
    cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True)
    arrg = np.array(
        [np.format_float_scientific(x, unique=False, precision=1) for x in arr_gamma]
    )
    arrg.astype(np.float32)
    ax = sns.heatmap(
        Cerr.T, xticklabels=arrg, yticklabels=arr_lambda, cmap=cmap, square=True
    )
    ax.set(xlabel="Gamma", ylabel="Lambda", title=f"Dataset = {dataset}")
    fig = ax.get_figure()
    fig.savefig(f"../figures/heatmap-gamma-lmbd-{dataset}-krr.png", bbox_inches="tight")
    del fig
    del ax
    return Cerr, arr_gamma, arr_lambda


def obj_vs_it(
    dataset, train_size, gamma, m, lmbd, stepsize, nb_iterations,
):
    # Empty run snacks
    empty = Snacks(nb_iterations=5, lambda_reg=lmbd, stepsize=stepsize, verbose=False)
    oXtr, oXts, oYtr, oYts = utils.dataloader(dataset, train_size)
    Xtr, Ytr, _, _ = utils.kernel_embedding(oXtr, oYtr, oXts, oYts, m, gamma=gamma)
    model = Snacks(
        nb_iterations=nb_iterations, lambda_reg=lmbd, stepsize=stepsize, verbose=True
    )
    empty.fit(
        np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]], dtype=np.float32),
        np.array([1, 1, 0, 0], dtype=np.float32),
    )
    model.fit(Xtr, Ytr)
    objs = model.objs
    it, obj_vals = [x[0] for x in objs], [x[1] for x in objs]
    sns.set_theme(style=SEABORN_STYLE, palette="colorblind")
    print(len(it))
    ax = sns.lineplot(x=it, y=obj_vals, legend="auto", markers=True)
    ax.set(
        xlabel="iterations", ylabel="Objective function value", title="Dataset = a9a"
    )
    fig = ax.get_figure()
    fig.savefig(f"../figures/obj-vs-it-{dataset}.png", bbox_inches="tight")
    del fig
    del ax
    return model.objs


if __name__ == "__main__":
    route = sys.argv[1]
    data = str(sys.argv[2])

    if data not in ["a9a", "SUSY", "HIGGS"]:
        raise ValueError(
            "Please provide one of the following values : a9a, SUSY, HIGGS"
        )

    arr_m = np.array([10, 20, 50, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000])
    arr_l = np.array(
        [1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
    )

    arrg = np.logspace(-5, -1, 12)
    arrm = np.linspace(100, 2200, 5)
    arrl = np.logspace(-9, -1, 40)

    if route == "obj_vs_it":  # peut en avoir une plus jolie
        print("Objective value vs It")
        ts = time.perf_counter()
        obj_vs_it(data, 0.8, 1e-3, 1000, 1e-6, 1.0, 1e6)
        te = time.perf_counter()
        print(f"Benchmarking done in {(te - ts):.2f} seconds")
    elif route == "cerr_vs_m":
        print("Cerr VS M")  # bah ya pas de diff mdrr
        ts = time.perf_counter()
        cerr_vs_m(data, 0.8, 1e-3, 35000, 15, 1e-6, 1.0, arrm)
        te = time.perf_counter()
        print(f"Benchmarking done in {(te - ts):.2f} seconds")
    elif route == "cerr_vs_lmbd":
        print("Cerr VS Lambda")
        ts = time.perf_counter()
        cerr_vs_lmbd(data, 0.8, 1e-3, 35000, 5, 1000, 1.0, arrl)
        te = time.perf_counter()
        print(f"Benchmarking done in {(te - ts):.2f} seconds")
    elif route == "grid_search_snacks":
        print("Grid Search Snacks")
        ts = time.perf_counter()
        if data == "a9a":
            grid_search_map(arr_l, data, 0.8, 35000, 1000, 5, arrg)
        elif data == "SUSY":
            grid_search_map(arr_l, data, 0.8, 35000, 2500, 5, arrg)
        te = time.perf_counter()
        print(f"Benchmarking done in {(te - ts):.2f} seconds")
    elif route == "cerr_m_lmbd":
        print("Combined behaviour")
        ts = time.perf_counter()
        cerr_m_lmbd(data, 0.8, 1e-3, 35000, 5, 1.0, arr_m, arr_l)
        te = time.perf_counter()
        print(f"Benchmarking done in {(te - ts):.2f} seconds")
    elif route == "heatmap":
        print("Combined behaviour KRR")
        ts = time.perf_counter()
        heatmap(data, 0.8, 1e-3, 5, arr_m, arr_l)
        te = time.perf_counter()
        print(f"Benchmarking done in {(te - ts):.2f} seconds")
    elif route == "grid_search_krr":
        print("Grid Search KRR")
        ts = time.perf_counter()
        grid_search_krr(data, 0.8, 1000, 5, arrg, arr_l)
        te = time.perf_counter()
        print(f"Benchmarking done in {(te - ts):.2f} seconds")
    else:
        print(f"{route} is not part of the benchmark functions implemented.")

    # Sur a9a, les heatmap-gamma-lmbd ne sont pas "arrangeantes"
