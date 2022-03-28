# Author : Sofiane Tanji
# License : GNU GPL V3

# Import libraries
import warnings
import sys
sys.path.append("../")
sys.path.append("../../")
warnings.simplefilter(action='ignore', category=FutureWarning)
from tqdm.contrib.itertools import product
import pandas as pd
import utils
import numpy as np
from time import perf_counter
from svm import Snacks
import seaborn as sns
from constants import BEST_VALUES, N_SAMPLES

def decision_function(w, X):
    return np.dot(w, X.T)

def predict(w, X):
    d = decision_function(w, X)
    d[d > 0] = 1
    d[d <= 0] = -1
    return d.astype(np.int32)

def score(w, X, Y):
    Ypred = predict(w, X)
    score = (Ypred == Y).sum() / len(Y)
    return score

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

def run(dset):
    oX, oY = utils.dataloader(dset)
    sns.set_theme(style=SEABORN_STYLE, palette="colorblind")
    m, g, lmbd, _ = BEST_VALUES[dset]
    Xtr, Ytr, Xts, Yts = utils.kernel_embedding(oX, oY, m, 0.8, "rbf", gamma = g)
    for i in range(8):
        model = Snacks(lmbd, n_iter = 1e6, verbose = True)
        model.fit(Xtr, Ytr)
        objs = model.objs
        it, weights = [x[0] for x in objs], [x[1] for x in objs]
        tr_scores, ts_scores = [], []
        for w in weights:
            tr_scores.append(score(w, Xtr, Ytr))
            ts_scores.append(score(w, Xts, Yts))
        data = pd.DataFrame({"it" : it, "on training set" : tr_scores, "on test set" : ts_scores})
        data.set_index("it", inplace = True)
        ax = sns.lineplot(data = data, legend="auto", markers = True)
        ax.ticklabel_format(scilimits = [-5,4], axis='x')
        ax.set(xlabel="iterations", ylabel="Accuracy", title=f"Dataset = {dset}")
        fig = ax.get_figure()
        fig.savefig(f"c_err-vs-it-{dset}-{i}.png", bbox_inches="tight")
