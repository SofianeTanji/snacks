# Import libraries

## System
import time
import sys
from tabulate import tabulate

sys.path.append("../")
sys.path.append("../../")

## All methods
from sklearn import svm
from constants import BEST_VALUES

import seaborn as sb
import matplotlib.pyplot as matplot
import matplotlib

## Scientific
import utils
import numpy as np

def table_print(dataset, solution, tr_scores, ts_scores, times):
    solution[0][1] = f"{np.round(np.mean(np.array(tr_scores)), 4)} ± {np.round(np.std(np.array(tr_scores)), 4)}"
    solution[0][2] = f"{np.round(np.mean(np.array(ts_scores)), 4)} ± {np.round(np.std(np.array(ts_scores)), 4)}"
    solution[0][3] = f"{np.round(np.mean(np.array(times)), 4)} ± {np.round(np.std(np.array(times)), 4)}"
    print(tabulate(solution, headers=[f"Method / {dataset}", "Accuracy on Train", "Accuracy on Test", "Time"], tablefmt="github"))
    return solution


def run_libsvm(Xtr, Ytr, Xts, Yts, g, lambda_reg):
    C = 1 / (2 * Xtr.shape[0] * lambda_reg)
    # C = 10
    model = svm.SVC(C = C, kernel = 'rbf', gamma = g)
    # model = svm.LinearSVC(C = C, loss = "hinge")
    ts = time.perf_counter()
    model.fit(Xtr, Ytr)
    te = time.perf_counter()
    ts_score = model.score(Xts, Yts)
    tr_score = model.score(Xtr, Ytr)
    t_fit, tr_score, ts_score = te - ts, 1 - tr_score, 1 - ts_score
    return t_fit, tr_score, ts_score


def test(dataset, nb_runs):
    num_centers, gamma, penalty, num_it_pegasos = BEST_VALUES[dataset]
    oX, oY = utils.dataloader(dataset)

    solution = [
        ["LibSVM - on full", None, None, None],
    ]
    print(oX.shape)
    print(f"Data is being embedded")
    time_start = time.perf_counter()
    Xtr, Ytr, Xts, Yts = utils.kernel_embedding(oX, oY, 1, gamma = gamma, tsvm = True)
    # Xtr, Ytr, Xts, Yts = utils.kernel_embedding(oX, oY, 35000, gamma = 0.01)
    time_end = time.perf_counter()
    print(f"Data embedded in {(time_end - time_start):.3f}s")

    tr_scores, ts_scores, times = [], [], []
    for i_run in range(nb_runs):
        print(f"LibSVM : run {i_run + 1}/{nb_runs}")
        t_fit, tr_score, ts_score = run_libsvm(Xtr, Ytr, Xts, Yts, gamma, penalty)
        tr_scores.append(tr_score)
        ts_scores.append(ts_score)
        times.append(t_fit)
    
    solution = table_print(dataset, solution, tr_scores, ts_scores, times)

test("SUSY", 1)
