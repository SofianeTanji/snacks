# Author : Sofiane Tanji
# License : GNU GPL V3

# Import libraries

## System
import time
import sys
from tabulate import tabulate

sys.path.append("../")
sys.path.append("../../")

## All methods
from thundersvm import SVC
from pegasos import PegasosSVMClassifier
from sklearn import svm
from svm import Snacks

## Gamma, Lambda grid
from constants import BEST_VALUES

## Scientific
import utils
import numpy as np
import seaborn as sns

## Obtain best accuracy
def run_thundersvm(gamma, penalty, dataset):
    oX, oY = utils.dataloader(dataset)
    time_start = time.perf_counter()
    Xtr, Ytr, Xts, Yts = utils.kernel_embedding(oX, oY, 2, gamma = gamma, tsvm = True)
    time_end = time.perf_counter()
    print(f"Data uploaded in {(time_end - time_start):.3f}s")
    C = 1 / (2 * Xtr.shape[0] * penalty)
    tsvm = SVC(kernel = "rbf", C=C, gamma = gamma)
    ts = time.perf_counter()
    tsvm.fit(Xtr, Ytr)
    te = time.perf_counter()
    ts_score = tsvm.score(Xts, Yts)
    tr_score = tsvm.score(Xtr, Ytr)
    t_fit, tr_score, ts_score = time_end - time_start + te - ts, 1 - tr_score, 1 - ts_score
    print(f"Best score on training set : {ts_score:.3f}")
    print(f"Best score on test set : {ts_score:.3f}")
    print(f"Total cpu time for training : {(te - ts):.3f} seconds")
    return t_fit, tr_score, ts_score

def run_snacks(gamma, penalty, dataset, threshold, t_threshold, tol):
    m = 100
    score = 1
    my_time = 0
    trmem, tsmem, fitmem = [], [], []
    while score > tol * threshold and t_threshold > my_time:
        oX, oY = utils.dataloader(dataset)
        time_start = time.perf_counter()
        Xtr, Ytr, Xts, Yts = utils.kernel_embedding(oX, oY, m, gamma = gamma)
        time_end = time.perf_counter()
        print(f"Data embedded in {(time_end - time_start):.3f}s")
        model = Snacks(penalty)
        empty = Snacks(penalty, n_iter = 1)
        empty.fit(Xtr, Ytr)
        ts = time.perf_counter()
        model.fit(Xtr, Ytr)
        te = time.perf_counter()
        ts_score = model.score(Xts, Yts)
        tr_score = model.score(Xtr, Ytr)
        trmem.append(1 - tr_score)
        tsmem.append(1 - ts_score)
        fitmem.append(te - ts + time_end - time_start)        
        t_fit, tr_score, ts_score = te - ts + time_end - time_start, 1 - tr_score, 1 - ts_score
        del model
        print(f"Score {ts_score:.3f} reached with m = {m} and needed score is {threshold:.3f}")
        m, score, my_time = m + 100, ts_score, t_fit
    return fitmem, trmem, tsmem, m

def run_pegasos(gamma, penalty, dataset, threshold, t_threshold, tol):
    m = 100
    score = 1
    my_time = 0
    trmem, tsmem, fitmem = [], [], []
    while score > tol * threshold and t_threshold > my_time:
        oX, oY = utils.dataloader(dataset)
        time_start = time.perf_counter()
        Xtr, Ytr, Xts, Yts = utils.kernel_embedding(oX, oY, m, gamma = gamma)
        time_end = time.perf_counter()
        print(f"Data embedded in {(time_end - time_start):.3f}s")
        C = 1 / (2 * Xtr.shape[0] * penalty)
        model = svm.LinearSVC(C = C, loss = "hinge")
        ts = time.perf_counter()
        model.fit(Xtr, Ytr)
        te = time.perf_counter()
        ts_score = model.score(Xts, Yts)
        tr_score = model.score(Xtr, Ytr)
        trmem.append(1 - tr_score)
        tsmem.append(1 - ts_score)
        fitmem.append(te - ts + time_end - time_start)
        t_fit, tr_score, ts_score = te - ts + time_end - time_start, 1 - tr_score, 1 - ts_score
        del model
        print(f"Score {ts_score:.3f} reached with m = {m} and needed score is {threshold:.3f}")
        m, score, my_time = m + 100, ts_score, t_fit
    return fitmem, trmem, tsmem, m

def run_liblinear(gamma, penalty, dataset, threshold, t_threshold, tol):
    m = 100
    score = 1
    my_time = 0
    trmem, tsmem, fitmem = [], [], []
    while score > tol * threshold and t_threshold > my_time:
        oX, oY = utils.dataloader(dataset)
        time_start = time.perf_counter()
        Xtr, Ytr, Xts, Yts = utils.kernel_embedding(oX, oY, m, gamma = gamma)
        time_end = time.perf_counter()
        print(f"Data embedded in {(time_end - time_start):.3f}s")
        model = Snacks(penalty)
        ts = time.perf_counter()
        model.fit(Xtr, Ytr)
        te = time.perf_counter()
        ts_score = model.score(Xts, Yts)
        tr_score = model.score(Xtr, Ytr)
        trmem.append(1 - tr_score)
        tsmem.append(1 - ts_score)
        fitmem.append(te - ts + time_end - time_start)
        t_fit, tr_score, ts_score = te - ts + time_end - time_start, 1 - tr_score, 1 - ts_score
        del model
        print(f"Score {ts_score:.3f} reached with m = {m} and needed score is {threshold:.3f}")
        m, score, my_time = m + 100, ts_score, t_fit
    return fitmem, trmem, tsmem, m

def table_print(method, solution, tr_scores, ts_scores, times):
    if method == "Snacks":
        idx = 0
    elif method == "Pegasos":
        idx = 1
    elif method == "LibSVM":
        idx = 2
    elif method == "ThunderSVM":
        idx = 3
    else:
        assert False, "Unknown method"
    solution[idx][1] = f"{np.round(np.mean(np.array(tr_scores)), 4)} ± {np.round(np.std(np.array(tr_scores)), 4)}"
    solution[idx][2] = f"{np.round(np.mean(np.array(ts_scores)), 4)} ± {np.round(np.std(np.array(ts_scores)), 4)}"
    solution[idx][3] = f"{np.round(np.mean(np.array(times)), 4)} ± {np.round(np.std(np.array(times)), 4)}"
    print(tabulate(solution, headers=[f"Method / {dataset}", "Accuracy on Train", "Accuracy on Test", "Time"], tablefmt="github"))
    return solution

def lengths(x):
    if isinstance(x,list):
        yield len(x)
        for y in x:
            yield from lengths(y)

if __name__ == "__main__":
    dataset = str(sys.argv[1])
    _, gamma, penalty, _ = BEST_VALUES[dataset]
    tol = 1.
    if dataset in ["SUSY", "HIGGS"]:
        tr_threshold, ts_threshold, tsvm_fit = 0.21, 0.21, 600000
    else:
        tsvm_fit, tr_threshold, ts_threshold = run_thundersvm(gamma, penalty, dataset)
    snacks_fit, snackstr, snacksts, snacks_bestm = run_snacks(gamma, penalty, dataset, ts_threshold, 2 * tsvm_fit, tol)
    peg_fit, pegasostr, pegasosts, pegasos_bestm = run_pegasos(gamma, penalty, dataset, ts_threshold, 2 * tsvm_fit, tol)
    lib_fit, libtr, libts, lib_bestm = run_liblinear(gamma, penalty, dataset, ts_threshold, 2 * tsvm_fit, tol)
    idx_snacks = snacksts.index(min(snacksts))
    idx_peg = pegasosts.index(min(pegasosts))
    idx_lib = libts.index(min(libts))
    solution = [
        ["ThunderSVM", tr_threshold, ts_threshold, None, tsvm_fit],
        ["Pegasos - good m", pegasostr[idx_peg], pegasosts[idx_peg], 100 * (idx_peg + 1), peg_fit[idx_peg]],
        ["Snacks - good m", snackstr[idx_snacks], snacksts[idx_snacks], 100 * (idx_snacks + 1), snacks_fit[idx_snacks]],
        ["LibLinear - good m", libtr[idx_lib], libts[idx_lib], 100 * (idx_lib + 1), lib_fit[idx_lib]]
    ]
    print(tabulate(solution, headers=[f"Method / {dataset}", "Accuracy on Train", "Accuracy on Test", "Best M", "Training time"], tablefmt="github"))
    figure_data = {"m" : [100 * (k + 1) for k in range(lengths([snacksts, pegasosts, libts]))], "Snacks" : snacksts, "Pegasos" : pegasosts, "LibLinear" : libts}
    ax = sns.lineplot(data = figure_data, legend="auto", markers = True)
    ax.ticklabel_format(scilimits = [-5,4], axis='x')
    ax.set(xlabel="m", ylabel="Classification error", title=f"Dataset = {dataset}")
    fig = ax.get_figure()
    fig.savefig(f"../../figures/best-m-{dataset}.png", bbox_inches="tight")