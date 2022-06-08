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

def run_snacks(gamma, penalty, dataset, threshold, tol):
    m = 50
    score = 1
    while score > tol * threshold:
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
        t_fit, tr_score, ts_score = te - ts + time_end - time_start, 1 - tr_score, 1 - ts_score
        del model
        print(f"Score {ts_score:.3f} reached with m = {m} and needed score is {threshold:.3f}")
        m = int(1.5 * m)
        score = ts_score
    return t_fit, tr_score, ts_score, m

def run_pegasos(gamma, penalty, dataset, threshold, tol):
    m = 50
    score = 1
    while score > tol * threshold:
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
        t_fit, tr_score, ts_score = te - ts + time_end - time_start, 1 - tr_score, 1 - ts_score
        del model
        print(f"Score {ts_score:.3f} reached with m = {m} and needed score is {threshold:.3f}")
        m = int(1.5 * m)
        score = ts_score
    return t_fit, tr_score, ts_score, m

def run_liblinear(gamma, penalty, dataset, threshold, tol):
    m = 50
    score = 1
    while score > tol * threshold:
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
        t_fit, tr_score, ts_score = te - ts + time_end - time_start, 1 - tr_score, 1 - ts_score
        del model
        print(f"Score {ts_score:.3f} reached with m = {m} and needed score is {threshold:.3f}")
        m = int(2 * m)
        score = ts_score
    return t_fit, tr_score, ts_score, m

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

if __name__ == "__main__":
    dataset = str(sys.argv[1])
    _, gamma, penalty, _ = BEST_VALUES[dataset]
    tol = 1.01
    tsvm_fit, tr_threshold, ts_threshold = run_thundersvm(gamma, penalty, dataset)
    snacks_fit, snackstr, snacksts, snacks_bestm = run_snacks(gamma, penalty, dataset, ts_threshold, tol)
    peg_fit, pegasosstr, pegasosts, pegasos_bestm = run_pegasos(gamma, penalty, dataset, ts_threshold, tol)
    lib_fit, libtr, libts, lib_bestm = run_liblinear(gamma, penalty, dataset, ts_threshold, tol)
    solution = [
        ["ThunderSVM", tr_threshold, ts_threshold, None, tsvm_fit],
        ["Pegasos - good m", pegasosstr, pegasosts, pegasos_bestm, peg_fit],
        ["Snacks - good m", snackstr, snacksts, snacks_bestm, snacks_fit],
        ["LibLinear - good m", libtr, libts, lib_bestm, lib_fit],
    ]
    print(tabulate(solution, headers=[f"Method / {dataset}", "Accuracy on Train", "Accuracy on Test", "Best M", "Training time"], tablefmt="github"))

