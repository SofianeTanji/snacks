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
from constants import BEST_VALUES

## Scientific
import utils
import numpy as np

def run_pegasos(Xtr, Ytr, Xts, Yts, nb_iterations, lambda_reg):
    model = PegasosSVMClassifier(iterations=nb_iterations, lambda_reg=lambda_reg)
    ts = time.perf_counter()
    model.fit(Xtr, Ytr)
    te = time.perf_counter()
    ts_score = model.score(Xts, Yts)
    tr_score = model.score(Xtr, Ytr)
    t_fit, tr_score, ts_score = te - ts, 1 - tr_score, 1 - ts_score
    return t_fit, tr_score, ts_score


def run_sklearn(Xtr, Ytr, Xts, Yts, gamma, lambda_reg):
    C = 1 / (2 * Xtr.shape[0] * lambda_reg)
    model = svm.SVC(C=C, gamma=gamma)
    ts = time.perf_counter()
    model.fit(Xtr, Ytr)
    te = time.perf_counter()
    ts_score = model.score(Xts, Yts)
    tr_score = model.score(Xtr, Ytr)
    t_fit, tr_score, ts_score = te - ts, 1 - tr_score, 1 - ts_score
    return t_fit, tr_score, ts_score

def run_libsvm(Xtr, Ytr, Xts, Yts, lambda_reg):
    C = 1 / (2 * Xtr.shape[0] * lambda_reg)
    model = svm.LinearSVC(C = C, loss = "hinge")
    ts = time.perf_counter()
    model.fit(Xtr, Ytr)
    te = time.perf_counter()
    ts_score = model.score(Xts, Yts)
    tr_score = model.score(Xtr, Ytr)
    t_fit, tr_score, ts_score = te - ts, 1 - tr_score, 1 - ts_score
    return t_fit, tr_score, ts_score

def run_snacks(Xtr, Ytr, Xts, Yts, nb_iterations, eta, D0, K, penalty):
    model = Snacks(nb_iterations, eta, D0, K, penalty)
    ts = time.perf_counter()
    model.fit(Xtr, Ytr)
    te = time.perf_counter()
    ts_score = model.score(Xts, Yts)
    tr_score = model.score(Xtr, Ytr)
    t_fit, tr_score, ts_score = te - ts, 1 - tr_score, 1 - ts_score
    del model
    return t_fit, tr_score, ts_score


def run_thundersvm(Xtr, Ytr, Xts, Yts, lambda_reg):
    C = 1 / (2 * Xtr.shape[0] * lambda_reg)
    tsvm = SVC(kernel = "precomputed", C=C)
    ts = time.time()
    tsvm.fit(Xtr, Ytr)
    te = time.time()
    ts_score = tsvm.score(Xts, Yts)
    tr_score = tsvm.score(Xtr, Ytr)
    t_fit, tr_score, ts_score = te - ts, 1 - tr_score, 1 - ts_score
    return t_fit, tr_score, ts_score


def compare(dataset, nb_runs):
    """Compares"""
    num_centers, gamma, n_iter, eta, D0, K, penalty, num_it_pegasos = BEST_VALUES[dataset]
    
    solution = [
        ["Snacks - on subset", None, None, None],
        ["Pegasos - on subset", None, None, None],
        ["ThunderSVM - on full", None, None, None],
        ["LibSVM - on subset", None, None, None],
    ]

    _, _, oXtr, oXts, oYtr, oYts = utils.dataloader(dataset, 0.8)
    print(f"Data is being embedded")
    Xtr, Ytr, Xts, Yts = utils.kernel_embedding(
        oXtr, oYtr, oXts, oYts, num_centers, gamma = gamma
    )
    print(f"Data is now embedded")
    run_snacks(Xtr, Ytr, Xts, Yts, 1, eta, D0, 1, penalty) # for compilation purposes

    # SNACKS
    tr_scores, ts_scores, times = [], [], []
    for i_run in range(nb_runs):
        print(f"Snacks : run {i_run + 1}/{nb_runs}")
        t_fit, tr_score, ts_score = run_snacks(Xtr, Ytr, Xts, Yts, n_iter, eta, D0, K, penalty)
        tr_scores.append(tr_score)
        ts_scores.append(ts_score)
        times.append(t_fit)

    solution[0][
        1
    ] = f"{np.round(np.mean(np.array(tr_scores)), 4)} ± {np.round(np.std(np.array(tr_scores)), 4)}"
    solution[0][
        2
    ] = f"{np.round(np.mean(np.array(ts_scores)), 4)} ± {np.round(np.std(np.array(ts_scores)), 4)}"
    solution[0][
        3
    ] = f"{np.round(np.mean(np.array(times)), 4)} ± {np.round(np.std(np.array(times)), 4)}"
    print(
        tabulate(
            solution,
            headers=["Method", "Accuracy on Train", "Accuracy on Test", "Time"],
            tablefmt="github",
        )
    )
    """
    # LibSVM
    tr_scores, ts_scores, times = [], [], []
    for i_run in range(1):
        print(f"Scikit-Learn : run {i_run + 1}/{nb_runs}")
        t_fit, tr_score, ts_score = run_sklearn(
            oXtr, oYtr, oXts, oYts, gamma, penalty
        )
        tr_scores.append(tr_score)
        ts_scores.append(ts_score)
        times.append(t_fit)

    solution[1][
        1
    ] = f"{np.round(np.mean(np.array(tr_scores)), 4)} ± {np.round(np.std(np.array(tr_scores)), 4)}"
    solution[1][
        2
    ] = f"{np.round(np.mean(np.array(ts_scores)), 4)} ± {np.round(np.std(np.array(ts_scores)), 4)}"
    solution[1][
        3
    ] = f"{np.round(np.mean(np.array(times)), 4)} ± {np.round(np.std(np.array(times)), 4)}"
    print(
        tabulate(
            solution,
            headers=["Method", "Accuracy on Train", "Accuracy on Test", "Time"],
            tablefmt="github",
        )
    )
    """
    # Pegasos
    tr_scores, ts_scores, times = [], [], []
    for i_run in range(nb_runs):
        print(f"Pegasos : run {i_run + 1}/{nb_runs}")
        t_fit, tr_score, ts_score = run_pegasos(
            Xtr, Ytr, Xts, Yts, num_it_pegasos, penalty
        )
        tr_scores.append(tr_score)
        ts_scores.append(ts_score)
        times.append(t_fit)

    solution[1][
        1
    ] = f"{np.round(np.mean(np.array(tr_scores)), 4)} ± {np.round(np.std(np.array(tr_scores)), 4)}"
    solution[1][
        2
    ] = f"{np.round(np.mean(np.array(ts_scores)), 4)} ± {np.round(np.std(np.array(ts_scores)), 4)}"
    solution[1][
        3
    ] = f"{np.round(np.mean(np.array(times)), 4)} ± {np.round(np.std(np.array(times)), 4)}"
    print(
        tabulate(
            solution,
            headers=["Method", "Accuracy on Train", "Accuracy on Test", "Time"],
            tablefmt="github",
        )
    )
    
    # ThunderSVM
    tr_scores, ts_scores, times = [], [], []
    for i_run in range(nb_runs):
        print(f"ThunderSVM : run {i_run + 1}/{nb_runs}")
        t_fit, tr_score, ts_score = run_thundersvm(
            oXtr, oYtr, oXts, oYts, penalty
        )
        tr_scores.append(tr_score)
        ts_scores.append(ts_score)
        times.append(t_fit)

    solution[2][
        1
    ] = f"{np.round(np.mean(np.array(tr_scores)), 4)} ± {np.round(np.std(np.array(tr_scores)), 4)}"
    solution[2][
        2
    ] = f"{np.round(np.mean(np.array(ts_scores)), 4)} ± {np.round(np.std(np.array(ts_scores)), 4)}"
    solution[2][
        3
    ] = f"{np.round(np.mean(np.array(times)), 4)} ± {np.round(np.std(np.array(times)), 4)}"
    print(
        tabulate(
            solution,
            headers=["Method", "Accuracy on Train", "Accuracy on Test", "Time"],
            tablefmt="github",
        )
    )

    # LibSVM 2
    tr_scores, ts_scores, times = [], [], []
    for i_run in range(nb_runs):
        print(f"LibSVM : run {i_run + 1}/{nb_runs}")
        t_fit, tr_score, ts_score = run_libsvm(Xtr, Ytr, Xts, Yts, penalty)
        tr_scores.append(tr_score)
        ts_scores.append(ts_score)
        times.append(t_fit)
    
    solution[3][1] = f"{np.round(np.mean(np.array(tr_scores)), 4)} ± {np.round(np.std(np.array(tr_scores)), 4)}"
    solution[3][2] = f"{np.round(np.mean(np.array(ts_scores)), 4)} ± {np.round(np.std(np.array(ts_scores)), 4)}"
    solution[3][3] = f"{np.round(np.mean(np.array(times)), 4)} ± {np.round(np.std(np.array(times)), 4)}"
    print(tabulate(solution, headers=["Method", "Accuracy on Train", "Accuracy on Test", "Time"], tablefmt="github"))
    

if __name__ == "__main__":
    dataset = str(sys.argv[1])
    n_runs = int(sys.argv[2])
    compare(dataset, n_runs)
