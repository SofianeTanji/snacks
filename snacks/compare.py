# Import libraries

## System
import time
import sys
from tabulate import tabulate

sys.path.append("../")

## All methods
from thundersvm import SVC
from pegasos import PegasosSVMClassifier
from sklearn import svm
from svm import Snacks

## Scientific
import utils
import numpy as np

def run_pegasos(Xtr, Ytr, Xts, Yts, nb_iterations, lambda_reg):
    print("Pegasos' performance : ")
    model = PegasosSVMClassifier(iterations=nb_iterations, lambda_reg=lambda_reg)
    ts = time.perf_counter()
    model.fit(Xtr, Ytr)
    te = time.perf_counter()
    ts_score = model.score(Xts, Yts)
    print(f"in {(te - ts):.2f}s, C-err is {100 - ts_score * 100:.2f}%")
    tr_score = model.score(Xtr, Ytr)
    print(f"also, train error is {100 - tr_score * 100:.2f}%")
    t_fit, tr_score, ts_score = te - ts, 1 - tr_score, 1 - ts_score
    return t_fit, tr_score, ts_score

def run_sklearn(Xtr, Ytr, Xts, Yts, gamma, lambda_reg):
    print("SKLearn's performance : ")
    C = 1 / (2 * Xtr.shape[0] * lambda_reg)
    model = svm.SVC(C = C, gamma = gamma)
    ts = time.perf_counter()
    model.fit(Xtr, Ytr)
    te = time.perf_counter()
    ts_score = model.score(Xts, Yts)
    print(f"in {(te - ts):.2f}s, C-err is {100 - ts_score * 100:.2f}%")
    tr_score = model.score(Xtr, Ytr)
    print(f"also, train error is {100 - tr_score * 100:.2f}%")
    t_fit, tr_score, ts_score = te - ts, 1 - tr_score, 1 - ts_score
    return t_fit, tr_score, ts_score

def run_snacks(Xtr, Ytr, Xts, Yts, nb_iterations, lambda_reg, stepsize):
    print("Snacks' performance : ")
    model = Snacks(
        nb_iterations=nb_iterations, lambda_reg=lambda_reg, stepsize=stepsize
    )
    ts = time.perf_counter()
    model.fit(Xtr, Ytr)
    te = time.perf_counter()
    ts_score = model.score(Xts, Yts)
    print(f"in {(te - ts):.2f}s, C-err is {100 - ts_score * 100:.2f}%")
    tr_score = model.score(Xtr, Ytr)
    print(f"also, train error is {100 - tr_score * 100:.2f}%")
    t_fit, tr_score, ts_score = te - ts, 1 - tr_score, 1 - ts_score
    return t_fit, tr_score, ts_score

def run_thundersvm(Xtr, Ytr, Xts, Yts, lambda_reg):
    print("ThunderSVM's performance : ")
    C = 1 / (2 * Xtr.shape[0] * lambda_reg)
    tsvm = SVC(gamma = 1e-1, C=C)
    ts = time.time()
    tsvm.fit(Xtr, Ytr)
    te = time.time()
    ts_score = tsvm.score(Xts, Yts)
    print(f"in {(te - ts):.2f}s, C-err is {100 - ts_score * 100:.2f}%")
    tr_score = tsvm.score(Xtr, Ytr)
    print(f"also, train error is {100 - tr_score * 100:.2f}%")
    t_fit, tr_score, ts_score = te - ts, 1 - tr_score, 1 - ts_score
    return t_fit, tr_score, ts_score

def run_liquidsvm(Xtr, Ytr, Xts, Yts, lambda_reg):
    print("ThunderSVM's performance : ")
    C = 1 / (2 * Xtr.shape[0] * lambda_reg)
    model = SVC(gamma = 1e-1, C=C, verbose = True)
    ts = time.time()
    model.fit(Xtr, Ytr)
    te = time.time()
    ts_score = model.score(Xts, Yts)
    print(f"in {(te - ts):.2f}s, C-err is {100 - ts_score * 100:.2f}%")
    tr_score = model.score(Xtr, Ytr)
    print(f"also, train error is {100 - tr_score * 100:.2f}%")
    t_fit, tr_score, ts_score = te - ts, 1 - tr_score, 1 - ts_score
    return t_fit, tr_score, ts_score

def compare(dataset, nb_runs):
    """Compares"""
    values = {
        "a9a": [1e-3, 1e-6, 1000],
        "SUSY": [1e-1, 5e-8, 2500],
        "HIGGS": [3e-2, 3e-8, 1.2e5] # à recalibrer
    }
    solution = [
        ["Snacks - on subset", None, None, None],
        ["LibSVM - on full", None, None, None],
        ["Pegasos - on subset", None, None, None],
        ["ThunderSVM - on full", None, None, None],
        ["LiquidSVM - on full", None, None, None]
    ]
    oXtr, oXts, oYtr, oYts = utils.dataloader(dataset, 0.8)
    Xtr, Ytr, Xts, Yts = utils.kernel_embedding(oXtr, oYtr, oXts, oYts, values[dataset][2], gamma=values[dataset][0])
    run_snacks(Xtr, Ytr, Xts, Yts, 3, 1e-7, 1.)

    # SNACKS
    tr_scores, ts_scores, times = [], [], []
    for i_run in range(nb_runs):
        print(f"Snacks : run {i_run + 1}/{nb_runs}")
        t_fit, tr_score, ts_score = run_snacks(Xtr, Ytr, Xts, Yts, 35000, values[dataset][1], 1.)
        tr_scores.append(tr_score)
        ts_scores.append(ts_score)
        times.append(t_fit)
    
    solution[0][1] = f"{np.round(np.mean(np.array(tr_scores)), 4)} ± {np.round(np.std(np.array(tr_scores)), 4)}"
    solution[0][2] = f"{np.round(np.mean(np.array(ts_scores)), 4)} ± {np.round(np.std(np.array(ts_scores)), 4)}"
    solution[0][3] = f"{np.round(np.mean(np.array(times)), 4)} ± {np.round(np.std(np.array(times)), 4)}"
    print(tabulate(solution, headers=["Method", "Accuracy on Train", "Accuracy on Test", "Time"], tablefmt="github"))

    # LibSVM
    tr_scores, ts_scores, times = [], [], []
    for i_run in range(1):
        print(f"Scikit-Learn : run {i_run + 1}/{nb_runs}")
        t_fit, tr_score, ts_score = run_sklearn(oXtr, oYtr, oXts, oYts, values[dataset][0], values[dataset][1])
        tr_scores.append(tr_score)
        ts_scores.append(ts_score)
        times.append(t_fit)
    
    solution[1][1] = f"{np.round(np.mean(np.array(tr_scores)), 4)} ± {np.round(np.std(np.array(tr_scores)), 4)}"
    solution[1][2] = f"{np.round(np.mean(np.array(ts_scores)), 4)} ± {np.round(np.std(np.array(ts_scores)), 4)}"
    solution[1][3] = f"{np.round(np.mean(np.array(times)), 4)} ± {np.round(np.std(np.array(times)), 4)}"
    print(tabulate(solution, headers=["Method", "Accuracy on Train", "Accuracy on Test", "Time"], tablefmt="github"))

    # Pegasos
    tr_scores, ts_scores, times = [], [], []
    for i_run in range(nb_runs):
        print(f"Pegasos : run {i_run + 1}/{nb_runs}")
        t_fit, tr_score, ts_score = run_pegasos(Xtr, Ytr, Xts, Yts, 300000, values[dataset][1])
        tr_scores.append(tr_score)
        ts_scores.append(ts_score)
        times.append(t_fit)
    
    solution[2][1] = f"{np.round(np.mean(np.array(tr_scores)), 4)} ± {np.round(np.std(np.array(tr_scores)), 4)}"
    solution[2][2] = f"{np.round(np.mean(np.array(ts_scores)), 4)} ± {np.round(np.std(np.array(ts_scores)), 4)}"
    solution[2][3] = f"{np.round(np.mean(np.array(times)), 4)} ± {np.round(np.std(np.array(times)), 4)}"
    print(tabulate(solution, headers=["Method", "Accuracy on Train", "Accuracy on Test", "Time"], tablefmt="github"))
    
    # ThunderSVM
    tr_scores, ts_scores, times = [], [], []
    for i_run in range(nb_runs):
        print(f"ThunderSVM : run {i_run + 1}/{nb_runs}")
        t_fit, tr_score, ts_score = run_thundersvm(oXtr, oYtr, oXts, oYts, values[dataset][1])
        tr_scores.append(tr_score)
        ts_scores.append(ts_score)
        times.append(t_fit)
    
    solution[3][1] = f"{np.round(np.mean(np.array(tr_scores)), 4)} ± {np.round(np.std(np.array(tr_scores)), 4)}"
    solution[3][2] = f"{np.round(np.mean(np.array(ts_scores)), 4)} ± {np.round(np.std(np.array(ts_scores)), 4)}"
    solution[3][3] = f"{np.round(np.mean(np.array(times)), 4)} ± {np.round(np.std(np.array(times)), 4)}"
    print(tabulate(solution, headers=["Method", "Accuracy on Train", "Accuracy on Test", "Time"], tablefmt="github"))
    """
    # LiquidSVM
    tr_scores, ts_scores, times = [], [], []
    for i_run in range(nb_runs):
        print(f"LiquidSVM : run {i_run + 1}/{nb_runs}")
        t_fit, tr_score, ts_score = run_liquidsvm(oXtr, oYtr, oXts, oYts, values[dataset][1])
        tr_scores.append(tr_score)
        ts_scores.append(ts_score)
        times.append(t_fit)
    
    solution[4][1] = f"{np.round(np.mean(np.array(tr_scores)), 4)} ± {np.round(np.std(np.array(tr_scores)), 4)}"
    solution[4][2] = f"{np.round(np.mean(np.array(ts_scores)), 4)} ± {np.round(np.std(np.array(ts_scores)), 4)}"
    solution[4][3] = f"{np.round(np.mean(np.array(times)), 4)} ± {np.round(np.std(np.array(times)), 4)}"
    print(tabulate(solution, headers=["Method", "Accuracy on Train", "Accuracy on Test", "Time"], tablefmt="github"))
    """
if __name__ == "__main__":
    dataset = str(sys.argv[1])
    print(f"Dataset {dataset} chosen")
    compare(dataset, 10)