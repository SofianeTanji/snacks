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
    C = 1 / (2 * Xtr.shape[0] * lambda_reg)
    model = PegasosSVMClassifier(iterations=nb_iterations, lambda_reg=lambda_reg)
    ts = time.perf_counter()
    model.fit(Xtr, Ytr)
    te = time.perf_counter()
    score = model.score(Xts, Yts)
    print(f"in {(te - ts):.2f}s, C-err is {100 - score * 100:.2f}%")
    tr_score = model.score(Xtr, Ytr)
    print(f"also, train error is {100 - tr_score * 100:.2f}%")
    t_fit, score = te - ts, 1 - score
    return t_fit, score

def run_sklearn(Xtr, Ytr, Xts, Yts, lambda_reg):
    print("SKLearn's performance : ")
    C = 1 / (2 * Xtr.shape[0] * lambda_reg)
    model = svm.LinearSVC(C=C, loss = "hinge")
    ts = time.perf_counter()
    model.fit(Xtr, Ytr)
    te = time.perf_counter()
    score = model.score(Xts, Yts)
    print(f"in {(te - ts):.2f}s, C-err is {100 - score * 100:.2f}%")
    t_fit, score = te - ts, 1 - score
    return t_fit, score

def run_snacks(Xtr, Ytr, Xts, Yts, nb_iterations, lambda_reg, stepsize):
    print("Snacks' performance : ")
    model = Snacks(
        nb_iterations=nb_iterations, lambda_reg=lambda_reg, stepsize=stepsize
    )
    ts = time.perf_counter()
    model.fit(Xtr, Ytr)
    te = time.perf_counter()
    score = model.score(Xts, Yts)
    print(f"in {(te - ts):.2f}s, C-err is {100 - score * 100:.2f}%")
    tr_score = model.score(Xtr, Ytr)
    print(f"also, train error is {100 - tr_score * 100:.2f}%")
    t_fit, score = te - ts, 1 - score
    return t_fit, score

def run_thundersvm(Xtr, Ytr, Xts, Yts, lambda_reg):
    print("ThunderSVM's performance : ")
    C = 1 / (2 * Xtr.shape[0] * lambda_reg)
    tsvm = SVC(gamma = 1e-1, C=C, verbose = True)
    ts = time.time()
    tsvm.fit(Xtr, Ytr)
    te = time.time()
    score = tsvm.score(Xts, Yts)
    print(f"in {(te - ts):.2f}s, C-err is {100 - score * 100:.2f}%")
    t_fit, score = te - ts, 1 - score
    return t_fit, score

def compare(dataset):
    """Compares"""
    values = {
        "a9a": [1e-3, 1e-6, 1000],
        "SUSY": [1e-1, 5e-8, 2500],
        "HIGGS": [3e-2, 3e-8, 1.2e5] # à recalibrer
    }
    solution = [
        ["Snacks", None, None],
        ["LibSVM", None, None],
        ["Pegasos", None, None],
        ["LiquidSVM", None, None],
        ["ThunderSVM", None, None]
    ]
    oXtr, oXts, oYtr, oYts = utils.dataloader(dataset, 0.8)
    Xtr, Ytr, Xts, Yts = utils.kernel_embedding(oXtr, oYtr, oXts, oYts, values[dataset][2], gamma=values[dataset][0])
    run_snacks(Xtr, Ytr, Xts, Yts, 3, 1e-7, 1.)

    # SNACKS
    scores, times = [], []
    for i_run in range(5):
        print(f"Snacks : run {i_run + 1}/5")
        t_fit, score = run_snacks(Xtr, Ytr, Xts, Yts, 35000, values[dataset][1], 1.)
        scores.append(score)
        times.append(t_fit)
    
    solution[0][1] = f"{np.mean(np.array(scores))} ± {np.std(np.array(scores))}"
    solution[0][2] = f"{np.mean(np.array(times))} ± {np.std(np.array(times))}"
    print(tabulate(solution, headers=["Method", "Accuracy", "Time"], tablefmt="github"))

    # LibSVM
    scores, times = [], []
    for i_run in range(5):
        print(f"Scikit-Learn : run {i_run + 1}/5")
        t_fit, score = run_sklearn(Xtr, Ytr, Xts, Yts, values[dataset][1])
        scores.append(score)
        times.append(t_fit)
    
    solution[1][1] = f"{np.mean(np.array(scores))} ± {np.std(np.array(scores))}"
    solution[1][2] = f"{np.mean(np.array(times))} ± {np.std(np.array(times))}"
    print(tabulate(solution, headers=["Method", "Accuracy", "Time"], tablefmt="github"))

    # Pegasos
    scores, times = [], []
    for i_run in range(5):
        print(f"Pegasos : run {i_run + 1}/5")
        t_fit, score = run_pegasos(Xtr, Ytr, Xts, Yts, 210000 * 3, values[dataset][1])
        scores.append(score)
        times.append(t_fit)
    
    solution[2][1] = f"{np.mean(np.array(scores, dtype = np.float32))} ± {np.std(np.array(scores, dtype = np.float32))}"
    solution[2][2] = f"{np.mean(np.array(times, dtype = np.float32))} ± {np.std(np.array(times, dtype = np.float32))}"
    print(tabulate(solution, headers=["Method", "Accuracy", "Time"], tablefmt="github"))
    
    # ThunderSVM
    scores, times = [], []
    for i_run in range(5):
        print(f"ThunnderSVM : run {i_run + 1}/5")
        t_fit, score = run_thundersvm(Xtr, Ytr, Xts, Yts, 120000 * 3, values[dataset][1])
        scores.append(score)
        times.append(t_fit)
    
    solution[4][1] = f"{np.mean(np.array(scores, dtype = np.float32))} ± {np.std(np.array(scores, dtype = np.float32))}"
    solution[4][2] = f"{np.mean(np.array(times, dtype = np.float32))} ± {np.std(np.array(times, dtype = np.float32))}"
    print(tabulate(solution, headers=["Method", "Accuracy", "Time"], tablefmt="github"))


if __name__ == "__main__":
    dataset = str(sys.argv[1])
    print(f"Dataset {dataset} chosen")
    compare(dataset)