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
from sklearn.model_selection import train_test_split
from svm import Snacks
from constants import BEST_VALUES
from sklearn.metrics import f1_score

## Scientific
import utils
import numpy as np

data = np.load('../../../data/libsvm/binary/ckn_mnist.npz'); y=data['y']; X = data['X']

def run_pegasos(Xtr, Ytr, Xts, Yts, nb_iterations, lambda_reg):
    model = PegasosSVMClassifier(iterations = int(5e6), lambda_reg=lambda_reg)
    ts = time.perf_counter()
    model.fit(Xtr, Ytr)
    te = time.perf_counter()
    ts_score = f1_score(Yts, model.predict(Xts), average = "binary")
    tr_score = f1_score(Ytr, model.predict(Xtr), average = "binary")
    t_fit, tr_score, ts_score = te - ts, tr_score, ts_score
    del model
    return t_fit, tr_score, ts_score


def run_sklearn(Xtr, Ytr, Xts, Yts, gamma, lambda_reg):
    C = 1 / (2 * Xtr.shape[0] * lambda_reg)
    model = svm.SVC(C=C, gamma=gamma)
    ts = time.perf_counter()
    model.fit(Xtr, Ytr)
    te = time.perf_counter()
    ts_score = f1_score(Yts, model.predict(Xts), average = "binary")
    tr_score = f1_score(Ytr, model.predict(Xtr), average = "binary")
    t_fit, tr_score, ts_score = te - ts, tr_score, ts_score
    return t_fit, tr_score, ts_score

def run_libsvm(Xtr, Ytr, Xts, Yts, lambda_reg):
    C = 1 / (2 * Xtr.shape[0] * lambda_reg)
    model = svm.LinearSVC(C = C, loss = "hinge")
    ts = time.perf_counter()
    model.fit(Xtr, Ytr)
    te = time.perf_counter()
    ts_score = f1_score(Yts, model.predict(Xts), average = "binary")
    tr_score = f1_score(Ytr, model.predict(Xtr), average = "binary")
    t_fit, tr_score, ts_score = te - ts, tr_score, ts_score
    return t_fit, tr_score, ts_score
    

def run_snacks(Xtr, Ytr, Xts, Yts, penalty, nit):
    if nit is not None:
        model = Snacks(penalty, n_iter = nit)
    else:
        model = Snacks(penalty)
    ts = time.perf_counter()
    model.fit(Xtr, Ytr)
    te = time.perf_counter()
    ts_score = f1_score(Yts, model.predict(Xts), average = "binary", zero_division = 1)
    tr_score = f1_score(Ytr, model.predict(Xtr), average = "binary", zero_division = 1)
    t_fit, tr_score, ts_score = te - ts, tr_score, ts_score
    del model
    return t_fit, tr_score, ts_score


def run_thundersvm(Xtr, Ytr, Xts, Yts, g, lambda_reg):
    C = 1 / (2 * Xtr.shape[0] * lambda_reg)
    tsvm = SVC(kernel = "rbf", gamma = g, C=C)
    ts = time.time()
    tsvm.fit(Xtr, Ytr)
    te = time.time()
    ts_score = f1_score(Yts, tsvm.predict(Xts), average = "binary")
    tr_score = f1_score(Ytr, tsvm.predict(Xtr), average = "binary")
    t_fit, tr_score, ts_score = te - ts, tr_score, ts_score
    return t_fit, tr_score, ts_score

def table_print(method, solution, tr_scores, ts_scores, times):
    if method == "Snacks":
        idx = 3
    elif method == "Pegasos":
        idx = 1
    elif method == "LibSVM":
        idx = 0
    elif method == "ThunderSVM":
        idx = 2
    else:
        assert False, "Unknown method"
    solution[idx][1] = f"{np.round(np.mean(np.array(tr_scores)), 4)} ± {np.round(np.std(np.array(tr_scores)), 4)}"
    solution[idx][2] = f"{np.round(np.mean(np.array(ts_scores)), 4)} ± {np.round(np.std(np.array(ts_scores)), 4)}"
    solution[idx][3] = f"{np.round(np.mean(np.array(times)), 4)} ± {np.round(np.std(np.array(times)), 4)}"
    print(tabulate(solution, headers=[f"Method / {dataset}", "Accuracy on Train", "Accuracy on Test", "Time"], tablefmt="github"))
    return solution

def compare(dataset, nb_runs, flag_tsvm):
    """Compares"""
    num_centers, gamma, penalty, num_it_pegasos = BEST_VALUES[dataset]
    
    solution = [
        ["LibSVM - on subset", None, None, None],
        ["Pegasos - on subset", None, None, None],
        ["ThunderSVM - on subset", None, None, None],
        ["Snacks - on subset", None, None, None],
    ]

    # oX, oY = utils.dataloader(dataset)
    oX, oY = X, y
    print(X.shape)
    oY[oY != 8] = -1
    oY[oY == 8] = +1
    oXtr, oXts, oYtr, oYts = train_test_split(oX, oY, train_size = 0.8)
    # oXtr, oYtr, oXts, oYts = utils.kernel_embedding(oX, oY, num_centers, 0.8, 'rbf', tsvm = True)
    print(f"Data is being embedded")
    time_start = time.perf_counter()
    # Xtr, Ytr, Xts, Yts = utils.kernel_embedding(oX, oY, num_centers, 0.8, "rbf", gamma = gamma)
    Xtr, Ytr, Xts, Yts = oXtr, oYtr, oXts, oYts
    time_end = time.perf_counter()
    print(f"Data embedded in {(time_end - time_start):.3f}s")
    run_snacks(Xtr, Ytr, Xts, Yts, penalty, 1) # for compilation purposes
    # SNACKS
    tr_scores, ts_scores, times = [], [], []
    for i_run in range(nb_runs):
        print(f"Snacks : run {i_run + 1}/{nb_runs}")
        t_fit, tr_score, ts_score = run_snacks(Xtr, Ytr, Xts, Yts, penalty, None)
        tr_scores.append(tr_score)
        ts_scores.append(ts_score)
        times.append(t_fit)

    solution = table_print("Snacks", solution, tr_scores, ts_scores, times)
    
    # Pegasos
    tr_scores, ts_scores, times = [], [], []
    for i_run in range(nb_runs):
        cXtr, cYtr, cXts, cYts = np.copy(Xtr), np.copy(Ytr), np.copy(Xts), np.copy(Yts) 
        #print(f"Pegasos : run {i_run + 1}/{1}")
        t_fit, tr_score, ts_score = run_pegasos(
            cXtr, cYtr, cXts, cYts, num_it_pegasos, penalty
        )
        del cXtr, cYtr, cXts, cYts
        tr_scores.append(tr_score)
        ts_scores.append(ts_score)
        times.append(t_fit)
    print(ts_scores)
    print(times)
    solution = table_print("Pegasos", solution, tr_scores, ts_scores, times)
    
    # LibSVM 2
    tr_scores, ts_scores, times = [], [], []
    for i_run in range(nb_runs):
        print(f"LibSVM : run {i_run + 1}/{nb_runs}")
        t_fit, tr_score, ts_score = run_libsvm(Xtr, Ytr, Xts, Yts, penalty)
        tr_scores.append(tr_score)
        ts_scores.append(ts_score)
        times.append(t_fit)
    
    solution = table_print("LibSVM", solution, tr_scores, ts_scores, times)
    # ThunderSVM
    tr_scores, ts_scores, times = [], [], []
    for i_run in range(nb_runs):
        print(f"ThunderSVM : run {i_run + 1}/{nb_runs}")
        t_fit, tr_score, ts_score = run_thundersvm(oXtr, oYtr, oXts, oYts, gamma, penalty)
        tr_scores.append(tr_score)
        ts_scores.append(ts_score)
        times.append(t_fit)

    solution = table_print("ThunderSVM", solution, tr_scores, ts_scores, times)
    print(f"Kernel matrix computed in {(time_end - time_start):.3f}")
    

if __name__ == "__main__":
    dataset = str(sys.argv[1])
    n_runs = int(sys.argv[2])
    compare(dataset, n_runs, True)
