# Author : Sofiane Tanji
# License : GNU GPL V3

# Basics
import time
import sys
sys.path.append("../")
sys.path.append("../../")

# Method
from svm import Snacks

# Utils
from constants import BEST_VALUES, N_SAMPLES
import utils
from psutil import virtual_memory
import numpy as np

available_ram = virtual_memory()[1] * 0.9

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

def run(dataset):
    num_centers, gamma, n_iter, eta, D0, K, penalty, _ = BEST_VALUES[dataset]
    n_samples = N_SAMPLES[dataset]

    necessary_ram = n_samples * num_centers * 4 * 1.2
    print(available_ram)
    if available_ram < necessary_ram:
        raise ValueError("Not enough RAM")

    _, _, oXtr, oXts, oYtr, oYts = utils.dataloader(dataset, 0.8)
    print(f"Data is being embedded")
    ts = time.perf_counter()
    Xtr, Ytr, Xts, Yts = utils.kernel_embedding(
        oXtr, oYtr, oXts, oYts, num_centers, gamma = gamma
    )
    te = time.perf_counter()
    print(f"Data has been embedded in {(te - ts):.2f} seconds.")
    run_snacks(Xtr[:20], Ytr[:20], Xts[:20], Yts[:20], 1, eta, D0, 1, penalty) # for compilation purposes
    t_fit, tr_score, ts_score = run_snacks(Xtr, Ytr, Xts, Yts, n_iter, eta, D0, K, penalty)
    print(f"Training took {t_fit:.1f} seconds")
    print(f"Training error is {tr_score:.2f} and Test error is {ts_score:.2f}")

if __name__ == "__main__":
    dataset = str(sys.argv[1])
    run(dataset)