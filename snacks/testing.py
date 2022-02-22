# %% [markdown]
# # SVM Comparison
# This notebook runs a comparison between ThunderSVM, LibSVM, Pegasos, ~~liquidSVM~~ and Snacks on 3 binary classification datasets:
#  - a9a
#  - SUSY
#  - HIGGS

# %%

# %%
import utils
import time
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append("../")

# %%
# Import all SVM solvers, data embedding functions and time method
from svm import Snacks
from pegasos import PegasosSVMClassifier
from sklearn import svm
from thundersvm import SVC

# %% [markdown]
# #### Utility functions for benchmarking

# %%
def prepare_data(data, num_centers, gamma, full):
    if data == "a9a":
        Xtr, Xts, Ytr, Yts = utils.dataloader("../datasets/a9a", 0.7)
    elif data == "SUSY":
        Xtr, Xts, Ytr, Yts = utils.dataloader("../datasets/SUSY", 0.7)
    elif data == "HIGGS":
        Xtr, Xts, Ytr, Yts = utils.dataloader("../datasets/HIGGS", 0.7)
    else:
        print(
            f"You asked for dataset {data} while Snacks only support a9a, SUSY and HIGGS"
        )

    Xtr, Ytr, Xts, Yts = utils.kernel_embedding(
        Xtr, Ytr, Xts, Yts, num_centers, full, gamma=gamma
    )

    return Xtr, Ytr, Xts, Yts

# %%
def run_snacks(Xtr, Ytr, Xts, Yts, nb_iterations, lambda_reg, stepsize):
    print("Snacks' performance : ")
    model = Snacks(
        nb_iterations=nb_iterations, lambda_reg=lambda_reg, stepsize=stepsize
    )
    ts = time.time()
    model.fit(Xtr, Ytr)
    te = time.time()
    score = model.score(Xts, Yts)
    print(f"in {(te - ts):.2f}s, C-err is {100 - score * 100:.2f}%")
    tr_score = model.score(Xtr, Ytr)
    print(f"also, train error is {100 - tr_score * 100:.2f}%")
    t_fit, score = te - ts, 1 - score
    return t_fit, score


def run_sklearn(Xtr, Ytr, Xts, Yts, lambda_reg):
    print("SKLearn's performance : ")
    C = 1 / (2 * Xtr.shape[0] * lambda_reg)
    model = svm.LinearSVC(loss="hinge", C=C)
    ts = time.time()
    model.fit(Xtr, Ytr)
    te = time.time()
    score = model.score(Xts, Yts)
    print(f"in {(te - ts):.2f}s, C-err is {100 - score * 100:.2f}%")
    t_fit, score = te - ts, 1 - score
    return t_fit, score

def run_thundersvm(Xtr, Ytr, Xts, Yts, lambda_reg):
    print("ThunderSVM's performance : ")
    C = 1 / (2 * Xtr.shape[0] * lambda_reg)
    model = SVC(C=C)
    ts = time.time()
    model.fit(Xtr, Ytr)
    te = time.time()
    score = model.score(Xts, Yts)
    print(f"in {(te - ts):.2f}s, C-err is {100 - score * 100:.2f}%")
    t_fit, score = te - ts, 1 - score
    return t_fit, score


def run_pegasos(Xtr, Ytr, Xts, Yts, nb_iterations, lambda_reg):
    print("Pegasos' performance : ")
    C = 1 / (2 * Xtr.shape[0] * lambda_reg)
    model = PegasosSVMClassifier(iterations=nb_iterations, lambda_reg=lambda_reg)
    ts = time.time()
    model.fit(Xtr, Ytr)
    te = time.time()
    score = model.score(Xts, Yts)
    print(f"in {(te - ts):.2f}s, C-err is {100 - score * 100:.2f}%")
    tr_score = model.score(Xtr, Ytr)
    print(f"also, train error is {100 - tr_score * 100:.2f}%")
    t_fit, score = te - ts, 1 - score
    return t_fit, score

# %% [markdown]
# #### Benchmarking

# %% [markdown]
# ##### a9a

# %%
Xtr, Ytr, Xts, Yts = prepare_data("a9a", 1400, 1e-1, False)
run_snacks(Xtr, Ytr, Xts, Yts, 45000, 1e-5, 1.0)
run_sklearn(Xtr, Ytr, Xts, Yts, 1e-5)
run_thundersvm(Xtr, Ytr, Xts, Yts, 1e-5)
run_pegasos(Xtr, Ytr, Xts, Yts, 45000 * 3, 1e-5)

# %% [markdown]
# Snacks' performance : 
# in 0.43s, C-err is 15.33%
# SKLearn's performance : ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
# in 8.04s, C-err is 15.25%
# Pegasos' performance : 
# in 1.63s, C-err is 16.72%

# %% [markdown]
# ##### SUSY

# %%
# Xtr, Ytr, Xts, Yts = prepare_data("SUSY", 1050, (1 / (2 * 4 * 4)), True)

# %%
# run_snacks(Xtr, Ytr, Xts, Yts, 35000, 3e-6, 0.5)
# run_sklearn(Xtr, Ytr, Xts, Yts, 1e-5)
# run_pegasos(Xtr, Ytr, Xts, Yts, 8000000 * 3, 3e-6)

# %% [markdown]
# 45s to load SUSY dataset (5e6 X 18)
# SKLearn's performance : 
# in 45.40s, C-err is 21.21%


