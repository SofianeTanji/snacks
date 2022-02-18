import utils
import time
import os
import sys
from profilehooks import profile
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append("../")

# Import all SVM solvers, data embedding functions and time method
from svm import Snacks
from pegasos import PegasosSVMClassifier
from thundersvm import *
from sklearn import svm

def prepare_data(data, num_centers, gamma):
    if data == "a9a":
        Xtr, Xts, Ytr, Yts = utils.dataloader('../datasets/a9a', 0.7)
    elif data == "SUSY":
        Xtr, Xts, Ytr, Yts = utils.dataloader('../datasets/SUSY', 0.7)
    elif data == "HIGGS":
        Xtr, Xts, Ytr, Yts = utils.dataloader('../datasets/HIGGS', 0.7)
    else:
        print(f"You asked for dataset {data} while Snacks only support a9a, SUSY and HIGGS")
    Xtr, Ytr, Xts, Yts = utils.kernel_embedding(Xtr, Ytr, Xts, Yts, False, num_centers, gamma = gamma)
    return Xtr, Ytr, Xts, Yts

def run_snacks(Xtr, Ytr, Xts, Yts, nb_iterations, lambda_reg, stepsize):    
    model = Snacks(nb_iterations = nb_iterations, lambda_reg = lambda_reg, stepsize = stepsize)
    ts = time.time()
    model.fit(Xtr, Ytr)
    te = time.time()
    score = model.score(Xts, Yts)
    print(f"in {(te - ts):.2f}s, C-err is {100 - score * 100:.2f}%")
    t_fit, score = te - ts, 1 - score
    return t_fit, score

def run_sklearn(Xtr, Ytr, Xts, Yts, lambda_reg):
    C = 1 / (2 * Xtr.shape[0] * lambda_reg)
    model = svm.LinearSVC(loss = "hinge", C = C)
    ts = time.time()
    model.fit(Xtr, Ytr)
    te = time.time()
    score = model.score(Xts, Yts)
    print(f"in {(te - ts):.2f}s, C-err is {100 - score * 100:.2f}%")
    t_fit, score = te - ts, 1 - score
    return t_fit, score

@profile(immediate = True)
def run_thundersvm(Xtr, Ytr, Xts, Yts, lambda_reg):
    C = 1 / (2 * Xtr.shape[0] * lambda_reg)
    model = SVC(kernel = "linear", C = C, max_iter=1000)
    ts = time.time()
    model.fit(Xtr, Ytr)
    te = time.time()
    score = model.score(Xts, Yts)
    print(f"in {(te - ts):.2f}s, C-err is {100 - score * 100:.2f}%")
    t_fit, score = te - ts, 1 - score
    return t_fit, score

def run_pegasos(Xtr, Ytr, Xts, Yts, nb_iterations, lambda_reg):
    C = 1 / (2 * Xtr.shape[0] * lambda_reg)
    model = PegasosSVMClassifier(iterations = nb_iterations, lambda_reg = lambda_reg)
    ts = time.time()
    model.fit(Xtr, Ytr)
    te = time.time()
    score = model.score(Xts, Yts)
    print(f"in {(te - ts):.2f}s, C-err is {100 - score * 100:.2f}%")
    t_fit, score = te - ts, 1 - score
    return t_fit, score

Xtr, Ytr, Xts, Yts = prepare_data("a9a", 10, 1e-1)

t_svm = SVC(kernel="linear", verbose = True)
t_svm.fit(Xtr, Ytr)

score = t_svm.score(Xts, Yts)

print(score)