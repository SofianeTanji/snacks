import utils
import time
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append("../")

# Import all SVM solvers, data embedding functions and time method
from svm import Snacks
from pegasos import PegasosSVMClassifier
from sklearn import svm


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
