import pandas as pd
import numpy as np
import os
from typing import List, Tuple, Dict, Any, Union
import sys

import time
from measures.utils import get_data, znorm
from measures.kl import kl_all
from measures.pca import pca_all, eros_all
from measures.dtw import dtw_d_all, dtw_i_all
from measures.intra import intra_eucl_all, intra_cov_all, intra_manh_all, intra_dtw_all
from measures.inter import inter_eucl_all, inter_eucl_sum_all
from measures.lcss import lcss_d_all, lcss_i_all
from model import Ensemble, Model
import itertools

from multiprocessing import Pool

K = 5
MAXN = 2000
TIMEOUT = 3600 * 4
OUTPATH = f"output/knn_experiment_k{K}_server.csv"

df_cols = ["Measure", "Dataset", "Normalized", "Accuracy", "Runtime"]

MODELS = {
        "pca": Ensemble([Model(r"$D_{PCA}$", pca_all)]),
        "eros": Ensemble([Model(r"$D_{Eros}$", eros_all)]),
        "kl": Ensemble([Model(r"$D_{KL}$", kl_all)]),
        "intra-l2": Ensemble([Model(r"Intra-$L_2$", intra_eucl_all)]),
        "intra-l1": Ensemble([Model(r"Intra-$L_1$", intra_manh_all)]),
        "intra-dtw": Ensemble([Model(r"Intra-DTW", intra_dtw_all)]),
        "intra-cov": Ensemble([Model(r"Intra-Cov", intra_cov_all)]),
        "inter-l2": Ensemble([Model(r"Inter-$L_2$", inter_eucl_all)]),
        "inter-dtwd": Ensemble([Model(r"Inter-DTW-D", dtw_d_all)]),
        "inter-dtwi": Ensemble([Model(r"Inter-DTW-I", dtw_i_all)]),
        "inter-lcssd": Ensemble([Model(r"Inter-LCSS-D", lcss_d_all)]),
        "inter-lcssi": Ensemble([Model(r"Inter-LCSS-I", lcss_i_all)]),
        "intra-cov+inter-l2-norm": Ensemble([
            Model(r"Intra-Cov", intra_cov_all),
            Model(r"Inter-$L_2$ norm", inter_eucl_all, {"normalize": True}),
        ]),
        "intra-cov+inter-dtwd": Ensemble([
            Model(r"Intra-Cov", intra_cov_all),
            Model(r"Inter-DTW-D", dtw_d_all),
        ]),
        "intra-dtw+inter-dtwd": Ensemble([
            Model(r"Intra-DTW", intra_dtw_all),
            Model(r"Inter-DTW-D", dtw_d_all),
        ]),
        "kl+inter-dtwd": Ensemble([
            Model(r"$D_{KL}$", kl_all),
            Model(r"Inter-DTW-D", dtw_d_all),
        ]),
        "kl+inter-lcssi": Ensemble([
            Model(r"$D_{KL}$", kl_all),
            Model(r"Inter-LCSS-i", lcss_i_all),
        ]),
}   

DATASETS = os.listdir("data")

def load_prepare_dataset(dname):
    X, y = get_data(dname)

    # Filter out datapoints with 0 variance
    X_idx = [i for i in range(len(X)) if np.all(np.std(X[i], axis=-1) > 1e-10)]
    X = [X[i] for i in X_idx]
    y = y[X_idx]

    if len(X) > MAXN:
        # Take a random subset of the data
        idx = np.random.choice(len(X), MAXN, replace=False)
        X = [X[i] for i in idx]
        y = y[idx]

    print(f"Loaded dataset {dname}, Shape: n={len(X)}, rest: {X[0].shape}, #Classes: {len(np.unique(y))}")

    # Pre-normalize the data
    Xz = [znorm(x) for x in X]
    return X, y, Xz

def knn_run(ensemble: Ensemble, X: np.ndarray, y: np.ndarray, dname:str, Xz: np.ndarray):
        print(f"Running {ensemble.name} on {dname}")
        start = time.time()
        acc = ensemble.ensemble_knn(X, y, dname, K, Xz, timeout=TIMEOUT)
        rt = time.time() - start
        norm = ensemble.models[0].normalize if len(ensemble.models) == 1 else False

        print(f"{ensemble.name}, {dname}, {norm}, \t Accuracy: {acc:.4f},\t Runtime: {rt:.2f}")

        # Append to file
        row = [ensemble.name, dname, norm, acc, rt]
        with open(OUTPATH, 'a') as f:
            f.write(",".join([str(x) for x in row]) + "\n")

def knn_run_wrapper(args):
    knn_run(*args)

# Get the accuracy of each of the distance measures for each dataset
def main(model: Ensemble, dname: str, norm:bool):
    # Get the dataset
    X, y, Xz = load_prepare_dataset(dname)

    # Make a copy of the model if we need to normalize
    if norm:
        model = model.copy()
        model.models[0].normalize = True

    knn_run(model, X, y, dname, Xz)


if __name__ == "__main__":
    print("Arguments:", sys.argv)

    # Expecting model name, dataset name, normalize as arguments
    if len(sys.argv) < 4:
        print("Expecting model name, dataset name, normalize as arguments")
        sys.exit(1)

    # Check if model exists
    model_name = sys.argv[1]
    if model_name not in MODELS:
        print("Model not found")
        sys.exit(1)

    # Check if dataset exists
    dataset_name = sys.argv[2]
    if dataset_name not in DATASETS:
        print("Dataset not found")
        sys.exit(1)

    # Check if normalize in arguments
    normalize = sys.argv[3]
    if normalize not in ["true", "false"]:
        print("Normalize should be true or false")
        sys.exit(1)

    # Print the arguments
    print(f"Running model {model_name} on dataset {dataset_name} with normalize={normalize}")

    # Run the main function
    model = MODELS[model_name]
    normalize = normalize == "true"
    main(model, dataset_name, normalize)