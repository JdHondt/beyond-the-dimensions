from typing import List, Tuple, Dict, Any, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd
import sys
from sklearn.metrics import accuracy_score
import signal
import os

from measures.utils import znorm

def _timeout_handler(signum, frame):
    raise TimeoutError("Timed out!")

@dataclass
class Model:
    name: str
    distance_func: callable
    kwargs: dict = None
    normalize: bool = False

    def copy(self):
        return Model(self.name, self.distance_func, self.kwargs, self.normalize)

    @staticmethod
    def prepare_dists(dists: np.ndarray):
        """Preprocess the distances"""
        # Ignore the diagonal
        nanmax = np.nanmax(dists)
        np.fill_diagonal(dists, nanmax)

        # Fill nans with max value
        dists = np.nan_to_num(dists, nan=nanmax)

        # Fill negatives with max value
        dists[dists < 0] = nanmax

        return dists

    def knn(self, X:np.ndarray, y:np.ndarray, k=1):
        """
        Peform k-nearest neighbor classification on the given dataset.

        returns:
        y_pred: The predicted labels
        y_prob: The probabilities of the predicted labels
        """

        dists = self.distance_func(X, self.kwargs) if self.kwargs else self.distance_func(X)
        dists = self.prepare_dists(dists)
        knns = y[np.argsort(dists, axis=1)[:, :k]]

        ypreds = []
        yprobs = []
        for knni in knns:
            labels, counts = np.unique(knni, return_counts=True)
            probs = counts / k
            maxprob_i = np.argmax(probs)

            # Get y_pred with probability
            ypreds.append(labels[maxprob_i])
            yprobs.append(probs[maxprob_i])

        return np.array(ypreds), np.array(yprobs)

@dataclass
class Ensemble:
    models: List[Model]

    def __post_init__(self):
        self.name = " + ".join([m.name for m in self.models])

    def copy(self):
        return Ensemble([m.copy() for m in self.models])

    def _ensemble_weights(self, k:int, dname:str) -> np.ndarray:
        """
        Get the accuracy-based weights for an ensemble of classifiers.

        returns:
        Array of weights, ordered by the order of measures passed
        """

        filename = f"output/knn_experiment_k{k}_server.csv"

        if not os.path.exists(filename):
            print(f"File {filename} does not exist")
            exit(1)

        accdf = pd.read_csv(filename, names=["Measure", "Dataset", "Normalized", "Accuracy", "Runtime"])

        model_names = [m.name for m in self.models]

        # Get the accuracies for this dataset and these measures
        accdf = accdf[(accdf.Dataset == dname) & (accdf.Measure.isin(model_names))].dropna()
        if len(accdf) == 0:
            print(f"No known accuracies for methods in ensemble for this dataset {dname}, using equal weights")
            n = len(self.models)
            return np.ones(n) / n
        
        # Get the mean accuracies for this dataset (ignoring the measures for which we don't have an accuracy)
        accuracies = accdf.groupby("Measure").Accuracy.mean()
        w = np.zeros(len(self.models))
        for i,m in enumerate(self.models):
            if m.name in accuracies:
                w[i] = accuracies[m.name]

        w /= np.sum(w)
        return w

    def ensemble_knn(self, X: np.ndarray,y: np.ndarray,dname:str, k=1, Xz: np.ndarray = None, timeout = 1800) -> float:
        """
        Perform hold-one-out classification test on the given dataset using the ensemble method.

        Parameters:
        X: The dataset to test on
        y: The labels of the dataset
        dname: The name of the dataset
        k: The number of neighbors to use
        Xz: The normalized dataset
        timeout: The maximum time to run for
        maxn: The maximum number of samples to run on

        returns:
        The accuracy of the ensemble method
        """

        if len(self.models) == 0:
            raise BaseException("No models in ensemble")

        # Normalize data  
        if Xz is None:
            Xz = [znorm(x) for x in X]

        # Ensemble method
        w = self._ensemble_weights(k, dname) if len(self.models) > 1 else np.ones(1)

        # Set timeout
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout)

        try:
            # Get weighted class probabilities and y_pred for each measure
            y_preds = []
            y_probs = []
            for i, model in enumerate(self.models):
                if w[i] == 0:
                    y_preds.append(np.zeros(len(y)))
                    y_probs.append(np.zeros(len(y)))
                    continue

                data = Xz if model.normalize else X
                loc_ypreds, loc_yprobs = model.knn(data, y, k)
                y_preds.append(loc_ypreds)
                y_probs.append(loc_yprobs)

            # Get the ensemble y_pred
            y_preds = np.array(y_preds) # (n_measures, n_samples)
            y_probs = np.array(y_probs) # (n_measures, n_samples)
            y_probs *= w[:, np.newaxis] # (n_measures, n_samples) * (n_measures, 1)

            y_preds = y_preds[np.argmax(y_probs, axis=0), np.arange(len(y))] # (n_samples)

            # Get accuracy
            return accuracy_score(y, y_preds)   
        except TimeoutError:
            print(f"Timed out after {timeout} seconds")
            return np.nan
        finally:
            signal.alarm(0)