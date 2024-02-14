from sklearn.decomposition import PCA
import numpy as np
import math
from .inter import distmat
from .utils import match_length

def pca(A):
    return PCA().fit(A.T)

def get_weights(dataset):
    """
    Gets the weights for Eros; the average eigenvalues of the covariance matrices of the whole dataset.

    Parameters
    ----------
    dataset : List or array of shape=(n_mts, n_dimensions, n_samples)
        The dataset to compute the weights for.
    """
    n, d, m = len(dataset), max([mts.shape[0] for mts in dataset]), max([mts.shape[-1] for mts in dataset])
    n_components = min(d, m)

    weights = np.zeros((n,n_components)) # shape=(n_mts, n_dimensions)
    exp_variance_ratios = np.zeros((n,n_components)) # shape=(n_mts, n_dimensions)
    pcas = np.zeros((n,n_components,d))

    for i,mts in enumerate(dataset):
        pca_obj = pca(mts)

        s = pca_obj.singular_values_
        Vt = pca_obj.components_ 
        exvar = pca_obj.explained_variance_ratio_

        nc, nf = Vt.shape

        # Only fill the relevant components, leave the rest as 0
        weights[i, :len(s)] = s
        nc = Vt.shape[0]
        pcas[i, :nc, :nf] = Vt
        exp_variance_ratios[i, :len(exvar)] = exvar

    # Average the weights and variances over the MTS
    weights = np.mean(weights, axis=0)
    exp_variance_ratios = np.median(exp_variance_ratios, axis=0)

    # Normalize the weights
    weights /= np.sum(weights)

    return weights, pcas, exp_variance_ratios

def pca_sim(pca1, pca2):
    return np.sum(np.square(pca1 @ pca2.T))

def pca_dist(pca1, pca2):
    return 1 / (1 + pca_sim(pca1, pca2))

def pca_all(X):
    # Get pcas
    weights, pcas, exvar_ratios = get_weights(X)

    # Only keep the components that cover 90% of the variance
    n_components = np.argmax(np.cumsum(exvar_ratios) >= 0.90) + 1
    pcas = pcas[:, :n_components, :n_components]

    return distmat(pcas, pca_dist)

def eros(pca1, pca2, w):
    """
    Compute the extended frobenius norm between the principal components of two MTS.
    The weights are based on the average eigenvalues of the covariance matrices of the whole dataset.
    """
    # Compute the weighted sum of dots of the eigenvectors
    s = 0
    for i in range(pca1.shape[0]):
        s += w[i] * np.abs(np.dot(pca1[i], pca2[i]))

    return s

def eros_dist(pca1, pca2, w):
    sim = min(eros(pca1, pca2, w), 1) # Set max similarity to 1
    return math.sqrt(2 - 2*sim)

def eros_all(X, n_components=None):
    """
    Compute the extended frobenius norm between the principal components of all MTS.
    The weights are based on the average eigenvalues of the covariance matrices of the whole dataset.
    """
    # Get weights
    w, pcas, _ = get_weights(X)

    if n_components is not None:
        if n_components < pcas.shape[1]:
            # Truncate pcas
            pcas = pcas[:, :n_components, :]

            # Renormalize weights
            w = w[:n_components] / np.sum(w[:n_components])

    # Compute the distance matrix
    return distmat(pcas, lambda x,y: eros_dist(x,y,w))