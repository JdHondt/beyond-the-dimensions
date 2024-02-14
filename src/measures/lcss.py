import numpy as np
from tslearn.metrics import lcss as lcss_tslearn
from .inter import distmat
from .utils import align_first_dimensions

def lcss(A,B):
    assert len(A.shape) == len(B.shape)

    m = min(A.shape[-1], B.shape[-1])

    # Epsilon = sigma A / 2
    eps = np.std(A) / 2
    delta = m / 20

    return 1 - lcss_tslearn(A.T,B.T, eps=eps, sakoe_chiba_radius=delta)

def lcss_d(A,B):
    # Align the first dimensions
    A,B = align_first_dimensions(A,B)
    return lcss(A.T,B.T)

def lcss_i(A,B):
    # Align the first dimensions
    A,B = align_first_dimensions(A,B)
    return np.sum([lcss(A[i], B[i]) for i in range(A.shape[0])])

def lcss_d_all(X):
    return distmat(X, lcss_d)

def lcss_i_all(X):
    return distmat(X, lcss_i)