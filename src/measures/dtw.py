import numpy as np
from tslearn.metrics import dtw as dtw_tslearn
from .inter import distmat
from .utils import align_first_dimensions

dtw = lambda x,y: dtw_tslearn(x,y, global_constraint='sakoe_chiba', sakoe_chiba_radius=10)

def dtw_d(A,B):
    # Align the dimensions
    A,B = align_first_dimensions(A,B)
    return dtw(A.T,B.T)

def dtw_i(A,B):
    # Align the dimensions
    A,B = align_first_dimensions(A,B)
    return np.sum([dtw(A[i], B[i]) for i in range(A.shape[0])])

def dtw_d_all(X):
    return distmat(X, dtw_d)

def dtw_i_all(X):
    return distmat(X, dtw_i)