import numpy as np
from .inter import distmat
from .utils import aligned_norm
import sys
from .dtw import dtw

def intra_dists_all(X, intra_dist_f:callable):
    distmats = [distmat(x, intra_dist_f) for x in X]
    return distmat(distmats, lambda dx, dy: aligned_norm(dx,dy, ord='fro'))

def intra_cov_all(X):
    distmats = np.array([np.cov(x) for x in X])
    return distmat(distmats, lambda dx, dy: aligned_norm(dx,dy, ord='fro'))

def intra_eucl_all(X):
    return intra_dists_all(X, intra_dist_f = lambda x,y: aligned_norm(x,y))

def intra_manh_all(X):
    return intra_dists_all(X, intra_dist_f = lambda x,y: aligned_norm(x,y, ord=1))

def intra_dtw_all(X):
    return intra_dists_all(X, intra_dist_f = lambda x,y: dtw(x,y))