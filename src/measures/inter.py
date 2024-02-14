import numpy as np
from .utils import aligned_norm

def distmat(X, dist_func: callable):
    n = len(X)
    dists = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            dists[i,j] = dist_func(X[i],X[j])
            dists[j,i] = dists[i,j]
    return dists

def inter_eucl_all(X):
    return distmat(X, lambda x,y: aligned_norm(x,y))

def inter_eucl_sum_all(X):
    return distmat(X, lambda x,y: np.sum(aligned_norm(x,y, axis=-1)))