import os
import numpy as np
from aeon.datasets import load_from_tsfile

def znorm(x):
	zx = np.copy(x)

	# 1d array
	if len(x.shape) == 1:
		zx -= np.mean(zx)
		zx /= np.std(zx)
	else:
		zx -= np.mean(zx, axis=-1)[:,None]
		zx /= np.std(zx, axis=-1)[:,None]
	return zx

def align_dimensions(x,y):
    """Align two arrays on all dimensions"""
    assert len(x.shape) == len(y.shape)
    outshape = [min(x.shape[i], y.shape[i]) for i in range(len(x.shape))]

    # Align two vectors
    if len(x.shape) == 1:
        return x[:outshape[0]], y[:outshape[0]]
    # Align two matrices
    elif len(x.shape) == 2:
        return x[:outshape[0], :outshape[1]], y[:outshape[0], :outshape[1]]
    else:
        raise ValueError("Aligning arrays with more than two dimensions are not supported")
    
def align_first_dimensions(A,B):
    assert len(A.shape) == len(B.shape) == 2 # Only matrices

    # Align the first dimensions
    d = min(A.shape[0], B.shape[0])
    return A[:d], B[:d]
    
def aligned_norm(x,y,**kwargs):
    # Align the two time series on all dimensions
    x,y = align_dimensions(x,y)
    return np.linalg.norm(x-y, **kwargs)

def get_data(name:str, DATAPATH:str="data"):
    X_train,y_train = load_from_tsfile(os.path.join(DATAPATH,name,name + "_TRAIN.ts"))
    X_test,y_test = load_from_tsfile(os.path.join(DATAPATH,name,name + "_TEST.ts"))

    # Concatenate train and test data
    if type(X_train) == list:
        X = X_train + X_test
    else:
        X = np.concatenate((X_train,X_test))

    y = np.concatenate((y_train,y_test))
    return X,y

def get_batch_size(X, max_size=8*1024**3):
    # Compute the maximum batch size
    row_size = X.size
    batch_size = max_size // row_size
    return batch_size

def match_length(x,y):
    # Match the length of the two arrays
    if x.shape[-1] > y.shape[-1]:
        x = x[..., :y.shape[-1]]
    elif x.shape[-1] < y.shape[-1]:
        y = y[..., :x.shape[-1]]
    return x,y