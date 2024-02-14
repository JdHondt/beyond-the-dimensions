import numpy as np
import scipy.linalg as la
from scipy.linalg import LinAlgError
from .utils import align_dimensions

def _kl_mvn(meanX:np.ndarray, covX:np.ndarray, meanY:np.ndarray,covY:np.ndarray):
    """Calculate `KL(to||fr)`, where `to` and `fr` are pairs of means and covariance matrices"""

    # Map the means and covariances to the same subspace
    meanX, meanY = align_dimensions(meanX, meanY)
    covX, covY = align_dimensions(covX, covY)

    m_to, S_to = meanX, covX
    m_fr, S_fr = meanY, covY
    
    d = m_fr - m_to
    
    c, lower = la.cho_factor(S_fr)
    def solve(B):
        return la.cho_solve((c, lower), B)
    
    def logdet(S):
        return np.linalg.slogdet(S)[1]

    term1 = np.trace(solve(S_to))
    term2 = logdet(S_fr) - logdet(S_to)
    term3 = d.T @ solve(d)
    return (term1 + term2 + term3 - len(d))/2.
    
def _kl_mvn_sym(meanX:np.ndarray, covX:np.ndarray, meanY:np.ndarray,covY:np.ndarray):
    return (_kl_mvn(meanX, covX, meanY,covY) + _kl_mvn(meanY,covY,meanX,covX))/2.

def kl_all(X):
    means = [np.mean(x, axis=-1) for x in X]
    covs = [np.cov(x) for x in X]

    n = len(means)
    kls = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1, n):
            try:
                val = _kl_mvn_sym(means[i], covs[i], means[j], covs[j])
            except LinAlgError:
                val = np.nan
            
            kls[i,j] = val
            kls[j,i] = kls[i,j]

    return kls