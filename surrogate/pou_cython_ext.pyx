import numpy as np

from scipy.spatial.distance import cdist
from sutils import innerMatrixProduct


def POUEval(X_cont, xc, f, g, h, delta, rho):

    neval = X_cont.shape[0]
    nsamples = xc.shape[0]

    y_ = np.zeros(neval)
    for k in range(neval):
        x = X_cont[k,:] 

        # exhaustive search for closest sample point, for regularization
        D = cdist(np.array([x]),xc)
        mindist = min(D[0]) 
        numer = 0
        denom = 0   

        # evaluate the surrogate, requiring the distance from every point
        work = x - xc
        dist = D[0][:] + delta#np.sqrt(D[0][i] + delta)
        expfac = np.exp(-rho*(dist-mindist))
        local = np.zeros(nsamples)
        for i in range(nsamples):
            local[i] = f[i] + higher_terms(work[i], g[i], h[i])
        numer = np.dot(local, expfac)
        denom = np.sum(expfac)

        y_[k] = numer/denom
    
    return y_

def higher_terms(dx, g, h):
    terms = np.dot(g, dx)
    terms += 0.5*np.dot(np.dot(dx.T, h), dx)#innerMatrixProduct(h, dx)
    return terms