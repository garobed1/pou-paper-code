import numpy as np
from numpy import linalg
#import matplotlib.pyplot as plt

def PCA(data):
    m, n = data.shape

    # mean-centered data
    mcd = data - data.mean(axis=0)

    # covariance matrix and eigenvalues
    cov = np.cov(data, rowvar=False)
    eigvals, eigvecs = linalg.eigh(cov)

    ind = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:,ind]
    eigvals = eigvals[ind]
    return np.dot(eigvecs, mcd.T).T, eigvecs, eigvals


# A = np.array([ [2.4,0.7,2.9,2.2,3.0,2.7,1.6,1.1,1.6,0.9],
#             [2.5,0.5,2.2,1.9,3.1,2.3,2,1,1.5,1.1] ])

# score, evecs, evals = PCA(A.T)
# import pdb; pdb.set_trace()

# m = np.mean(A,axis=1)
# plt.plot([0, -evecs[0,0]*2]+m[0], [0, -evecs[0,1]*2]+m[1],'--k')
# plt.plot([0, evecs[1,0]*2]+m[0], [0, evecs[1,1]*2]+m[1],'--k')
# plt.plot(A[0,:],A[1,:],'ob') # the data
# plt.axis('equal')
# plt.savefig('pcatest.png')