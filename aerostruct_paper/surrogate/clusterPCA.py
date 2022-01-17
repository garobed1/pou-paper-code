import numpy as np
from pca import PCA
from gradcluster import findClusters
import scipy.cluster.hierarchy as hc
from matplotlib import pyplot as plt
from smt.sampling_methods import LHS
from smt.problems import Branin
from smt.surrogate_models import gekpls

def clusteredPCA(x, f, g, J, rho = 0.9):
    """
    Determines a set of J data clusters for a set of initial data based on 
    the gradient, and performs PCA on each cluster

    Parameters
    ----------
    x : array
        Initial data locations
    f : array
        Initial data function values
    g : array
        Initial data gradients
    J : int
        Number of clusters to use
    rho : float
        PCA dimension reduction ratio

    """
    n, m = x.shape

    Z = findClusters(x, g, method='average', metric='cosine')
    clusterid = hc.fcluster(Z, J, criterion='maxclust')

    # split the data into clusters
    nclust = clusterid.max()
    cdatax = []
    cdataf = []
    cdatag = []
    clusterid = clusterid - 1
    j = np.bincount(clusterid)
    for k in range(nclust):
        cdatax.append(np.zeros([j[k],m]))
        cdataf.append(np.zeros([j[k],m]))
        cdatag.append(np.zeros([j[k],m]))
    
    c = np.zeros(nclust, dtype=int)
    for i in range(n):
        cid = clusterid[i]
        cdatax[cid][c[cid]] = x[i]
        cdataf[cid][c[cid]] = f[i]
        cdatag[cid][c[cid]] = g[i]
        c[cid] += 1

    return cdatax, cdataf, cdatag

dim = 2

trueFunc = Branin(ndim=dim)
xlimits = trueFunc.xlimits
sampling = LHS(xlimits=xlimits)

ntr = 100
nte = 10

tr = sampling(ntr)
gr = np.zeros([ntr,dim])

fr = trueFunc(tr)
for i in range(dim):
    gr[:,i:i+1] = trueFunc(tr,i)

cx, cf, cg = clusteredPCA(tr, fr, gr, 4)

plt.plot(cx[0][:,0], cx[0][:,1], 'o')
plt.plot(cx[1][:,0], cx[1][:,1], 'o')
plt.plot(cx[2][:,0], cx[2][:,1], 'o')
plt.plot(cx[3][:,0], cx[3][:,1], 'o')
plt.savefig('clusters.png')