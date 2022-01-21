from email.utils import collapse_rfc2231_value
import numpy as np
from pca import PCA
from gradcluster import findClusters
import scipy.cluster.hierarchy as hc
from matplotlib import pyplot as plt
from smt.sampling_methods import LHS
from smt.problems import Branin
from ellipse import Ellipse
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

    # perform PCA for each cluster gradient
    cpcas  = []
    cpcaev = []
    cpcae  = []
    for k in range(nclust):
        s, eve, eva = PCA(cdatag[k])
        cpcas.append(s)
        cpcaev.append(eve)
        cpcae.append(eva)

    return cdatax, cdataf, cdatag, cpcas, cpcaev, cpcae

dim = 2

trueFunc = Ellipse(foci = [3,3])
xlimits = trueFunc.xlimits
sampling = LHS(xlimits=xlimits)

ntr = 200
nte = 10

tr = sampling(ntr)
gr = np.zeros([ntr,dim])

fr = trueFunc(tr)
for i in range(dim):
    gr[:,i:i+1] = trueFunc(tr,i)

J = 4
cx, cf, cg, cps, cpv, cpe = clusteredPCA(tr, fr, gr, J)


#PLOTTING
m = []
for i in range(J):
    m.append(np.mean(cx[i],axis=0))

#import pdb; pdb.set_trace()
for i in range(J):
    plt.plot(cx[i][:,0], cx[i][:,1], 'o')
    plt.arrow(m[i][0], m[i][1], -cpv[i][0,0]*2, -cpv[i][0,1]*2, head_width=0.28, head_length=0.004)
    #plt.savefig('branin.png')
    #import pdb; pdb.set_trace()
    #plt.plot([0, cpv[i][1,0]*2]+m[i][0], [0, cpv[i][1,1]*2]+m[i][1],'--k')

# Contour
ndir = 100
x = np.linspace(xlimits[0][0], xlimits[0][1], ndir)
y = np.linspace(xlimits[1][0], xlimits[1][1], ndir)

X, Y = np.meshgrid(x, y)
Z = np.zeros([ndir, ndir])

for i in range(ndir):
    for j in range(ndir):
        xi = np.zeros([1,2])
        xi[0,0] = x[i]
        xi[0,1] = y[j]
        Z[i,j] = trueFunc(xi)

plt.savefig('clusters.png')
plt.contour(X, Y, Z, levels = 15)
plt.savefig('ellipseclust.png')


