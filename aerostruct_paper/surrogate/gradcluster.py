import numpy as np
from scipy.cluster import hierarchy as hc
from scipy.special import binom

'''
Gradient-based data clustering, based on Xiong et al. 2021
'''

def findClusters(x, g, eta = 1, method='average', metric='combined'):
    """
    Computes agglomerated clusters of initial data based on the gradients at the
    points, or a combined distance function based on the work of 
    Xiong et al. 2021

    Parameters
    ----------
    x : array
        Initial data locations
    g : array
        Initial data gradients
    eta : float
        Parameter balancing influence between physical distance and cosine
            gradient distance for combined distance
    method : string
        Linkage criterion
    metric : 
        Distance function, with option for the combined gradient/euclidean
            distance

    Returns
    -------
    Z : array
        Hierarchichal clustering tree based on the gradients
    """

    n, m = x.shape

    # precompute distances based on cosDistCorrected
    if(metric == 'combined'):
        y = np.zeros(n*(n-1)/2)
        return
    else:
        Z = hc.linkage(g, method=method, metric=metric)

    return Z


# need to find a way to use this
def cosDistCorrected(x1, x2, g1, g2, eta):
    """
    Compute a "safe" cosine distance function between gradients that uses
    Euclidean distance as a correction in extreme cases

    Parameters
    ----------
    x1, x2 : array
        Locations of points to be compared
    g1, g2 : array
        Gradients at points to be compared
    eta : float
        Parameter balancing influence between physical distance and cosine
            gradient distance

    Returns
    -------
    dist : float
        Balanced physical/gradient cosine distance measure
    """
    dim = len(x1)
    graddist = 1 - abs(np.cos(np.dot(g1,g2)))
    xdist = np.linalg.norm(x2 - x1)/np.sqrt(dim)

    dist = eta*graddist + (1-eta)*xdist

    return dist


