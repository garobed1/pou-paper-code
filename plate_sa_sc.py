from pyDOE import lhs
from itertools import product
import numpy as np
import math

# Test to generate sample points in the space of Spalart-Allmaras constants,
# starting with kappa, cb1, cb2, and sigma. Assume uniform distributions as follows:

# following these approximate conditions

# (1+cb2)/sigma = [1.7, 2.7]
# Diffusion
# kappa^2 * (1+cb2)/(sigma/cb1) \approx 3
# s = Collocation Order
# kappa = []

# ff: do full factorial, equal weights
def genSC(s, ff = False):
    cb1 = [0.128, 0.137]
    cb2 = [0.6, 0.7]
    sigma = [0.6, 1.0]

    # find bounds of kappa
    kap = np.zeros(8)
    count = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                temp = (1+cb2[j])/sigma[k]/cb1[i]
                kap[count] = math.sqrt(3/temp)
                count = count + 1

    kappa = [min(kap), max(kap)]
    n = 4
    
    # get list of all combinations
    if ff == False:
        x, y = np.polynomial.legendre.leggauss(s)
        y = y/2
    else:
        x = np.linspace(0., 1., s)
        y = np.ones(s)*(1./s)
    
    total = len(x)**n
    ind = [*range(s)]
    comb = list(product(ind, repeat = n)) 

    #print(total)
    #print(len(comb))
    dist = np.zeros([total,n])#total
    weights = np.ones(total)#total
    add = 1.
    div = 2.
    if ff == True:
        add = 0.
        div = 1.
    for i in range(total):
        dist[i][0] = (x[comb[i][0]]+add)*(kappa[1]-kappa[0])/div + kappa[0]
        dist[i][1] = (x[comb[i][1]]+add)*(cb1[1]-cb1[0])/div + cb1[0]
        dist[i][2] = (x[comb[i][2]]+add)*(cb2[1]-cb2[0])/div + cb2[0]
        dist[i][3] = (x[comb[i][3]]+add)*(sigma[1]-sigma[0])/div + sigma[0]
        for j in range(n):
            weights[i] *= y[comb[i][j]]


    #import pdb; pdb.set_trace()
    return dist, weights

#print(genSC(2, ff = False))


#print(kap)
#print(dist)


