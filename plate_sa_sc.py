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
def genSC(s):
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
    x, y = np.polynomial.legendre.leggauss(s)
    y = y/2
    total = len(x)**n
    ind = [*range(s)]
    comb = list(product(ind, repeat = n)) 

    print(total)
    print(len(comb))
    dist = np.zeros([total,n])#total
    weights = np.ones(total)#total
    for i in range(total):
        dist[i][0] = (x[comb[i][0]]+1.)*(kappa[1]-kappa[0])/2 + kappa[0]
        dist[i][1] = (x[comb[i][1]]+1.)*(cb1[1]-cb1[0])/2 + cb1[0]
        dist[i][2] = (x[comb[i][2]]+1.)*(cb2[1]-cb2[0])/2 + cb2[0]
        dist[i][3] = (x[comb[i][3]]+1.)*(sigma[1]-sigma[0])/2 + sigma[0]
        for j in range(n):
            weights[i] *= y[comb[i][j]]


    #print(dist)
    return dist, weights

print(genSC(3))


#print(kap)
#print(dist)


