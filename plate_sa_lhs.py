from pyDOE import lhs
import numpy as np
import random
import math

# Test to generate sample points in the space of Spalart-Allmaras constants,
# starting with kappa, cb1, cb2, and sigma. Assume uniform distributions as follows:

# following these approximate conditions

# (1+cb2)/sigma = [1.7, 2.7]

# Diffusion
# kappa^2 * (1+cb2)/(sigma/cb1) \approx 3


# kappa = []

# Pure monte carlo function
def mc(n, s):
    set = []
    for j in range(s):
        set.append([])
        for i in range(n):
            set[j].append(random.random())
    return set

def genLHS(s, mcs = False):
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
    
    set = []
    if mcs == False:
        set = lhs(n, samples = s)#, criterion = 'center')
    else:
        set = mc(n, s)

    dist = np.zeros([s,n])
    for i in range(s):
        dist[i][0] = set[i][0]*(kappa[1]-kappa[0]) + kappa[0]
        dist[i][1] = set[i][1]*(cb1[1]-cb1[0]) + cb1[0]
        dist[i][2] = set[i][2]*(cb2[1]-cb2[0]) + cb2[0]
        dist[i][3] = set[i][3]*(sigma[1]-sigma[0]) + sigma[0]

    #print(dist)
    return dist


#print(genLHS(10, mcs = False))
#print(kap)



