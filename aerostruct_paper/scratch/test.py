import numpy as np

import pougrad

# test the surrogate

# define an analytic 2d function
# x: array of 2
# returns f: single value
def trueFunction(x):
    f = x[0]*x[0] + x[1]*x[1] #+ 2*x[1] + 1
    return f 

# function gradient
def trueFunctionGrad(x):
    g = np.zeros(2)
    g[0] = 2*x[0]
    g[1] = 2*x[1]
    return g

# generate "training" and "test" data
dim = 2
ntr = 25
nte = 10

tr = np.random.rand(ntr, dim)*20 - 10
fr = np.zeros(ntr)
gr = np.zeros((ntr,dim))

te = np.random.rand(nte, dim)*20 - 10
fe = np.zeros(nte)

for i in range(ntr):
    fr[i] = trueFunction(tr[i])
    gr[i] = trueFunctionGrad(tr[i])

for i in range(nte):
    fe[i] = trueFunction(te[i])

pou = pougrad.POUSurrogate(tr,fr,gr,1000)

# test the surrogate
fes = np.zeros(nte)

for i in range(nte):
    fes[i] = pou.eval(te[i])

err = fes - fe

# check x gradient

step = 1e-6

festep = np.zeros(dim)
for k in range(dim):
    stepvec = np.zeros(dim)
    stepvec[k] = step
    festep[k] = pou.eval(te[0] + stepvec)

xgradfd = (festep - fes[0])/step
xgradad = pou.evalGrad(te[0])

import pdb; pdb.set_trace()