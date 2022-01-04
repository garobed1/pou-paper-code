import numpy as np

import pougrad

# test the surrogate

# define an analytic 2d function
# x: array of 2
# returns f: single value
def trueFunction(x):
    f = x[0]*x[0] + 2*x[1] + 1
    return f 

# function gradient
def trueFunctionGrad(x):
    g = np.zeros(2)
    g[0] = 2*x[0]
    g[1] = 2
    return g

# generate "training" and "test" data
ntr = 10
nte = 50

tr = np.random.rand(ntr, 2)
fr = np.zeros(ntr)
gr = np.zeros((ntr,2))

te = np.random.rand(nte, 2)
fe = np.zeros(nte)

for i in range(ntr):
    fr[i] = trueFunction(tr[i])
    gr[i] = trueFunctionGrad(tr[i])

for i in range(nte):
    fe[i] = trueFunction(te[i])

pou = pougrad.POUSurrogate(tr,fr,gr,10)

# test the surrogate
fes = np.zeros(nte)

for i in range(nte):
    fes[i] = pou.eval(te[i])

err = fes - fe
import pdb; pdb.set_trace()