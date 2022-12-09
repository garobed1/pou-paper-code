import sys, os
import shutil
import copy
import pickle
from mpi4py import MPI
sys.path.insert(1,"../surrogate")

import numpy as np
from error import rmse, meane
from example_problems import FuhgP3, FuhgSingleHump, Ishigami, Peaks2D, QuadHadamard, MultiDimJump, MultiDimJumpTaper, FuhgP8, FuhgP9, FuhgP10, FakeShock
from smt.problems import Branin, Sphere, LpNorm, Rosenbrock, WaterFlow, WeldedBeam, RobotArm, CantileverBeam, WingWeight, TensorProduct
from smt.sampling_methods import LHS


pFuncs = []
### 1D
# pFuncs.append(MultiDimJump(ndim=1, alpha = 8.)) #0 
# #pFuncs.append(TensorProduct(ndim=1, func = "cos")) #1
# pFuncs.append(FuhgP3(ndim=1)) #1
# #pFuncs.append(LpNorm(ndim=1)) #2
# pFuncs.append(FuhgSingleHump(ndim=1)) #2

### 2D
# pFuncs.append(MultiDimJump(ndim=2, alpha = 8.)) #3
pFuncs.append(Rosenbrock(ndim=2)) #4
# pFuncs.append(Peaks2D(ndim=2)) #5

# ### 3D
# pFuncs.append(Ishigami(ndim=3))

# ### 4D
# #pFuncs.append(Rosenbrock(ndim=4)) #6

# ### 6D
# #pFuncs.append(MultiDimJump(ndim=6, alpha = 8.)) #7
# #pFuncs.append(LpNorm(ndim=6)) #8

# ### 8D
# #pFuncs.append(Rosenbrock(ndim=8)) #9
# pFuncs.append(WaterFlow(ndim=8)) #9

# ### HIGH DIM
# pFuncs.append(WingWeight(ndim=10)) #10
#pFuncs.append(MultiDimJump(ndim=12, alpha = 8.)) #11
#pFuncs.append(QuadHadamard(ndim=16)) #12

nfunc = len(pFuncs)
Nerrbs = [8]#[2, 8]
if len(Nerrbs) > 1:
    numc = int(Nerrbs[1] - Nerrbs[0] + 1)
    Nerr = np.logspace(Nerrbs[0], Nerrbs[1], num=numc, dtype=int)
    pmeans = np.zeros([nfunc, numc])
    pstdvs = np.zeros([nfunc, numc])
else:
    numc = 1
    Nerr = [10**Nerrbs[0]]
    pmeans = np.zeros([nfunc, numc])
    pstdvs = np.zeros([nfunc, numc])

for j in range(numc):
    print("Sample Size: ", Nerr[j])
    
    xlimits = []
    sampling = []
    xtest = []
    ftest = []
    for i in range(nfunc):
        print("Problem: ", i)
        xlimits.append(pFuncs[i].xlimits)
        sampling.append(LHS(xlimits=xlimits[i]))
        xtest.append(sampling[i](Nerr[j]))
        ftest.append(pFuncs[i](xtest[i]))

        pmeans[i][j] = np.mean(ftest[i])
        pstdvs[i][j] = np.std(ftest[i])


with open(f'./rosen_paper_stats.pickle', 'wb') as f:
    pickle.dump((pmeans, pstdvs), f)

import pdb; pdb.set_trace()