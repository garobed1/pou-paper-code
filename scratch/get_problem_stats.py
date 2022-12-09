import sys, os
import shutil
import copy
import pickle
from mpi4py import MPI
sys.path.insert(1,"../surrogate")

import numpy as np
from error import rmse, meane
from example_problems import Peaks2D, QuadHadamard, MultiDimJump, MultiDimJumpTaper, FuhgP8, FuhgP9, FuhgP10, FakeShock
from smt.problems import Branin, Sphere, LpNorm, Rosenbrock, WaterFlow, WeldedBeam, RobotArm, CantileverBeam, WingWeight, TensorProduct
from smt.sampling_methods import LHS


pFuncs = []
### 1D
pFuncs.append(MultiDimJump(ndim=1, alpha = 8.)) #0 
pFuncs.append(TensorProduct(ndim=1, func = "cos")) #1
pFuncs.append(LpNorm(ndim=1)) #2

### 2D
pFuncs.append(MultiDimJump(ndim=2, alpha = 8.)) #3
pFuncs.append(Rosenbrock(ndim=2)) #4
pFuncs.append(Peaks2D(ndim=2)) #5

### 4D
pFuncs.append(Rosenbrock(ndim=4)) #6

### 6D
pFuncs.append(MultiDimJump(ndim=6, alpha = 8.)) #7
pFuncs.append(LpNorm(ndim=6)) #8

### 8D
pFuncs.append(Rosenbrock(ndim=8)) #9

### HIGH DIM
pFuncs.append(WingWeight(ndim=10)) #10
pFuncs.append(MultiDimJump(ndim=12, alpha = 8.)) #11
pFuncs.append(QuadHadamard(ndim=16)) #12


with open(f'./paper_stats.pickle', 'rb') as f:
    pmeans, pstdvs = pickle.load(f)

import pdb; pdb.set_trace()