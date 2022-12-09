import numpy as np
import sys, os
import time
import copy
import pickle
from mpi4py import MPI
sys.path.insert(1,"../surrogate")

import matplotlib.pyplot as plt
from error import rmse, meane
from direct_gek import DGEK
from grbf import GRBF
from lsrbf import LSRBF
from shock_problem import ImpingingShock
from example_problems import  QuadHadamard, MultiDimJump, MultiDimJumpTaper, FuhgP8, FuhgP9, FuhgP10, Peaks2D
from smt.problems import Branin, Sphere, LpNorm, Rosenbrock, WaterFlow, WeldedBeam, RobotArm, CantileverBeam, TensorProduct
from smt.surrogate_models import KPLS, GEKPLS, KRG
from pougrad import POUSurrogate
from smt.sampling_methods import LHS
from scipy.stats import qmc

"""
Compute statistics by sampling a 1st-order Taylor POU model 
"""
prob  = "rosenbrock"    #problem
plot  = -1

# Conditions
dim = 2      #problem dimension
nt0 = 10*dim    #number of true function data points
ntr = 50*dim    
ns0 = ntr       #number of samples of the 1st order model  
nsr = 10*ntr
tbatch = 10
sbatch = ntr

Nerr = 5000       #number of test points to evaluate the true statistics

nruns = int((ntr-nt0)/tbatch)+1
nruns = 7

# Problem Settings
alpha = 8.       #arctangent jump strength
if(prob == "arctan"):
    trueFunc = MultiDimJump(ndim=dim, alpha=alpha)
elif(prob == "arctantaper"):
    trueFunc = MultiDimJumpTaper(ndim=dim, alpha=alpha)
elif(prob == "rosenbrock"):
    trueFunc = Rosenbrock(ndim=dim)
elif(prob == "branin"):
    trueFunc = Branin(ndim=dim)
elif(prob == "sphere"):
    trueFunc = Sphere(ndim=dim)
elif(prob == "fuhgp8"):
    trueFunc = FuhgP8(ndim=dim)
elif(prob == "fuhgp9"):
    trueFunc = FuhgP9(ndim=dim)
elif(prob == "fuhgp10"):
    trueFunc = FuhgP10(ndim=dim)
elif(prob == "waterflow"):
    trueFunc = WaterFlow(ndim=dim)
elif(prob == "weldedbeam"):
    trueFunc = WeldedBeam(ndim=dim)
elif(prob == "robotarm"):
    trueFunc = RobotArm(ndim=dim)
elif(prob == "cantilever"):
    trueFunc = CantileverBeam(ndim=dim)
elif(prob == "hadamard"):
    trueFunc = QuadHadamard(ndim=dim)
elif(prob == "peaks"):
    trueFunc = Peaks2D(ndim=dim)
elif(prob == "tensorexp"):
    trueFunc = TensorProduct(ndim=dim, func="gaussian")
elif(prob == "shock"):
    xlimits = np.zeros([dim,2])
    xlimits[0,:] = [23., 27.]
    xlimits[1,:] = [0.36, 0.51]
    trueFunc = ImpingingShock(ndim=dim, input_bounds=xlimits, comm=MPI.COMM_SELF)
else:
    raise ValueError("Given problem not valid.")
xlimits = trueFunc.xlimits
#sampling = LHS(xlimits=xlimits, criterion='m')
tsampling = qmc.Sobol(dim)
sampling = qmc.Sobol(dim)

# Error
xtest = None 
ftest = None
testdata = None
#xtest = sampling(Nerr)
xtest = tsampling.random_base2(m=12)
xtest = qmc.scale(xtest, xlimits[:,0], xlimits[:,1])
ftest = trueFunc(xtest)
testdata = [xtest, ftest]

# Training data
nt = np.linspace(nt0, ntr, nruns, dtype=int)
xt = []
ft = []
gt = []
for k in range(nruns):
    xt.append(qmc.scale(sampling.random_base2(m=k+1), xlimits[:,0], xlimits[:,1]))
    ft.append(trueFunc(xt[k]))
    gt.append(np.zeros([xt[k].shape[0], dim]))
    for j in range(dim):
        gt[k][:,j:j+1] = trueFunc(xt[k], j)
    sampling.reset()

modelpou = POUSurrogate(rho=200)
modelpou.options.update({"print_prediction":True})




# Run tests
mpou = []
errpou = []
errnmd = []
for k in range(nruns):
    # Model based
    modelpou.set_training_values(xt[k], ft[k])
    for j  in range(dim):
        modelpou.set_training_derivatives(xt[k], gt[k][:,j:j+1], j)
    modelpou.train()
    mpou.append(copy.deepcopy(modelpou))
    errpou.append(meane(modelpou, trueFunc, N=Nerr, xdata=xtest, fdata=ftest))

    # No model
    tmean, tstdev = meane(modelpou, trueFunc, N=Nerr, xdata=xtest, fdata=ftest, return_benchmark=True)
    errnmd.append((abs(tmean - np.mean(ft[k])), abs(tstdev - np.std(ft[k]))))

print("POU Sampling Statistics Error")
print(errpou)

print("No Model Statistics Error")
print(errnmd)
    
#errpoumean = [errpou[k][0] for k in range(nruns)]

# # Plot Convergence
plt.loglog(nt, [errpou[k][0] for k in range(nruns)], "k-", label=f'POU')
plt.loglog(nt, [errnmd[k][0] for k in range(nruns)], 'b-', label='NMD')

plt.xlabel("Number of samples")
plt.ylabel("Mean Error")
plt.grid()
plt.legend(loc=1)
plt.savefig(f"./pou_mean_test_err.png", bbox_inches="tight")
plt.clf()

plt.loglog(nt, [errpou[k][1] for k in range(nruns)], "k-", label=f'POU')
plt.loglog(nt, [errnmd[k][1] for k in range(nruns)], 'b-', label='NMD')

plt.xlabel("Number of samples")
plt.ylabel("Std. Dev. Error")
plt.grid()
plt.legend(loc=1)
plt.savefig(f"./pou_stdv_test_err.png", bbox_inches="tight")
plt.clf()

# # Plot points
if(dim == 1):  
    
    ndir = 75
    xlimits = trueFunc.xlimits
    x = np.linspace(xlimits[0][0], xlimits[0][1], ndir)
    TF = np.zeros([ndir, ndir])
    for run in range(nruns):
    
        Zpou = np.zeros([ndir, ndir])
        Fpou = np.zeros([ndir, ndir])
        for i in range(ndir):
            xi = np.zeros([1,1])
            xi[0] = x[i]
            TF[i] = trueFunc(xi)
            Fpou[i] = mpou[run].predict_values(xi)
            Zpou[i] = abs(Fpou[i] - TF[i])

        # Plot Non-Adaptive Error
        plt.plot(x, TF, "-k")
        plt.plot(x, Fpou, "-r", label=f'POU')
        plt.xlabel(r"$x$")
        plt.ylabel(r"$f$")
        #plt.legend(loc=1)
        plt.plot(xt[run][:,0], ft[run][:], "o", fillstyle='full', markerfacecolor='b', markeredgecolor='b', label='Initial Samples')

        plt.savefig(f"./poustat_1dplot_{run}.png", bbox_inches="tight")
        plt.clf()



