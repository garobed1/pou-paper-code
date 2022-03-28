import sys, os
import copy
import pickle
from mpi4py import MPI
sys.path.insert(1,"../surrogate")

import numpy as np
#import matplotlib.pyplot as plt
from refinecriteria import looCV, HessianFit
from aniso_criteria import AnisotropicRefine
from getxnew import getxnew, adaptivesampling
from defaults import DefaultOptOptions
from utils import divide_cases
from error import rmse, meane

from example_problems import MultiDimJump, MultiDimJumpTaper, FuhgP8
from smt.problems import Sphere, LpNorm, Rosenbrock
from smt.surrogate_models import KPLS, GEKPLS, KRG
#from smt.surrogate_models.rbf import RBF
from pougrad import POUSurrogate
from smt.sampling_methods import LHS

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

"""
Perform adaptive sampling and estimate error
"""

# Conditions
Nruns = 5
multistart = 3     #aniso opt multistart
stype = "gekpls"    #surrogate type
rtype = "aniso"     #criteria type
corr  = "abs_exp" #kriging correlation
poly  = "linear"    #kriging regression 
prob  = "arctantaper"    #problem
extra = 1           #gek extra points
dim = 6            #problem dimension
rho = 10            #POU parameter
nt0  = dim*10       #initial design size
ntr = dim*10        #number of points to add
ntot = nt0 + ntr    #total number of points
batch = 0.1         #batch size for refinement, as a percentage of ntr
Nerr = 5000       #number of test points to evaluate the error
pperb = int(batch*ntr)
if(pperb == 0):
    pperb = 1

# Fan out parallel cases
cases = divide_cases(Nruns, size)

# Refinement Settings
neval = 1+dim*2
hess  = "neighborhood"
interp = "honly"
criteria = "distance"
perturb = True

# Problem Settings
alpha = 8.       #arctangent jump strength
if(prob == "arctan"):
    trueFunc = MultiDimJump(ndim=dim, alpha=alpha)
elif(prob == "arctantaper"):
    trueFunc = MultiDimJumpTaper(ndim=dim, alpha=alpha)
elif(prob == "rosenbrock"):
    trueFunc = Rosenbrock(ndim=dim)
elif(prob == "sphere"):
    trueFunc = Sphere(ndim=dim)
elif(prob == "fuhgp8"):
    trueFunc = FuhgP8(ndim=dim)
else:
    raise ValueError("Given problem not valid.")
xlimits = trueFunc.xlimits
sampling = LHS(xlimits=xlimits, criterion='m')

# Error
xtest = None 
ftest = None
testdata = None
if rank == 0:
    xtest = sampling(Nerr)
    ftest = trueFunc(xtest)
    testdata = [xtest, ftest]

xtest = comm.bcast(xtest, root=0)
ftest = comm.bcast(ftest, root=0)
testdata = comm.bcast(testdata, root=0)

# Adaptive Sampling Conditions
options = DefaultOptOptions
options["localswitch"] = True
options["errorcheck"] = testdata

# Print Conditions
if rank == 0:
    print("\n")
    print("\n")
    print("Surrogate Type       : ", stype)
    print("Refinement Type      : ", rtype)
    print("Refinement Multistart: ", multistart)
    print("Correlation Function : ", corr)
    print("Regression Function  : ", poly)
    print("GEK Extra Points     : ", extra)
    print("Problem              : ", prob)
    print("Problem Dimension    : ", dim)
    print("Initial Sample Size  : ", nt0)
    print("Refined Points Size  : ", ntr)
    print("Total Points         : ", ntot)
    print("Points Per Iteration : ", int(batch*ntr))
    print("RMSE Size            : ", Nerr)
    print("\n")




    print("Computing Initial Designs for Adaptive Sampling ...")

    # Adaptive Sampling Initial Design
xtrain0 = []
ftrain0 = []
gtrain0 = []
if rank == 0:
    for n in range(Nruns):
        xtrain0.append(sampling(nt0))
        ftrain0.append(trueFunc(xtrain0[n]))
        gtrain0.append(np.zeros([nt0,dim]))
        for i in range(dim):
            gtrain0[n][:,i:i+1] = trueFunc(xtrain0[n],i)

xtrain0 = comm.bcast(xtrain0, root=0)
ftrain0 = comm.bcast(ftrain0, root=0)
gtrain0 = comm.bcast(gtrain0, root=0)


samplehistK = np.linspace(nt0, ntot, int((ntot-nt0)/pperb)+1, dtype=int)
if rank == 0:
    print("Computing Non-Adaptive Designs ...")

    # Final Design(s)
xtrainK = []
ftrainK = []
gtrainK = []
if rank == 0:
    for n in range(Nruns):
        xtrainK.append([])
        ftrainK.append([])
        gtrainK.append([])
        for m in range(len(samplehistK)):
            xtrainK[n].append(sampling(nt0+m*int(batch*ntr)))
            ftrainK[n].append(trueFunc(xtrainK[n][m]))
            gtrainK[n].append(np.zeros([nt0+m*int(batch*ntr),dim]))
            for i in range(dim):
                gtrainK[n][m][:,i:i+1] = trueFunc(xtrainK[n][m],i)

xtrainK = comm.bcast(xtrainK, root=0)
ftrainK = comm.bcast(ftrainK, root=0)
gtrainK = comm.bcast(gtrainK, root=0)

if rank == 0:
    print("Training Initial Surrogate ...")

# Initial Design Surrogate
if(stype == "gekpls"):
    modelbase = GEKPLS(xlimits=xlimits)
    modelbase.options.update({"extra_points":extra})
    modelbase.options.update({"corr":corr})
    modelbase.options.update({"poly":poly})
    modelbase.options.update({"n_start":5})

elif(stype == "pou"):
    modelbase = POUSurrogate()
    modelbase.options.update({"rho":rho})
elif(stype == "kpls"):
    modelbase = KPLS()
    modelbase.options.update({"corr":corr})
    modelbase.options.update({"poly":poly})
    modelbase.options.update({"n_start":5})
else:
    modelbase = KRG()
    modelbase.options.update({"corr":corr})
    modelbase.options.update({"poly":poly})
    modelbase.options.update({"n_start":5})
modelbase.options.update({"print_global":False})

model0 = []
co = 0

for n in cases[rank]: #range(Nruns):
    model0.append(copy.deepcopy(modelbase))
    model0[co].set_training_values(xtrain0[n], ftrain0[n])
    if(isinstance(model0[co], GEKPLS) or isinstance(model0[co], POUSurrogate)):
        for i in range(dim):
            model0[co].set_training_derivatives(xtrain0[n], gtrain0[n][:,i:i+1], i)
    model0[co].train()
    co += 1



if rank == 0:
    print("Computing Initial Surrogate Error ...")

# Initial Model Error
err0rms = []
err0mean = []
co = 0
for n in cases[rank]:
    err0rms.append(rmse(model0[co], trueFunc, N=Nerr, xdata=xtest, fdata=ftest))
    err0mean.append(meane(model0[co], trueFunc, N=Nerr, xdata=xtest, fdata=ftest))
    co += 1


if rank == 0:
    print("Computing Final Non-Adaptive Surrogate Error ...")

# Non-Adaptive Model Error
errkrms = []
errkmean = []
modelK = copy.deepcopy(modelbase)
co = 0
for n in cases[rank]:
    errkrms.append([])
    errkmean.append([])
    for m in range(len(samplehistK)):
        modelK.set_training_values(xtrainK[n][m], ftrainK[n][m])
        if(isinstance(modelbase, GEKPLS) or isinstance(modelbase, POUSurrogate)):
            for i in range(dim):
                modelK.set_training_derivatives(xtrainK[n][m], gtrainK[n][m][:,i:i+1], i)
        modelK.train()
        errkrms[co].append(rmse(modelK, trueFunc, N=Nerr, xdata=xtest, fdata=ftest))
        errkmean[co].append(meane(modelK, trueFunc, N=Nerr, xdata=xtest, fdata=ftest))
    co += 1


if rank == 0:
    print("Initial Refinement Criteria ...")

# Initial Refinement Criteria
RC0 = []
co = 0
for n in cases[rank]:
    if(rtype == "aniso"):
        RC0.append(AnisotropicRefine(model0[co], gtrain0[n], improve=pperb, neval=neval, hessian=hess, interp=interp, multistart=multistart) )
    else:
        raise ValueError("Given criteria not valid.")
    co += 1

if rank == 0:
    print("Performing Adaptive Sampling ...")

# Perform Adaptive Sampling
modelf = []
RCF = []
hist = []
errhrms = []
errhmean = []
co = 0
for n in cases[rank]:
    print("RUN: ", n)
    mf, rF, hf, ef, ef2 = adaptivesampling(trueFunc, model0[co], RC0[co], xlimits, ntr, options=options)
    modelf.append(mf)
    RCF.append(rF)
    hist.append(hf)
    errhrms.append(ef)
    errhmean.append(ef2)
    co += 1

modelf = comm.gather(modelf, root=0)
RCF = comm.gather(RCF, root=0)
hist = comm.gather(hist, root=0)
errkrms = comm.gather(errkrms, root=0)
errkmean = comm.gather(errkmean, root=0)
err0rms = comm.gather(err0rms, root=0)
err0mean = comm.gather(err0mean, root=0)
errhrms = comm.gather(errhrms, root=0)
errhmean = comm.gather(errhmean, root=0)


if rank == 0:
    print("\n")
    print("Experiment Complete")

    title = f'{prob}_{rtype}_{stype}_{corr}_{dim}d_{Nruns}runs_{nt0}to{ntot}pts_{batch}batch_{multistart}mstart'
    if not os.path.isdir(title):
        os.mkdir(title)

    # LHS Data
    with open(f'./{title}/xk.pickle', 'wb') as f:
        pickle.dump(xtrainK, f)

    with open(f'./{title}/fk.pickle', 'wb') as f:
        pickle.dump(ftrainK, f)

    with open(f'./{title}/gk.pickle', 'wb') as f:
        pickle.dump(gtrainK, f)

    with open(f'./{title}/errkrms.pickle', 'wb') as f:
        pickle.dump(errkrms, f)

    with open(f'./{title}/errkmean.pickle', 'wb') as f:
        pickle.dump(errkmean, f)


    # Adaptive Data
    with open(f'./{title}/modelf.pickle', 'wb') as f:
        pickle.dump(modelf, f)

    with open(f'./{title}/err0rms.pickle', 'wb') as f:
        pickle.dump(err0rms, f)

    with open(f'./{title}/err0mean.pickle', 'wb') as f:
        pickle.dump(err0mean, f)

    with open(f'./{title}/hist.pickle', 'wb') as f:
        pickle.dump(hist, f)

    with open(f'./{title}/errhrms.pickle', 'wb') as f:
        pickle.dump(errhrms, f)

    with open(f'./{title}/errhmean.pickle', 'wb') as f:
        pickle.dump(errhmean, f)





