import sys, os
import shutil
import copy
import pickle
from mpi4py import MPI
sys.path.insert(1,"../surrogate")

import numpy as np
#import matplotlib.pyplot as plt
from refinecriteria import looCV, HessianFit, TEAD
from aniso_criteria import AnisotropicRefine
from taylor_criteria import TaylorRefine, TaylorExploreRefine
from hess_criteria import HessianRefine, POUSSA
from loocv_criteria import POUSFCVT
from aniso_transform import AnisotropicTransform
from getxnew import getxnew, adaptivesampling
from defaults import DefaultOptOptions
from sutils import divide_cases
from error import rmse, meane
from problem_picker import GetProblem
from smt.surrogate_models import KPLS, GEKPLS, KRG
from direct_gek import DGEK
#from smt.surrogate_models.rbf import RBF
from pougrad import POUSurrogate, POUHessian
from smt.sampling_methods import LHS
from scipy.stats import qmc

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

"""
Perform adaptive sampling and estimate error
"""

# All variables not initialized come from this import
from results_settings import *

Nruns = size*runs_per_proc

# Generate results folder and list of inputs
title = f"{header}_{prob}_{dim}D"
if(path == None):
    path = "."
if rank == 0:
    if not os.path.isdir(f"{path}/{title}"):
        os.mkdir(f"{path}/{title}")
    shutil.copy("./results_settings.py", f"{path}/{title}/settings.py")

# Problem Settings
trueFunc = GetProblem(prob, dim)
xlimits = trueFunc.xlimits
sampling = LHS(xlimits=xlimits, criterion='m')
if(rtype == 'anisotransform'):
    sequencer = []
    for n in range(Nruns):
        sequencer.append(qmc.Halton(d=dim))

# Fan out parallel cases
cases = divide_cases(Nruns, size)

# Error
xtest = None 
ftest = None
testdata = None
if rank == 0:
    xtest = sampling(Nerr)
    ftest = trueFunc(xtest)
    if(dim > 3):
        intervals = np.arange(0, ntr, dim)
        intervals = np.append(intervals, ntr - 1)
        intervals = np.delete(intervals, 0)
    else:
        intervals = np.arange(0, ntr)
    testdata = [xtest, ftest, intervals]

xtest = comm.bcast(xtest, root=0)
ftest = comm.bcast(ftest, root=0)
testdata = comm.bcast(testdata, root=0)
# Adaptive Sampling Conditions
options = DefaultOptOptions
options["local"] = local
options["localswitch"] = True
options["errorcheck"] = testdata
options["multistart"] = mstarttype
options["lmethod"] = opt

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
    print("Points Per Iteration : ", batch)
    print("RMSE Size            : ", Nerr)
    print("\n")




    print("Computing Initial Designs for Adaptive Sampling ...")

    # Adaptive Sampling Initial Design
xtrain0 = []
ftrain0 = []
gtrain0 = []
if rank == 0:
    if(rtype == 'anisotransform'):
        for n in range(Nruns):
            sample = sequencer[n].random(nt0)
            sample = qmc.scale(sample, xlimits[:,0], xlimits[:,1])
            xtrain0.append(sample)
            ftrain0.append(trueFunc(xtrain0[n]))
            gtrain0.append(np.zeros([nt0,dim]))
            for i in range(dim):
                gtrain0[n][:,i:i+1] = trueFunc(xtrain0[n],i)


    else:
        for n in range(Nruns):
            xtrain0.append(sampling(nt0))
            ftrain0.append(trueFunc(xtrain0[n]))
            gtrain0.append(np.zeros([nt0,dim]))
            for i in range(dim):
                gtrain0[n][:,i:i+1] = trueFunc(xtrain0[n],i)

xtrain0 = comm.bcast(xtrain0, root=0)
ftrain0 = comm.bcast(ftrain0, root=0)
gtrain0 = comm.bcast(gtrain0, root=0)


samplehistK = np.linspace(nt0, ntot, LHS_batch+1, dtype=int)

if rank == 0:
    print("Computing Non-Adaptive Designs ...")

if(not skip_LHS):
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
                xtrainK[n].append(sampling(nt0+m*pperbk))
                ftrainK[n].append(trueFunc(xtrainK[n][m]))
                gtrainK[n].append(np.zeros([nt0+m*pperbk,dim]))
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
    # modelbase.options.update({"hyper_opt":'TNC'})
    # modelbase.options.update({"theta0":t0g})
    # modelbase.options.update({"theta_bounds":tbg})
    modelbase.options.update({"n_comp":dim})
    modelbase.options.update({"extra_points":extra})
    modelbase.options.update({"corr":corr})
    modelbase.options.update({"poly":poly})
    modelbase.options.update({"n_start":5})
elif(stype == "dgek"):
    modelbase = DGEK(xlimits=xlimits)
    # modelbase.options.update({"hyper_opt":'TNC'})
    modelbase.options.update({"corr":corr})
    modelbase.options.update({"poly":poly})
    modelbase.options.update({"n_start":5})
    modelbase.options.update({"theta0":t0})
    modelbase.options.update({"theta_bounds":tb})
elif(stype == "pou"):
    modelbase = POUSurrogate()
    modelbase.options.update({"rho":rho})
elif(stype == "pouhess"):
    modelbase = POUHessian(bounds=xlimits, rscale=rscale)
    modelbase.options.update({"rho":rho})
    modelbase.options.update({"neval":neval})
elif(stype == "kpls"):
    modelbase = KPLS()
    # modelbase.options.update({"hyper_opt":'TNC'})
    modelbase.options.update({"n_comp":dim})
    modelbase.options.update({"corr":corr})
    modelbase.options.update({"poly":poly})
    modelbase.options.update({"n_start":5})
else:
    modelbase = KRG()
    # modelbase.options.update({"hyper_opt":'TNC'})
    modelbase.options.update({"corr":corr})
    modelbase.options.update({"poly":poly})
    modelbase.options.update({"n_start":5})
modelbase.options.update({"print_global":False})
# modelbase.options.update({"print_training":True})
# modelbase.options.update({"print_prediction":True})
# modelbase.options.update({"print_problem":True})
# modelbase.options.update({"print_solver":True})

model0 = []
co = 0

for n in cases[rank]: #range(Nruns):
    model0.append(copy.deepcopy(modelbase))
    model0[co].set_training_values(xtrain0[n], ftrain0[n])
    if(isinstance(model0[co], GEKPLS) or isinstance(model0[co], POUSurrogate) or isinstance(model0[co], DGEK) or isinstance(model0[co], POUHessian)):
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

if(not skip_LHS):
    # Non-Adaptive Model Error
    errkrms = []
    errkmean = []
    modelK = copy.deepcopy(modelbase)
    co = 0
    for n in cases[rank]:
        errkrms.append([])
        errkmean.append([])
        xtrainK[n][0] = xtrain0[n]
        ftrainK[n][0] = ftrain0[n]
        gtrainK[n][0] = gtrain0[n]

        for m in range(len(samplehistK)):
            modelK.set_training_values(xtrainK[n][m], ftrainK[n][m])
            if(isinstance(modelbase, GEKPLS) or isinstance(modelbase, POUSurrogate) or isinstance(model0[co], DGEK)):
                for i in range(dim):
                    modelK.set_training_derivatives(xtrainK[n][m], gtrainK[n][m][:,i:i+1], i)
            modelK.train()
            errkrms[co].append(rmse(modelK, trueFunc, N=Nerr, xdata=xtest, fdata=ftest))
            errkmean[co].append(meane(modelK, trueFunc, N=Nerr, xdata=xtest, fdata=ftest))
        co += 1

    errkrms = comm.gather(errkrms, root=0)
    errkmean = comm.gather(errkmean, root=0)
    
    # LHS Data
    with open(f'{path}/{title}/xk.pickle', 'wb') as f:
        pickle.dump(xtrainK, f)

    with open(f'{path}/{title}/fk.pickle', 'wb') as f:
        pickle.dump(ftrainK, f)

    with open(f'{path}/{title}/gk.pickle', 'wb') as f:
        pickle.dump(gtrainK, f)

    with open(f'{path}/{title}/errkrms.pickle', 'wb') as f:
        pickle.dump(errkrms, f)

    with open(f'{path}/{title}/errkmean.pickle', 'wb') as f:
        pickle.dump(errkmean, f)
        

if rank == 0:
    print("Initial Refinement Criteria ...")

# Initial Refinement Criteria
RC0 = []
co = 0
for n in cases[rank]:
    if(rtype == "aniso"):
        RC0.append(AnisotropicRefine(model0[co], gtrain0[n], xlimits, rscale=rscale, nscale=nscale, improve=pperb, neval=neval, hessian=hess, interp=interp, bpen=bpen, objective=obj, multistart=multistart) )
    elif(rtype == "anisotransform"):
        RC0.append(AnisotropicTransform(model0[co], sequencer[n], gtrain0[n], improve=pperb, nmatch=nmatch, neval=neval, hessian=hess, interp=interp))
    elif(rtype == "tead"):
        RC0.append(TEAD(model0[co], gtrain0[n], xlimits, gradexact=True))
    elif(rtype == "taylor"):
        RC0.append(TaylorRefine(model0[co], gtrain0[n], xlimits, volume_weight=perturb, rscale=rscale, improve=pperb, multistart=multistart) )
    elif(rtype == "taylorexp"):
        RC0.append(TaylorExploreRefine(model0[co], gtrain0[n], xlimits, rscale=rscale, improve=pperb, objective=obj, multistart=multistart) ) 
    elif(rtype == "hess"):
        RC0.append(HessianRefine(model0[co], gtrain0[n], xlimits, neval=neval, rscale=rscale, improve=pperb, multistart=multistart, print_rc_plots=rc_print) )
    elif(rtype == "poussa"):
        RC0.append(POUSSA(model0[co], gtrain0[n], xlimits, improve=pperb, multistart=multistart, print_rc_plots=rc_print))
    elif(rtype == "pousfcvt"):
        RC0.append(POUSFCVT(model0[co], gtrain0[n], xlimits, improve=pperb, multistart=multistart, print_rc_plots=rc_print))
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


try:
    comp_hist
    iters = len(ef)
    itersk = len(LHS_batch)
    histc = []
    ind_alt = np.linspace(0, iters, itersk, dtype=int)
    for n in range(co):
        histc.append(hf[n][ind_alt])
    hist = histc
    if rank == 0:
        print("hello from comp hist")
except:
    pass

modelf = comm.gather(modelf, root=0)
RCF = comm.gather(RCF, root=0)
hist = comm.gather(hist, root=0)
err0rms = comm.gather(err0rms, root=0)
err0mean = comm.gather(err0mean, root=0)
errhrms = comm.gather(errhrms, root=0)
errhmean = comm.gather(errhmean, root=0)


if rank == 0:
    print("\n")
    print("Experiment Complete")

    # Adaptive Data
    with open(f'{path}/{title}/modelf.pickle', 'wb') as f:
        pickle.dump(modelf, f)

    with open(f'{path}/{title}/err0rms.pickle', 'wb') as f:
        pickle.dump(err0rms, f)

    with open(f'{path}/{title}/err0mean.pickle', 'wb') as f:
        pickle.dump(err0mean, f)

    with open(f'{path}/{title}/hist.pickle', 'wb') as f:
        pickle.dump(hist, f)

    with open(f'{path}/{title}/errhrms.pickle', 'wb') as f:
        pickle.dump(errhrms, f)

    with open(f'{path}/{title}/errhmean.pickle', 'wb') as f:
        pickle.dump(errhmean, f)

    if(dim > 3):
        with open(f'{path}/{title}/intervals.pickle', 'wb') as f:
            pickle.dump(intervals, f)





