import sys, os
import shutil
import copy
import pickle
import importlib
# from mpi4py import MPI

import numpy as np
#import matplotlib.pyplot as plt
from infill.refinecriteria import looCV, HessianFit, TEAD
from infill.aniso_criteria import AnisotropicRefine
from infill.taylor_criteria import TaylorRefine, TaylorExploreRefine
from infill.hess_criteria import HessianRefine, POUSSA
from infill.loocv_criteria import POUSFCVT
from infill.aniso_transform import AnisotropicTransform
from infill.getxnew import getxnew, adaptivesampling
from optimization.defaults import DefaultOptOptions
from utils.sutils import divide_cases, convert_to_smt_grads
from utils.error import rmse, meane
from functions.problem_picker import GetProblem


from surrogate.pce_strict import PCEStrictSurrogate
from optimization.robust_objective import CollocationSampler
from smt.sampling_methods import LHS
from scipy.stats import qmc



"""
Generate strict PCE models and estimate error
"""


# All variables not initialized come from this import
from results_settings import * 
# Generate results folder and list of inputs
title = f"{header}_{prob}_{dim}D"
if(path == None):
    path = "."
if rank == 0:
    if not os.path.isdir(f"{path}/{title}"):
        os.mkdir(f"{path}/{title}")
    shutil.copy("./results_settings.py", f"{path}/{title}/settings.py")

# Problem Settings
ud = False
ud = perturb
trueFunc = GetProblem(prob, dim, use_design=ud)
xlimits = trueFunc.xlimits
N = dim*[nt0]
pdfs = dim*['uniform']
sampling = CollocationSampler(None, N=N,
                                xlimits=xlimits, 
                                probability_functions=pdfs, 
                                retain_uncertain_points=True)



# Error
xtest = None 
ftest = None
testdata = None

if(prob != 'shock'):
    xtest = sampling(Nerr)
    ftest = trueFunc(xtest)
    testdata = [xtest, ftest, None]


# Print Conditions
if rank == 0:
    print("\n")
    print("\n")
    print("Surrogate Type       : ", stype)
    print("Problem              : ", prob)
    print("Problem Dimension    : ", dim)
    print("Initial P Order      : ", nt0)
    print("Added P Order        : ", ntr)
    print("Total P Order        : ", nt0+ntr)
    print("RMSE Size            : ", Nerr)
    print("\n")




print("Computing Non-Adaptive Designs ...")

if rank == 0:
    print("Training Initial Surrogate ...")

# Initial Design Surrogate
modelbase = PCEStrictSurrogate(bounds=xlimits, sampler=sampler)
# modelbase.options.update({"hyper_opt":'TNC'})
modelbase.options.update({"corr":corr})
modelbase.options.update({"poly":poly})
modelbase.options.update({"n_start":5})
modelbase.options.update({"print_global":False})


# Non-Adaptive Model Error
errhrms = []
errhmean = []
csamplers = []
modelK = copy.deepcopy(modelbase)
for n in range(ntr):
    N = dim*[nt0+n]
    csamplers.append(CollocationSampler(None, N=N,
                                xlimits=xlimits, 
                                probability_functions=pdfs, 
                                retain_uncertain_points=True))
    modelK.set_collocation_sampler(csamplers[n])
    modelK.set_training_values(csamplers[n].current_samples['x'], csamplers[n].current_samples['f'])

    modelK.train()
    errhrms.append(rmse(modelK, trueFunc, N=Nerr, xdata=xtest, fdata=ftest))
    errhmean.append(meane(modelK, trueFunc, N=Nerr, xdata=xtest, fdata=ftest))

    

# LHS Data
with open(f'{path}/{title}/csamplers.pickle', 'wb') as f:
    pickle.dump(csamplers, f)
with open(f'{path}/{title}/errhkrms.pickle', 'wb') as f:
    pickle.dump(errhrms, f)
with open(f'{path}/{title}/errhmean.pickle', 'wb') as f:
    pickle.dump(errhmean, f)
        
