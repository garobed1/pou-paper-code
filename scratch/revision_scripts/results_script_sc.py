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

if len(sys.argv) == 2: #assume we're given the name of a python file and import it
    # All variables not initialized come from this import
    ssplit = sys.argv[1].split(sep='/')
    suse = '.'.join(ssplit)
    set = importlib.import_module(suse)
    # from results_settings import * 
    # Generate results folder and list of inputs
    title = f"{set.header}_{set.prob}_{set.dim}D"
    if(set.path == None):
        path = "."
    else:
        path = set.path
    if not os.path.isdir(f"{path}/{title}"):
        os.mkdir(f"{path}/{title}")
    shutil.copy(f"./{sys.argv[1]}.py", f"{path}/{title}/settings.py")

# All variables not initialized come from this import
else:
    import results_settings as set
    # Generate results folder and list of inputs
    title = f"{set.header}_{set.prob}_{set.dim}D"
    if(set.path == None):
        path = "."
    else:
        path = set.path

    if not os.path.isdir(f"{path}/{title}"):
        os.mkdir(f"{path}/{title}")
    shutil.copy("./results_settings.py", f"{path}/{title}/settings.py")

# Problem Settings
ntr = set.ntr
nt0 = set.nt0

ud = False
# ud = set.perturb
trueFunc = GetProblem(set.prob, set.dim, use_design=ud)
xlimits = trueFunc.xlimits
N = set.dim*[nt0]
pdfs = set.dim*['uniform']
sampling = LHS(xlimits=xlimits, criterion='m')
sampler = CollocationSampler(None, N=N,
                                xlimits=xlimits, 
                                probability_functions=pdfs, 
                                retain_uncertain_points=True)



# Error
xtest = None 
ftest = None
testdata = None

if(set.prob != 'shock'):
    xtest = sampling(set.Nerr)
    ftest = trueFunc(xtest)
    testdata = [xtest, ftest, None]


# Print Conditions
print("\n")
print("\n")
print("Surrogate Type       : ", "SC")
print("Problem              : ", set.prob)
print("Problem Dimension    : ", set.dim)
print("Initial P Order      : ", nt0)
print("Added P Order        : ", ntr)
print("Total P Order        : ", nt0+ntr)
print("RMSE Size            : ", set.Nerr)
print("\n")




print("Computing Non-Adaptive Designs ...")

print("Training Initial Surrogate ...")

# Initial Design Surrogate
modelbase = PCEStrictSurrogate(bounds=xlimits, sampler=sampler)
modelbase.options.update({"print_global":False})


# Non-Adaptive Model Error
errhrms = []
errhmean = []
csamplers = []
modelK = copy.deepcopy(modelbase)
for n in range(ntr):
    N = set.dim*[nt0+n]
    csamplers.append(CollocationSampler(None, N=N,
                                xlimits=xlimits, 
                                probability_functions=pdfs, 
                                retain_uncertain_points=True))
    xt = csamplers[n].current_samples['x']
    ft = trueFunc(xt)
    gt = convert_to_smt_grads(trueFunc, xt)
    csamplers[n].set_evaluated_func(ft)
    csamplers[n].set_evaluated_grad(gt)
    modelK.set_collocation_sampler(csamplers[n])
    modelK.set_training_values(csamplers[n].current_samples['x'], csamplers[n].current_samples['f'])
    # import pdb; pdb.set_trace()

    modelK.train()
    errhrms.append(rmse(modelK, trueFunc, N=set.Nerr, xdata=xtest, fdata=ftest))
    errhmean.append(meane(modelK, trueFunc, N=set.Nerr, xdata=xtest, fdata=ftest))

    

# Data
with open(f'{path}/{title}/modelf.pickle', 'wb') as f:
    pickle.dump(modelK, f)
with open(f'{path}/{title}/csamplers.pickle', 'wb') as f:
    pickle.dump(csamplers, f)
with open(f'{path}/{title}/errhrms.pickle', 'wb') as f:
    pickle.dump(errhrms, f)
with open(f'{path}/{title}/errhmean.pickle', 'wb') as f:
    pickle.dump(errhmean, f)
        
import pdb; pdb.set_trace()