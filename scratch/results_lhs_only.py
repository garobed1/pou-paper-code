import sys, os
import shutil
import copy
import pickle
import importlib
from mpi4py import MPI

import numpy as np
from utils.sutils import divide_cases
from utils.error import rmse, meane
from functions.problem_picker import GetProblem

from smt.surrogate_models import KPLS, GEKPLS, KRG
from surrogate.direct_gek import DGEK
#from smt.surrogate_models.rbf import RBF
from surrogate.pougrad import POUSurrogate, POUHessian
from smt.sampling_methods import LHS
from scipy.stats import qmc

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

"""
Perform adaptive sampling and estimate error
"""

fresh = True



# If folder and integer arg are given, start from there

# Just taking existing settings, since we're only doing LHS
if len(sys.argv) > 1:
    fresh = False
    args = sys.argv[1:]
    fulltitle = args[0]
    ntr = args[1]

    tsplit = fulltitle.split('/')
    if len(tsplit) == 1:
        path = "."
    else:
        path = '/'.join(tsplit[:-1])
    title = tsplit[-1]

    if rank == 0:
        shutil.copy(f"{path}/{title}/settings.py", "./results_settings.py")
    #sys.path.append(title)

else:
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

from results_settings import *
Nruns = size*runs_per_proc
# Fan out parallel cases
cases = divide_cases(Nruns, size)



# Problem Settings
ud = False
ud = perturb
trueFunc = GetProblem(prob, dim, use_design=ud)
xlimits = trueFunc.xlimits
sampling = LHS(xlimits=xlimits, criterion='m')


# Error
xtest = None 
ftest = None
testdata = None

if(prob != 'shock'):
    if rank == 0:
        xtest = sampling(Nerr)
        ftest = trueFunc(xtest)

        testdata = [xtest, ftest, intervals]

    xtest = comm.bcast(xtest, root=0)
    ftest = comm.bcast(ftest, root=0)
    testdata = comm.bcast(testdata, root=0)

# Print Conditions
if rank == 0:
    print("\n")
    print("\n")
    print("Surrogate Type       : ", stype)
    print("Correlation Function : ", corr)
    print("Regression Function  : ", poly)
    print("GEK Extra Points     : ", extra)
    print("Problem              : ", prob)
    print("Problem Dimension    : ", dim)
    print("RMSE Size            : ", Nerr)
    print("\n")


if(dim > 3):
    intervals = np.arange(0, ntr, dim)
    intervals = np.append(intervals, ntr-1)
else:
    intervals = np.arange(0, ntr)
idx = np.round(np.linspace(0, len(intervals)-1, LHS_batch+1)).astype(int)
intervalsk = intervals[idx]
samplehistK = intervalsk + nt0*np.ones(len(intervalsk), dtype=int)
samplehistK[-1] += 1

if rank == 0:
    print("Computing Non-Adaptive Designs ...")


xtrainK = []
ftrainK = []
gtrainK = []

co = 0
for n in cases[rank]:
    xtrainK.append([])
    ftrainK.append([])
    gtrainK.append([])

    for m in range(len(samplehistK)):
        xtrainK[co].append(sampling(samplehistK[m]))
        ftrainK[co].append(trueFunc(xtrainK[co][m]))
        gtrainK[co].append(np.zeros([samplehistK[m],dim]))
        for i in range(dim):
            gtrainK[co][m][:,i:i+1] = trueFunc(xtrainK[co][m],i)
        
    co += 1

xtrainKlists = comm.allgather(xtrainK)
ftrainKlists = comm.allgather(ftrainK)
gtrainKlists = comm.allgather(gtrainK)

#need to put them back in order

xtrainK = [None]*Nruns
ftrainK = [None]*Nruns
gtrainK = [None]*Nruns

for s in range(size):
    for n in range(len(cases[s])):
        xtrainK[cases[s][n]] = xtrainKlists[s][n]
        ftrainK[cases[s][n]] = ftrainKlists[s][n]
        gtrainK[cases[s][n]] = gtrainKlists[s][n]

# if rank == 0:
#     print("Training Initial Surrogate ...")

# # Surrogates to test
# if(stype == "gekpls"):
#     modelbase = GEKPLS(xlimits=xlimits)
#     # modelbase.options.update({"hyper_opt":'TNC'})
#     # modelbase.options.update({"theta0":t0g})
#     # modelbase.options.update({"theta_bounds":tbg})
#     modelbase.options.update({"n_comp":dim})
#     modelbase.options.update({"extra_points":extra})
#     modelbase.options.update({"corr":corr})
#     modelbase.options.update({"poly":poly})
#     modelbase.options.update({"n_start":5})
# elif(stype == "dgek"):
#     modelbase = DGEK(xlimits=xlimits)
#     # modelbase.options.update({"hyper_opt":'TNC'})
#     modelbase.options.update({"corr":corr})
#     modelbase.options.update({"poly":poly})
#     modelbase.options.update({"n_start":5})
#     modelbase.options.update({"theta0":t0})
#     modelbase.options.update({"theta_bounds":tb})
# elif(stype == "pou"):
#     modelbase = POUSurrogate()
#     modelbase.options.update({"rho":rho})
# elif(stype == "pouhess"):
#     modelbase = POUHessian(bounds=xlimits, rscale=rscale)
#     modelbase.options.update({"rho":rho})
#     modelbase.options.update({"neval":neval})
# elif(stype == "kpls"):
#     modelbase = KPLS()
#     # modelbase.options.update({"hyper_opt":'TNC'})
#     modelbase.options.update({"n_comp":dim})
#     modelbase.options.update({"corr":corr})
#     modelbase.options.update({"poly":poly})
#     modelbase.options.update({"n_start":5})
# else:
#     modelbase = KRG()
#     # modelbase.options.update({"hyper_opt":'TNC'})
#     modelbase.options.update({"corr":corr})
#     modelbase.options.update({"poly":poly})
#     modelbase.options.update({"n_start":5})
# modelbase.options.update({"print_global":False})
# model0 = []
# co = 0
# for n in cases[rank]: #range(Nruns):
#     model0.append(copy.deepcopy(modelbase))
#     model0[co].set_training_values(xtrain0[n], ftrain0[n])
#     if(isinstance(model0[co], GEKPLS) or isinstance(model0[co], POUSurrogate) or isinstance(model0[co], DGEK) or isinstance(model0[co], POUHessian)):
#         for i in range(dim):
#             model0[co].set_training_derivatives(xtrain0[n], gtrain0[n][:,i:i+1], i)
#     model0[co].train()
#     co += 1

    # modelbase.options.update({"print_training":True})
    # modelbase.options.update({"print_prediction":True})
    # modelbase.options.update({"print_problem":True})
    # modelbase.options.update({"print_solver":True})




# if rank == 0:
#     print("Computing Final Non-Adaptive Surrogate Error ...")


# # Non-Adaptive Model Error
# errkrms = []
# errkmean = []
# modelK = copy.deepcopy(modelbase)
# co = 0
# for n in cases[rank]:
#     errkrms.append([])
#     errkmean.append([])
#     for m in range(len(samplehistK)):
#         modelK.set_training_values(xtrainK[n][m], ftrainK[n][m])
#         if(isinstance(modelbase, GEKPLS) or isinstance(modelbase, POUSurrogate) or isinstance(model0[co], DGEK)):
#             for i in range(dim):
#                 modelK.set_training_derivatives(xtrainK[n][m], gtrainK[n][m][:,i:i+1], i)
#         modelK.train()
#         errkrms[co].append(rmse(modelK, trueFunc, N=Nerr, xdata=xtest, fdata=ftest))
#         errkmean[co].append(meane(modelK, trueFunc, N=Nerr, xdata=xtest, fdata=ftest))
#     co += 1
# errkrms = comm.gather(errkrms, root=0)
# errkmean = comm.gather(errkmean, root=0)

# LHS Data
with open(f'{path}/{title}/xk.pickle', 'wb') as f:
    pickle.dump(xtrainK, f)
with open(f'{path}/{title}/fk.pickle', 'wb') as f:
    pickle.dump(ftrainK, f)
with open(f'{path}/{title}/gk.pickle', 'wb') as f:
    pickle.dump(gtrainK, f)
# with open(f'{path}/{title}/errkrms.pickle', 'wb') as f:
#     pickle.dump(errkrms, f)
# with open(f'{path}/{title}/errkmean.pickle', 'wb') as f:
#     pickle.dump(errkmean, f)
        