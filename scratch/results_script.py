import sys, os
import shutil
import copy
import pickle
import importlib
from mpi4py import MPI

import numpy as np
#import matplotlib.pyplot as plt
from infill.refinecriteria import looCV, HessianFit, TEAD
from infill.aniso_criteria import AnisotropicRefine
from infill.taylor_criteria import TaylorRefine, TaylorExploreRefine
from infill.hess_criteria import HessianRefine, POUSSA
from infill.loocv_criteria import POUSFCVT, SFCVT
from infill.aniso_transform import AnisotropicTransform
from infill.getxnew import getxnew, adaptivesampling
from optimization.defaults import DefaultOptOptions
from utils.sutils import divide_cases, convert_to_smt_grads
from utils.error import rmse, meane
from functions.problem_picker import GetProblem

from smt.surrogate_models import KPLS, GEKPLS, KRG
from surrogate.gek_1d import GEK1D
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
    if rank == 0:
        if not os.path.isdir(f"{path}/{title}"):
            os.mkdir(f"{path}/{title}")
        shutil.copy(f"./{sys.argv[1]}.py", f"{path}/{title}/settings.py")



elif len(sys.argv) > 2:
    fresh = False
    
    args = sys.argv[1:]
    fulltitle = args[0]
    tsplit = fulltitle.split('/')
    if len(tsplit) == 1:
        path = "."
    else:
        path = '/'.join(tsplit[:-1])
    title = tsplit[-1]
    
    # if rank == 0:
    #     shutil.copy(f"{path}/{title}/settings.py", "./results_settings.py")
    #sys.path.append(title)

    # from results_settings import *
    # import results_settings as set
    set = importlib.import_module(f"{path}.{title}.settings")
    if(path == None):
        path = "."
    
    
    ntr = args[1]

    

    

    #need to load the current model
    with open(f'{path}/{title}/modelf.pickle', 'rb') as f:
        model0lists = pickle.load(f)

    

else:
    # All variables not initialized come from this import
    import results_settings as set
    # Generate results folder and list of inputs
    title = f"{set.header}_{set.prob}_{set.dim}D"
    if(set.path == None):
        path = "."
    else:
        path = set.path
    if rank == 0:
        if not os.path.isdir(f"{path}/{title}"):
            os.mkdir(f"{path}/{title}")
        shutil.copy("./results_settings.py", f"{path}/{title}/settings.py")

Nruns = size*set.runs_per_proc
# Fan out parallel cases
cases = divide_cases(Nruns, size)

# override the added points with what's on the command line
if(not fresh):
    ntr = int(args[1])
    nt0 = model0lists[0][0].training_points[None][0][0].shape[0]

    model0 = [None]*Nruns
    # put model0s in order
    for s in range(size):
        for n in range(len(cases[s])):
            model0[cases[s][n]] = model0lists[s][n]
else:
    ntr = set.ntr
    nt0 = set.nt0

# Problem Settings
ud = set.perturb
trueFunc = GetProblem(set.prob, set.dim, use_design=ud)
xlimits = trueFunc.xlimits
sampling = LHS(xlimits=xlimits, criterion='m')
if(set.rtype == 'anisotransform'):
    sequencer = []
    for n in range(Nruns):
        sequencer.append(qmc.Halton(d=set.dim))



# Error
xtest = None 
ftest = None
testdata = None
if(set.dim > 3):
    intervals = np.arange(0, ntr, set.dim)
    intervals = np.append(intervals, ntr-1)
else:
    intervals = np.arange(0, ntr)

if(set.prob != 'shock'):
    if rank == 0:
        xtest = sampling(set.Nerr)
        ftest = trueFunc(xtest)

        testdata = [xtest, ftest, intervals]

    xtest = comm.bcast(xtest, root=0)
    ftest = comm.bcast(ftest, root=0)
    testdata = comm.bcast(testdata, root=0)
# Adaptive Sampling Conditions
options = DefaultOptOptions
options["local"] = set.local
options["localswitch"] = set.localswitch
options["errorcheck"] = testdata
options["multistart"] = set.mstarttype
options["lmethod"] = set.opt
try:
    options["method"] = set.gopt
except:
    pass

# Print Conditions
if rank == 0:
    print("\n")
    print("\n")
    print("Surrogate Type       : ", set.stype)
    print("Refinement Type      : ", set.rtype)
    print("Refinement Multistart: ", set.multistart)
    print("Correlation Function : ", set.corr)
    print("Regression Function  : ", set.poly)
    print("GEK Extra Points     : ", set.extra)
    print("Problem              : ", set.prob)
    print("Problem Dimension    : ", set.dim)
    print("Initial Sample Size  : ", nt0)
    print("Refined Points Size  : ", ntr)
    print("Total Points         : ", nt0+ntr)
    print("Points Per Iteration : ", set.batch)
    print("RMSE Size            : ", set.Nerr)
    print("\n")




    print("Computing Initial Designs for Adaptive Sampling ...")

    # Adaptive Sampling Initial Design
xtrain0 = []
ftrain0 = []
gtrain0 = []

if fresh:
    # if(rtype == 'anisotransform'):
    #     for n in range(Nruns):
    #         sample = sequencer[n].random(nt0)
    #         sample = qmc.scale(sample, xlimits[:,0], xlimits[:,1])
    #         xtrain0.append(sample)
    #         ftrain0.append(trueFunc(xtrain0[n]))
    #         gtrain0.append(np.zeros([nt0,dim]))
    #         for i in range(dim):
    #             gtrain0[n][:,i:i+1] = trueFunc(xtrain0[n],i)
    # else:
    #for n in range(Nruns):
    co = 0
    for n in cases[rank]:
        xtrain0.append(sampling(nt0))
        ftrain0.append(trueFunc(xtrain0[co]))
        gtrain0.append(convert_to_smt_grads(trueFunc, xtrain0[co]))
        # gtrain0.append(np.zeros([nt0,dim]))
        # for i in range(dim):
        #     gtrain0[co][:,i:i+1] = trueFunc(xtrain0[co],i)

else:
    co = 0
    for n in cases[rank]:
        xtrain0.append(model0[n].training_points[None][0][0])
        ftrain0.append(model0[n].training_points[None][0][1])
        gtrain0.append(convert_to_smt_grads(trueFunc, xtrain0[co]))
        # gtrain0.append(np.zeros([nt0,dim]))
        # for i in range(dim):
        #     gtrain0[co][:,i:i+1] = model0[n].training_points[None][i+1][1]
        co += 1

xtrain0lists = comm.allgather(xtrain0)
ftrain0lists = comm.allgather(ftrain0)
gtrain0lists = comm.allgather(gtrain0)

#need to put them back in order

xtrain0 = [None]*Nruns
ftrain0 = [None]*Nruns
gtrain0 = [None]*Nruns

for s in range(size):
    for n in range(len(cases[s])):
        xtrain0[cases[s][n]] = xtrain0lists[s][n]
        ftrain0[cases[s][n]] = ftrain0lists[s][n]
        gtrain0[cases[s][n]] = gtrain0lists[s][n]

# xtrain0 = comm.bcast(xtrain0, root=0)
# ftrain0 = comm.bcast(ftrain0, root=0)
# gtrain0 = comm.bcast(gtrain0, root=0)

idx = np.round(np.linspace(0, len(intervals)-1, set.LHS_batch+1)).astype(int)


intervalsk = intervals[idx]
samplehistK = intervalsk + nt0*np.ones(len(intervalsk), dtype=int)
samplehistK[-1] += 1

if rank == 0:
    print("Computing Non-Adaptive Designs ...")

if(not set.skip_LHS):
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
                xtrainK[n].append(sampling(samplehistK[m]))
                ftrainK[n].append(trueFunc(xtrainK[n][m]))
                gtrainK[n].append(convert_to_smt_grads(trueFunc, xtrainK[n][m]))
                # gtrainK[n].append(np.zeros([samplehistK[m],dim]))
                # for i in range(dim):
                #     gtrainK[n][m][:,i:i+1] = trueFunc(xtrainK[n][m],i)

    xtrainK = comm.bcast(xtrainK, root=0)
    ftrainK = comm.bcast(ftrainK, root=0)
    gtrainK = comm.bcast(gtrainK, root=0)
if rank == 0:
    print("Training Initial Surrogate ...")

# Initial Design Surrogate
if fresh:
    if(set.stype == "gekpls"):
        if(set.dim > 1):
            modelbase = GEKPLS(xlimits=xlimits)
            # modelbase.options.update({"hyper_opt":'TNC'})
            modelbase.options.update({"n_comp":set.dim})
            modelbase.options.update({"extra_points":set.extra})
            modelbase.options.update({"delta_x":set.delta_x})
            if(set.dim > 2):
                modelbase.options.update({"zero_out_y":True})
        else: # to get gek runs to work in 1D
            modelbase = GEK1D(xlimits=xlimits)
            #modelgek.options.update({"hyper_opt":"TNC"})
        modelbase.options.update({"theta0":set.t0})
        modelbase.options.update({"theta_bounds":set.tb})
        modelbase.options.update({"corr":set.corr})
        modelbase.options.update({"poly":set.poly})
        modelbase.options.update({"n_start":5})

    elif(set.stype == "dgek"):
        modelbase = DGEK(xlimits=xlimits)
        # modelbase.options.update({"hyper_opt":'TNC'})
        modelbase.options.update({"theta0":set.t0})
        modelbase.options.update({"theta_bounds":set.tb})
        modelbase.options.update({"corr":set.corr})
        modelbase.options.update({"poly":set.poly})
        modelbase.options.update({"n_start":5})
    elif(set.stype == "pou"):
        modelbase = POUSurrogate()
        modelbase.options.update({"rho":set.rho})
    elif(set.stype == "pouhess"):
        modelbase = POUHessian(bounds=xlimits, rscale=set.rscale)
        modelbase.options.update({"rho":set.rho})
        modelbase.options.update({"neval":set.neval})
    elif(set.stype == "kpls"):
        modelbase = KPLS()
        # modelbase.options.update({"hyper_opt":'TNC'})
        modelbase.options.update({"n_comp":set.dim})
        modelbase.options.update({"corr":set.corr})
        modelbase.options.update({"poly":set.poly})
        modelbase.options.update({"n_start":5})
    else:
        modelbase = KRG()
        # modelbase.options.update({"hyper_opt":'TNC'})
        modelbase.options.update({"corr":set.corr})
        modelbase.options.update({"poly":set.poly})
        modelbase.options.update({"n_start":5})
    modelbase.options.update({"print_global":False})


else:
    modelbase = copy.deepcopy(model0[0])

model0 = []
co = 0
for n in cases[rank]: #range(Nruns):
    model0.append(copy.deepcopy(modelbase))
    model0[co].set_training_values(xtrain0[n], ftrain0[n])
    convert_to_smt_grads(model0[co], xtrain0[n], gtrain0[n])
    # if(isinstance(model0[co], GEKPLS) or isinstance(model0[co], POUSurrogate) or isinstance(model0[co], DGEK) or isinstance(model0[co], POUHessian)):
    #     for i in range(dim):
    #         model0[co].set_training_derivatives(xtrain0[n], gtrain0[n][:,i:i+1], i)
    model0[co].train()
    co += 1
    # modelbase.options.update({"print_training":True})
    # modelbase.options.update({"print_prediction":True})
    # modelbase.options.update({"print_problem":True})
    # modelbase.options.update({"print_solver":True})







# Initial Model Error
err0rms = []
err0mean = []
if(set.prob != 'shock'):
    if rank == 0:
        print("Computing Initial Surrogate Error ...")

    co = 0
    for n in cases[rank]:
        err0rms.append(rmse(model0[co], trueFunc, N=set.Nerr, xdata=xtest, fdata=ftest))
        err0mean.append(meane(model0[co], trueFunc, N=set.Nerr, xdata=xtest, fdata=ftest))
        co += 1


if rank == 0:
    print("Computing Final Non-Adaptive Surrogate Error ...")

if(not set.skip_LHS):
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
            try:
                modelK.set_training_values(xtrainK[n][m], ftrainK[n][m])
                convert_to_smt_grads(modelK, xtrainK[n][m], gtrainK[n][m])
                # if(isinstance(modelbase, GEKPLS) or isinstance(modelbase, POUSurrogate) or isinstance(model0[co], DGEK)):
                #     for i in range(dim):
                #         modelK.set_training_derivatives(xtrainK[n][m], gtrainK[n][m][:,i:i+1], i)
                modelK.train()
                errkrms[co].append(rmse(modelK, trueFunc, N=set.Nerr, xdata=xtest, fdata=ftest))
                errkmean[co].append(meane(modelK, trueFunc, N=set.Nerr, xdata=xtest, fdata=ftest))
            except:
                print("LHS runs started failing on this processor!")
                continue
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
    if(set.rtype == "aniso"):
        RC0.append(AnisotropicRefine(model0[co], gtrain0[n], xlimits, rscale=set.rscale, nscale=set.nscale, improve=set.pperb, neval=set.neval, hessian=set.hess, interp=set.interp, bpen=set.bpen, objective=set.obj, multistart=set.multistart) )
    elif(set.rtype == "anisotransform"):
        RC0.append(AnisotropicTransform(model0[co], sequencer[n], gtrain0[n], improve=set.pperb, nmatch=set.nmatch, neval=set.neval, hessian=set.hess, interp=set.interp))
    elif(set.rtype == "tead"):
        RC0.append(TEAD(model0[co], gtrain0[n], xlimits, gradexact=True))
    elif(set.rtype == "taylor"):
        RC0.append(TaylorRefine(model0[co], gtrain0[n], xlimits, volume_weight=set.perturb, rscale=set.rscale, improve=set.pperb, multistart=set.multistart) )
    elif(set.rtype == "taylorexp"):
        RC0.append(TaylorExploreRefine(model0[co], gtrain0[n], xlimits, rscale=set.rscale, improve=set.pperb, objective=set.obj, multistart=set.multistart) ) 
    elif(set.rtype == "hess"):
        RC0.append(HessianRefine(model0[co], gtrain0[n], xlimits, neval=set.neval, rscale=set.rscale, improve=set.pperb, multistart=set.multistart, print_rc_plots=set.rc_print) )
    elif(set.rtype == "poussa"):
        RC0.append(POUSSA(model0[co], gtrain0[n], xlimits, improve=set.pperb, multistart=set.multistart, print_rc_plots=set.rc_print))
    elif(set.rtype == "pousfcvt"):
        RC0.append(POUSFCVT(model0[co], gtrain0[n], xlimits, improve=set.pperb, multistart=set.multistart, print_rc_plots=set.rc_print))
    elif(set.rtype == "sfcvt"):
        RC0.append(SFCVT(model0[co], gtrain0[n], xlimits,  print_rc_plots=set.rc_print)) # improve=pperb, multistart=multistart, not implemented
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

# import pdb; pdb.set_trace()
# #try:
# if("comp_hist" in ssettings):
#     iters = len(ef)
#     itersk = LHS_batch
#     histc = []
#     ind_alt = np.linspace(0, iters-1, itersk, dtype=int)
#     for n in range(co):
#         dummy = []
#         for p in ind_alt:
#             dummy.append(hist[n][p])
#         histc.append(dummy)
#     hist = histc
#     if rank == 0:
#         print("hello from comp hist")


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

    if(fresh):
        affix = ""
    else:
        affix = f"_{nt0}"

    # Adaptive Data
    with open(f'{path}/{title}/modelf.pickle', 'wb') as f:
        pickle.dump(modelf, f)

    with open(f'{path}/{title}/err0rms{affix}.pickle', 'wb') as f:
        pickle.dump(err0rms, f)

    with open(f'{path}/{title}/err0mean{affix}.pickle', 'wb') as f:
        pickle.dump(err0mean, f)

    with open(f'{path}/{title}/hist{affix}.pickle', 'wb') as f:
        pickle.dump(hist, f)

    with open(f'{path}/{title}/errhrms{affix}.pickle', 'wb') as f:
        pickle.dump(errhrms, f)

    with open(f'{path}/{title}/errhmean{affix}.pickle', 'wb') as f:
        pickle.dump(errhmean, f)

    if(set.dim > 3):
        with open(f'{path}/{title}/intervals{affix}.pickle', 'wb') as f:
            pickle.dump(intervals, f)

# import pdb; pdb.set_trace()