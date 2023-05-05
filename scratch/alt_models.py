import sys, os
import copy
import pickle
from mpi4py import MPI

import numpy as np
import math
import importlib
import matplotlib.pyplot as plt
from infill.refinecriteria import looCV, HessianFit
from infill.aniso_criteria import AnisotropicRefine
from infill.getxnew import getxnew, adaptivesampling
from optimization.defaults import DefaultOptOptions
from utils.sutils import divide_cases
from utils.error import rmse, meane, full_error

from functions.problem_picker import GetProblem
from smt.surrogate_models import KPLS, GEKPLS, KRG
#from smt.surrogate_models.rbf import RBF
from surrogate.pougrad import POUSurrogate, POUHessian
from surrogate.direct_gek import DGEK
import matplotlib as mpl
import matplotlib.ticker as mticker
from smt.sampling_methods import LHS

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

"""
generate desired alternative surrogate models from
those used in adaptive sampling, using the data obtained from adaptive
sampling, and save error to file
"""

# Give directory with desired results as argument
title = sys.argv[1]
alt_model = ['KRG','GEK']#sys.argv[2]
#impath = title.rsplit('.')
sys.path.append(title)
setmod = importlib.import_module(f'settings')
ssettings = setmod.__dict__

if not os.path.isdir(title):
    os.mkdir(title)

prob = title.split("_")[-2]
plt.rcParams['font.size'] = '18'
plt.rc('legend',fontsize=14)
# import pdb; pdb.set_trace()

# Adaptive Data
sys.path.append('../surrogate')
with open(f'{title}/modelf.pickle', 'rb') as f:
    modelf = pickle.load(f)
with open(f'{title}/err0rms.pickle', 'rb') as f:
    err0rms = pickle.load(f)
with open(f'{title}/err0mean.pickle', 'rb') as f:
    err0mean = pickle.load(f)
with open(f'{title}/hist.pickle', 'rb') as f:
    hist = pickle.load(f)
with open(f'{title}/errhrms.pickle', 'rb') as f:
    errhrms = pickle.load(f)
with open(f'{title}/errhmean.pickle', 'rb') as f:
    errhmean = pickle.load(f)
# import pdb; pdb.set_trace()
# LHS Data
with open(f'{title}/xk.pickle', 'rb') as f:
    xtrainK = pickle.load(f)
with open(f'{title}/fk.pickle', 'rb') as f:
    ftrainK = pickle.load(f)
with open(f'{title}/gk.pickle', 'rb') as f:
    gtrainK = pickle.load(f)
with open(f'{title}/errkrms.pickle', 'rb') as f:
    errkrms = pickle.load(f)
with open(f'{title}/errkmean.pickle', 'rb') as f:
    errkmean = pickle.load(f)



# Concatenate lists
xk = []
fk = []
gk = []
ekr = []
ekm = []
mf = []
e0r = []
e0m = []
hi = []
ehr = []
ehm = []
nprocs = len(modelf)
for i in range(nprocs):
    xk = xk + xtrainK[i][:]
    fk = fk + ftrainK[i][:]
    gk = gk + gtrainK[i][:]
    ekr = ekr + errkrms[i][:]
    ekm = ekm + errkmean[i][:]
    mf = mf + modelf[i][:]
    e0r = e0r + err0rms[i]
    e0m = e0m + err0mean[i]
    hi = hi + hist[i][:]
    ehr = ehr + errhrms[i][:]
    ehm = ehm + errhmean[i][:]

nruns = len(mf)
nperr = int(nruns/size)
dim = xk[0].shape[1]

# Problem Settings
trueFunc = GetProblem(prob, dim)
xlimits = trueFunc.xlimits

# Get the original testing data
testdata = None
Nerr = 5000*dim
sampling = LHS(xlimits=xlimits, criterion='m')

# Error
xtest = None 
ftest = None
testdata = None

if rank == 0:
    xtest = sampling(Nerr)
    ftest = trueFunc(xtest)

xtest = comm.bcast(xtest, root=0)
ftest = comm.bcast(ftest, root=0)

# Generate Alternative Surrogate
if(dim > 1):
    modelbase2 = GEKPLS(xlimits=xlimits)
    # modelbase.options.update({"hyper_opt":'TNC'})
    modelbase2.options.update({"theta0":ssettings["t0"]})
    modelbase2.options.update({"theta_bounds":ssettings["tb"]})
    modelbase2.options.update({"n_comp":dim})
    modelbase2.options.update({"extra_points":ssettings["extra"]})
    modelbase2.options.update({"corr":"squar_exp"})#ssettings["corr"]})
    modelbase2.options.update({"poly":ssettings["poly"]})
    modelbase2.options.update({"n_start":5})
    modelbase2.options.update({"delta_x":ssettings["delta_x"]})
    if(dim > 2):
        modelbase2.options.update({"zero_out_y":True})
else:
    modelbase2 = KRG()
    #modelgek.options.update({"hyper_opt":"TNC"})
    modelbase2.options.update({"theta0":ssettings["t0"]})
    modelbase2.options.update({"theta_bounds":ssettings["tb"]})
    modelbase2.options.update({"corr":"squar_exp"})#ssettings["corr"]})
    modelbase2.options.update({"poly":ssettings["poly"]})
    modelbase2.options.update({"n_start":5})
    modelbase2.options.update({"print_prediction":False})


# elif(alt_model == "dgek"):
#     modelbase = DGEK(xlimits=xlimits)
#     # modelbase.options.update({"hyper_opt":'TNC'})
#     modelbase.options.update({"corr":ssettings["corr"]})
#     modelbase.options.update({"poly":ssettings["poly"]})
#     modelbase.options.update({"n_start":5})
#     modelbase.options.update({"theta0":ssettings["t0"]})
#     modelbase.options.update({"theta_bounds":ssettings["tb"]})
# elif(alt_model == "pou"):
#     modelbase = POUSurrogate()
#     modelbase.options.update({"rho":ssettings["rho"]})
# elif(alt_model == "pouhess"):
#     modelbase = POUHessian(bounds=xlimits)
#     modelbase.options.update({"rho":ssettings["rho"]})
#     modelbase.options.update({"neval":ssettings["neval"]})
# elif(alt_model == "kpls"):
#     modelbase = KPLS()
#     # modelbase.options.update({"hyper_opt":'TNC'})
#     modelbase.options.update({"n_comp":dim})
#     modelbase.options.update({"corr":ssettings["corr"]})
#     modelbase.options.update({"poly":ssettings["poly"]})
#     modelbase.options.update({"n_start":5})
# elif(alt_model == "kriging"):
modelbase1 = KRG()
# modelbase.options.update({"hyper_opt":'TNC'})
modelbase1.options.update({"theta0":ssettings["t0"]})
modelbase1.options.update({"theta_bounds":ssettings["tb"]})
modelbase1.options.update({"corr":"squar_exp"})#ssettings["corr"]})
modelbase1.options.update({"poly":ssettings["poly"]})
modelbase1.options.update({"n_start":5})
# else:
#     raise ValueError("Given alternative model not valid.")
modelbase1.options.update({"print_global":True})
modelbase2.options.update({"print_global":True})
# modelbase.options.update({"print_training":True})
# modelbase.options.update({"print_prediction":True})
# modelbase.options.update({"print_problem":True})
# modelbase.options.update({"print_solver":True})








for i in range(nruns):
    ehr[i] = [e0r[i]] + ehr[i] #[errf] #errh
    ehm[i] = [e0m[i]] + ehm[i]


# Plot Error History
iters = len(ehr[0])
if(dim > 3):
    with open(f'{title}/intervals.pickle', 'rb') as f:
        intervals = pickle.load(f)
    if(intervals[0] != 0):
        intervals = np.insert(intervals, 0, 0)
else:
    intervals = np.arange(iters)

itersk = len(ekr[0])

samplehist = np.zeros(intervals.shape[0], dtype=int)


#for i in range(iters-1):
#    samplehist[i] = hi[0][i][0][0].shape[0] #training_points
samplehist[0] = hi[0][0][0][0].shape[0] #training_points 
for i in range(1, iters):
    samplehist[i] = samplehist[i-1] + (intervals[i] - intervals[i-1])
samplehist[-1] = mf[0].training_points[None][0][0].shape[0]


idx = np.round(np.linspace(0, len(intervals)-1, itersk)).astype(int)
intervalsk = intervals[idx]
samplehistk = intervalsk + samplehist[0]*np.ones(len(intervalsk), dtype=int)
samplehistk[-1] += 1


# Grab data from the lhs and adaptive sample sets
#ind_alt = np.linspace(0, iters, itersk, dtype=int)
ind_alt = idx
# if("comp_hist" in ssettings):
#     ind_alt = np.arange(ssettings["LHS_batch"])


xa = []
fa = []
ga = []
xh = []
fh = []
gh = []
for k in range(nruns):
    xa.append([])
    fa.append([])
    ga.append([])
    xh.append([])
    fh.append([])
    gh.append([])
    for i in range(itersk-1):
        xa[k].append(hi[k][ind_alt[i]][0][0])
        fa[k].append(hi[k][ind_alt[i]][0][1])
        ga[k].append(np.zeros_like(hi[k][ind_alt[i]][0][0]))
        for j in range(dim):
            ga[k][i][:,j:j+1] = hi[k][ind_alt[i]][j+1][1]
        xh[k].append(xk[i+k*itersk])
        fh[k].append(fk[i+k*itersk])
        gh[k].append(gk[i+k*itersk])

    xa[k].append(mf[k].training_points[None][0][0])
    fa[k].append(mf[k].training_points[None][0][1])
    ga[k].append(np.zeros_like(mf[k].training_points[None][0][0]))
    for j in range(dim):
        ga[k][-1][:,j:j+1] = mf[k].training_points[None][j+1][1]
    xh[k].append(xk[itersk-1+k*itersk])
    fh[k].append(fk[itersk-1+k*itersk])
    gh[k].append(gk[itersk-1+k*itersk])
    
#import pdb; pdb.set_trace()

# Train alternative surrogates
ma1 = [[] for _ in range(nperr)]
ma2 = [[] for _ in range(nperr)]
mh1 = [[] for _ in range(nperr)]
mh2 = [[] for _ in range(nperr)]
ear1 = np.zeros([nperr, itersk])
eam1 = np.zeros([nperr, itersk])
eas1 = np.zeros([nperr, itersk])
ear2 = np.zeros([nperr, itersk])
eam2 = np.zeros([nperr, itersk])
eas2 = np.zeros([nperr, itersk])
ehr1 = np.zeros([nperr, itersk])
ehm1 = np.zeros([nperr, itersk])
ehs1 = np.zeros([nperr, itersk])
ehr2 = np.zeros([nperr, itersk])
ehm2 = np.zeros([nperr, itersk])
ehs2 = np.zeros([nperr, itersk])

slim = 250

slima1 = slim*10
slima2 = slim*10
slimh1 = slim*10
slimh2 = slim*10




for k in range(nperr):
    ind = k + rank*nperr
    for i in range(itersk):

        ma1[k].append(copy.deepcopy(modelbase1))
        ma1[k][i].set_training_values(xa[ind][i], fa[ind][i])
        # try:
        if(xa[ind][i].shape[0]<slima1):
            ma1[k][i].train()
            ear1[k][i], eam1[k][i], eas1[k][i] = full_error(ma1[k][i], trueFunc, N=Nerr, xdata=xtest, fdata=ftest)
        else:
            print(f'{i}, {rank}, oops')
            ear1[k][i] = np.nan
            eam1[k][i] = np.nan
            eas1[k][i] = np.nan


        ma2[k].append(copy.deepcopy(modelbase2))
        if(dim > 1):
            ma2[k][i].set_training_values(xa[ind][i], fa[ind][i])
            for j in range(dim):
                ma2[k][i].set_training_derivatives(xa[ind][i], ga[ind][i][:,j:j+1], j)
        else:
            dx = 1e-4
            nex = xa[ind][i].shape[0]
            xaug = np.zeros([nex, 1])
            faug = np.zeros([nex, 1])
            for l in range(nex):
                xaug[l] = xa[ind][i][l] + dx
                faug[l] = fa[ind][i][l] + dx*ga[ind][i][l]
            xtot = np.append(xa[ind][i], xaug, axis=0)
            ftot = np.append(fa[ind][i], faug, axis=0)
            ma2[k][i].set_training_values(xtot, ftot)
        # try:
        if(xa[ind][i].shape[0]<slima2):
            ma2[k][i].train()
            ear2[k][i], eam2[k][i], eas2[k][i] = full_error(ma2[k][i], trueFunc, N=Nerr, xdata=xtest, fdata=ftest)
        else:
            print(f'{i}, {rank}, oops')
            ear2[k][i] = np.nan
            eam2[k][i] = np.nan
            eas2[k][i] = np.nan


        mh1[k].append(copy.deepcopy(modelbase1))
        mh1[k][i].set_training_values(xh[ind][i], fh[ind][i])
        # try: 
        if(xa[ind][i].shape[0]<slimh1 or dim < 8):
            mh1[k][i].train()
            ehr1[k][i], ehm1[k][i], ehs1[k][i] = full_error(mh1[k][i], trueFunc, N=Nerr, xdata=xtest, fdata=ftest)
        else:
            print(f'{i}, {rank}, oops')
            ehr1[k][i] = np.nan
            ehm1[k][i] = np.nan
            ehs1[k][i] = np.nan

        mh2[k].append(copy.deepcopy(modelbase2))
        if(dim > 1):
            mh2[k][i].set_training_values(xh[ind][i], fh[ind][i])
            for j in range(dim):
                mh2[k][i].set_training_derivatives(xh[ind][i], gh[ind][i][:,j:j+1], j)
        else:
            dx = 1e-4
            nex = xh[ind][i].shape[0]
            xaug = np.zeros([nex, 1])
            faug = np.zeros([nex, 1])
            for l in range(nex):
                xaug[l] = xh[ind][i][l] + dx
                faug[l] = fh[ind][i][l] + dx*gh[ind][i][l]
            xtot = np.append(xh[ind][i], xaug, axis=0)
            ftot = np.append(fh[ind][i], faug, axis=0)
            mh2[k][i].set_training_values(xtot, ftot)
        if(xa[ind][i].shape[0]<slimh2 or dim < 8):
            mh2[k][i].train()
            ehr2[k][i], ehm2[k][i], ehs2[k][i] = full_error(mh2[k][i], trueFunc, N=Nerr, xdata=xtest, fdata=ftest)
        else:
            print(f'{i}, {rank}, oops')
            ehr2[k][i] = np.nan
            ehm2[k][i] = np.nan
            ehs2[k][i] = np.nan

        print(f'{i}, {rank}')

print("HUGE SUCCESS")

#ma1 = comm.allgather(ma1)
ear1 = comm.allgather(ear1)
eam1 = comm.allgather(eam1)
eas1 = comm.allgather(eas1)
ear1 = np.concatenate(ear1[:], axis=0)
eam1 = np.concatenate(eam1[:], axis=0)
eas1 = np.concatenate(eas1[:], axis=0)

#ma2 = comm.allgather(ma2)
ear2 = comm.allgather(ear2)
eam2 = comm.allgather(eam2)
eas2 = comm.allgather(eas2)
ear2 = np.concatenate(ear2[:], axis=0)
eam2 = np.concatenate(eam2[:], axis=0)
eas2 = np.concatenate(eas2[:], axis=0)

#mh1 = comm.allgather(mh1)
ehr1 = comm.allgather(ehr1)
ehm1 = comm.allgather(ehm1)
ehs1 = comm.allgather(ehs1)
ehr1 = np.concatenate(ehr1[:], axis=0)
ehm1 = np.concatenate(ehm1[:], axis=0)
ehs1 = np.concatenate(ehs1[:], axis=0)

#mh2 = comm.allgather(mh2)
ehr2 = comm.allgather(ehr2)
ehm2 = comm.allgather(ehm2)
ehs2 = comm.allgather(ehs2)
ehr2 = np.concatenate(ehr2[:], axis=0)
ehm2 = np.concatenate(ehm2[:], axis=0)
ehs2 = np.concatenate(ehs2[:], axis=0)

# Average out runs
ehrm = np.zeros(iters)
ehmm = np.zeros(iters)
ehsm = np.zeros(iters) 
ekrm = np.zeros(itersk)
ekmm = np.zeros(itersk)
eksm = np.zeros(itersk)

ehrs = np.zeros(iters)
ehms = np.zeros(iters)
ehss = np.zeros(iters) 
ekrs = np.zeros(itersk)
ekms = np.zeros(itersk)
ekss = np.zeros(itersk)

ehrssq = np.zeros(iters)
ehmssq = np.zeros(iters)
ehsssq = np.zeros(iters) 
ekrssq = np.zeros(itersk)
ekmssq = np.zeros(itersk)
eksssq = np.zeros(itersk)

eamm1 = np.zeros(itersk)
earm1 = np.zeros(itersk)
easm1 = np.zeros(itersk)
eamm2 = np.zeros(itersk)
earm2 = np.zeros(itersk)
easm2 = np.zeros(itersk)
ehmm1 = np.zeros(itersk)
ehrm1 = np.zeros(itersk)
ehsm1 = np.zeros(itersk)
ehmm2 = np.zeros(itersk)
ehrm2 = np.zeros(itersk)
ehsm2 = np.zeros(itersk)

eams1 = np.zeros(itersk)
ears1 = np.zeros(itersk)
eass1 = np.zeros(itersk)
eams2 = np.zeros(itersk)
ears2 = np.zeros(itersk)
eass2 = np.zeros(itersk)
ehms1 = np.zeros(itersk)
ehrs1 = np.zeros(itersk)
ehss1 = np.zeros(itersk)
ehms2 = np.zeros(itersk)
ehrs2 = np.zeros(itersk)
ehss2 = np.zeros(itersk)

eams1sq = np.zeros(itersk)
ears1sq = np.zeros(itersk)
eass1sq = np.zeros(itersk)
eams2sq = np.zeros(itersk)
ears2sq = np.zeros(itersk)
eass2sq = np.zeros(itersk)
ehms1sq = np.zeros(itersk)
ehrs1sq = np.zeros(itersk)
ehss1sq = np.zeros(itersk)
ehms2sq = np.zeros(itersk)
ehrs2sq = np.zeros(itersk)
ehss2sq = np.zeros(itersk)



for i in range(nruns):
    ehrm += np.array(ehr[i]).T[0]/nruns
    ehmm += np.array(ehm[i]).T[0][0]/nruns
    ehsm += np.array(ehm[i]).T[0][1]/nruns
    ekrm += np.array(ekr[i]).T[0]/nruns
    ekmm += np.array(ekm[i]).T[0][0]/nruns
    eksm += np.array(ekm[i]).T[0][1]/nruns
    
    ehrssq += np.square(np.array(ehr[i]).T[0])/nruns
    ehmssq += np.square(np.array(ehm[i]).T[0][0])/nruns
    ehsssq += np.square(np.array(ehm[i]).T[0][1])/nruns
    ekrssq += np.square(np.array(ekr[i]).T[0])/nruns
    ekmssq += np.square(np.array(ekm[i]).T[0][0])/nruns
    eksssq += np.square(np.array(ekm[i]).T[0][1])/nruns

    earm1 += np.array(ear1[i]).T/nruns
    eamm1 += np.array(eam1[i]).T/nruns
    easm1 += np.array(eas1[i]).T/nruns
    earm2 += np.array(ear2[i]).T/nruns
    eamm2 += np.array(eam2[i]).T/nruns
    easm2 += np.array(eas2[i]).T/nruns

    ehrm1 += np.array(ehr1[i]).T/nruns
    ehmm1 += np.array(ehm1[i]).T/nruns
    ehsm1 += np.array(ehs1[i]).T/nruns
    ehrm2 += np.array(ehr2[i]).T/nruns
    ehmm2 += np.array(ehm2[i]).T/nruns
    ehsm2 += np.array(ehs2[i]).T/nruns

    ears1sq += np.square(np.array(ear1[i]).T)/nruns
    eams1sq += np.square(np.array(eam1[i]).T)/nruns
    eass1sq += np.square(np.array(eas1[i]).T)/nruns
    ears2sq += np.square(np.array(ear2[i]).T)/nruns
    eams2sq += np.square(np.array(eam2[i]).T)/nruns
    eass2sq += np.square(np.array(eas2[i]).T)/nruns

    ehrs1sq += np.square(np.array(ehr1[i]).T)/nruns
    ehms1sq += np.square(np.array(ehm1[i]).T)/nruns
    ehss1sq += np.square(np.array(ehs1[i]).T)/nruns
    ehrs2sq += np.square(np.array(ehr2[i]).T)/nruns
    ehms2sq += np.square(np.array(ehm2[i]).T)/nruns
    ehss2sq += np.square(np.array(ehs2[i]).T)/nruns

ehrs = np.sqrt(ehrssq - ehrm**2)
ehms = np.sqrt(ehmssq - ehmm**2)
ehss = np.sqrt(ehsssq - ehsm**2)
ekrs = np.sqrt(ekrssq - ekrm**2)
ekms = np.sqrt(ekmssq - ekmm**2)
ekss = np.sqrt(eksssq - eksm**2)

ears1 = np.sqrt(ears1sq - earm1**2)
eams1 = np.sqrt(eams1sq - eamm1**2)
eass1 = np.sqrt(eass1sq - easm1**2)
ears2 = np.sqrt(ears2sq - earm2**2)
eams2 = np.sqrt(eams2sq - eamm2**2)
eass2 = np.sqrt(eass2sq - easm2**2)

ehrs1 = np.sqrt(ehrs1sq - ehrm1**2)
ehms1 = np.sqrt(ehms1sq - ehmm1**2)
ehss1 = np.sqrt(ehss1sq - ehsm1**2)
ehrs2 = np.sqrt(ehrs2sq - ehrm2**2)
ehms2 = np.sqrt(ehms2sq - ehmm2**2)
ehss2 = np.sqrt(ehss2sq - ehsm2**2)

if rank == 0:
    print("\n")
    print("Saving Results")

    # Adaptive Data
    with open(f'{title}/samplehist.pickle', 'wb') as f:
        pickle.dump(samplehist, f)
    with open(f'{title}/samplehistk.pickle', 'wb') as f:
        pickle.dump(samplehistk, f)

    meanspou = [ehrm, ehmm, ehsm, ekrm, ekmm, eksm]
    meanskrg = [earm1, eamm1, easm1, ehrm1, ehmm1, ehsm1]
    meansgek = [earm2, eamm2, easm2, ehrm2, ehmm2, ehsm2]

    stdvspou = [ehrs, ehms, ehss, ekrs, ekms, ekss]
    stdvskrg = [ears1, eams1, eass1, ehrs1, ehms1, ehss1]
    stdvsgek = [ears2, eams2, eass2, ehrs2, ehms2, ehss2]




    with open(f'{title}/errhrmsgek.pickle', 'wb') as f:
        pickle.dump(ear2, f)
    with open(f'{title}/errhrmskrg.pickle', 'wb') as f:
        pickle.dump(ear1, f)
    with open(f'{title}/errkrmsgek.pickle', 'wb') as f:
        pickle.dump(ehr2, f)
    with open(f'{title}/errkrmskrg.pickle', 'wb') as f:
        pickle.dump(ehr1, f)

    with open(f'{title}/meanspou.pickle', 'wb') as f:
        pickle.dump(meanspou, f)
    with open(f'{title}/meanskrg.pickle', 'wb') as f:
        pickle.dump(meanskrg, f)
    with open(f'{title}/meansgek.pickle', 'wb') as f:
        pickle.dump(meansgek, f)

    with open(f'{title}/stdvspou.pickle', 'wb') as f:
        pickle.dump(stdvspou, f)
    with open(f'{title}/stdvskrg.pickle', 'wb') as f:
        pickle.dump(stdvskrg, f)
    with open(f'{title}/stdvsgek.pickle', 'wb') as f:
        pickle.dump(stdvsgek, f)


