import sys, os
import copy
import pickle
from mpi4py import MPI
sys.path.insert(1,"../surrogate")

import numpy as np
import math
import importlib
import matplotlib.pyplot as plt
from refinecriteria import looCV, HessianFit
from aniso_criteria import AnisotropicRefine
from getxnew import getxnew, adaptivesampling
from defaults import DefaultOptOptions
from sutils import divide_cases
from error import rmse, meane, full_error

from problem_picker import GetProblem
from smt.surrogate_models import KPLS, GEKPLS, KRG
#from smt.surrogate_models.rbf import RBF
from pougrad import POUSurrogate, POUHessian
from direct_gek import DGEK
import matplotlib as mpl
import matplotlib.ticker as mticker
from smt.sampling_methods import LHS

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

"""
In this plotting script, generate desired alternative surrogate models from
those used in adaptive sampling, using the data obtained from adaptive
sampling, and compare performance.
"""

# Give directory with desired results as argument
title = sys.argv[1]
alt_model = ['KRG','GEK']#sys.argv[2]
setmod = importlib.import_module(f'{title[:-1]}.settings')
ssettings = setmod.__dict__

if not os.path.isdir(title):
    os.mkdir(title)

prob = title.split("_")[-2]
plt.rcParams['font.size'] = '13'


# Adaptive Data
with open(f'./{title}/modelf.pickle', 'rb') as f:
    modelf = pickle.load(f)
with open(f'./{title}/err0rms.pickle', 'rb') as f:
    err0rms = pickle.load(f)
with open(f'./{title}/err0mean.pickle', 'rb') as f:
    err0mean = pickle.load(f)
with open(f'./{title}/hist.pickle', 'rb') as f:
    hist = pickle.load(f)
with open(f'./{title}/errhrms.pickle', 'rb') as f:
    errhrms = pickle.load(f)
with open(f'./{title}/errhmean.pickle', 'rb') as f:
    errhmean = pickle.load(f)
# import pdb; pdb.set_trace()
# LHS Data
with open(f'./{title}/xk.pickle', 'rb') as f:
    xtrainK = pickle.load(f)
with open(f'./{title}/fk.pickle', 'rb') as f:
    ftrainK = pickle.load(f)
with open(f'./{title}/gk.pickle', 'rb') as f:
    gtrainK = pickle.load(f)
with open(f'./{title}/errkrms.pickle', 'rb') as f:
    errkrms = pickle.load(f)
with open(f'./{title}/errkmean.pickle', 'rb') as f:
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
Nerr = 5000
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
modelbase1.options.update({"print_global":False})
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
itersk = len(ekr[0])

samplehist = np.zeros(iters, dtype=int)
samplehistk = np.zeros(itersk, dtype=int)

for i in range(iters-1):
    samplehist[i] = hi[0][i][0][0].shape[0] #training_points
samplehist[iters-1] = mf[0].training_points[None][0][0].shape[0]
for i in range(itersk):
    samplehistk[i] = len(xk[i])

# Grab data from the lhs and adaptive sample sets
ind_alt = np.linspace(0, iters, itersk, dtype=int)
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

for k in range(nperr):
    ind = k + rank*nperr
    for i in range(itersk):
        ma1[k].append(copy.deepcopy(modelbase1))
        ma1[k][i].set_training_values(xa[ind][i], fa[ind][i])
        ma1[k][i].train()
        ear1[k][i], eam1[k][i], eas1[k][i] = full_error(ma1[k][i], trueFunc, N=5000, xdata=xtest, fdata=ftest)

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
        ma2[k][i].train()
        ear2[k][i], eam2[k][i], eas2[k][i] = full_error(ma2[k][i], trueFunc, N=5000, xdata=xtest, fdata=ftest)

        mh1[k].append(copy.deepcopy(modelbase1))
        mh1[k][i].set_training_values(xh[ind][i], fh[ind][i])
        mh1[k][i].train()
        ehr1[k][i], ehm1[k][i], ehs1[k][i] = full_error(mh1[k][i], trueFunc, N=5000, xdata=xtest, fdata=ftest)

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
        mh2[k][i].train()
        ehr2[k][i], ehm2[k][i], ehs2[k][i] = full_error(mh2[k][i], trueFunc, N=5000, xdata=xtest, fdata=ftest)
        # import pdb; pdb.set_trace()
ma1 = comm.allgather(ma1)
ear1 = comm.allgather(ear1)
eam1 = comm.allgather(eam1)
eas1 = comm.allgather(eas1)
ear1 = np.concatenate(ear1[:], axis=0)
eam1 = np.concatenate(eam1[:], axis=0)
eas1 = np.concatenate(eas1[:], axis=0)

ma2 = comm.allgather(ma2)
ear2 = comm.allgather(ear2)
eam2 = comm.allgather(eam2)
eas2 = comm.allgather(eas2)
ear2 = np.concatenate(ear2[:], axis=0)
eam2 = np.concatenate(eam2[:], axis=0)
eas2 = np.concatenate(eas2[:], axis=0)

mh1 = comm.allgather(mh1)
ehr1 = comm.allgather(ehr1)
ehm1 = comm.allgather(ehm1)
ehs1 = comm.allgather(ehs1)
ehr1 = np.concatenate(ehr1[:], axis=0)
ehm1 = np.concatenate(ehm1[:], axis=0)
ehs1 = np.concatenate(ehs1[:], axis=0)

mh2 = comm.allgather(mh2)
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


for i in range(nruns):
    ehrm += np.array(ehr[i]).T[0]/nruns
    ehmm += np.array(ehm[i]).T[0][0]/nruns
    ehsm += np.array(ehm[i]).T[0][1]/nruns
    ekrm += np.array(ekr[i]).T[0]/nruns
    ekmm += np.array(ekm[i]).T[0][0]/nruns
    eksm += np.array(ekm[i]).T[0][1]/nruns

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




if rank == 0:
    #NRMSE
    ax = plt.gca()
    plt.loglog(samplehist, ehrm, "b-", label=f'H. Adapt (POU)')
    plt.loglog(samplehistk, ekrm, 'b--', label='LHS (POU)')
    plt.loglog(samplehistk, earm1, 'g-', label=f'H. Adapt ({alt_model[0]})')
    plt.loglog(samplehistk, ehrm1, 'g--',  label=f'LHS ({alt_model[0]})')
    plt.loglog(samplehistk, earm2, 'r-', label=f'H. Adapt ({alt_model[1]})')
    plt.loglog(samplehistk, ehrm2, 'r--', label=f'LHS ({alt_model[1]})')
    plt.xlabel("Number of samples")
    plt.ylabel("NRMSE")
    plt.gca().set_ylim(top=10 ** math.ceil(math.log10(ehrm[0])))
    plt.gca().set_ylim(bottom=10 ** math.floor(math.log10(ehrm[-1])))
    plt.xticks(ticks=np.arange(min(samplehist), max(samplehist), 40), labels=np.arange(min(samplehist), max(samplehist), 40) )
    plt.grid()
    ax.xaxis.set_minor_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    # ax.ticklabel_format(style='plain', axis='x')
    plt.legend(loc=3)
    plt.savefig(f"./{title}/err_nrmse_ensemble_alt.pdf", bbox_inches="tight")
    plt.clf()

    ax = plt.gca()
    plt.loglog(samplehist, ehmm, "b-", label=f'H. Adapt (POU)')
    plt.loglog(samplehistk, ekmm, 'k-', label='LHS (POU)')
    plt.loglog(samplehistk, eamm1,  label=f'H. Adapt {alt_model[0]}')
    plt.loglog(samplehistk, ehmm1,  label=f'LHS {alt_model[0]}')
    plt.loglog(samplehistk, eamm2,  label=f'H. Adapt {alt_model[1]}')
    plt.loglog(samplehistk, ehmm2,  label=f'LHS {alt_model[1]}')
    plt.xlabel("Number of samples")
    plt.ylabel("Mean Error")
    plt.gca().set_ylim(top=10 ** math.ceil(math.log10(ehmm[0])))
    plt.gca().set_ylim(bottom=10 ** math.floor(math.log10(ehmm[-1])))
    plt.xticks(ticks=np.arange(min(samplehist), max(samplehist), 40), labels=np.arange(min(samplehist), max(samplehist), 40) )
    plt.grid()
    ax.xaxis.set_minor_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    # ax.ticklabel_format(style='plain', axis='x')

    plt.legend(loc=3)
    plt.savefig(f"./{title}/err_mean_ensemble_alt.pdf", bbox_inches="tight")
    plt.clf()

    ax = plt.gca()
    plt.loglog(samplehist, ehsm, "b-", label=f'H. Adapt (POU)')
    plt.loglog(samplehistk, eksm, 'k-', label='LHS (POU)')
    plt.loglog(samplehistk, easm1,  label=f'H. Adapt {alt_model[0]}')
    plt.loglog(samplehistk, ehsm1,  label=f'LHS {alt_model[0]}')
    plt.loglog(samplehistk, easm2,  label=f'H. Adapt {alt_model[1]}')
    plt.loglog(samplehistk, ehsm2,  label=f'LHS {alt_model[1]}')
    plt.xlabel("Number of samples")
    plt.ylabel(r"$\sigma$ Error")
    plt.gca().set_ylim(top=10 ** math.ceil(math.log10(ehsm[0])))
    plt.gca().set_ylim(bottom=10 ** math.floor(math.log10(ehsm[-1])))
    plt.xticks(ticks=np.arange(min(samplehist), max(samplehist), 40), labels=np.arange(min(samplehist), max(samplehist), 40) )
    plt.grid()
    ax.xaxis.set_minor_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    # ax.ticklabel_format(style='plain', axis='x')

    plt.legend(loc=3)
    plt.savefig(f"./{title}/err_stdv_ensemble_alt.pdf", bbox_inches="tight")
    plt.clf()

