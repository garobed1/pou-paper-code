import sys, os
import copy
import pickle
import math
from matplotlib.transforms import Bbox
sys.path.insert(1,"../surrogate")

import numpy as np
import matplotlib.pyplot as plt

from functions.problem_picker import GetProblem
from smt.surrogate_models import KPLS, GEKPLS, KRG
#from smt.surrogate_models.rbf import RBF
from surrogate.pougrad import POUSurrogate, POUHessian
import matplotlib as mpl
import matplotlib.ticker as mticker
from smt.sampling_methods import LHS

# Give directory with desired results as argument
title = sys.argv[1]

"adding comm line option to double x axis for gradient models"
title2 = None
fac = 1.0
if len(sys.argv) > 2:
    fac = sys.argv[2]
if len(sys.argv) > 3:
    title2 = sys.argv[3]

if not os.path.isdir(title):
    os.mkdir(title)

prob = title.split("_")[-2]
plt.rcParams['font.size'] = '16'

if(title2):
    
    with open(f'{title2}/modelf.pickle', 'rb') as f:
        modelft = pickle.load(f)
    with open(f'{title2}/err0rms.pickle', 'rb') as f:
        err0rmst = pickle.load(f)
    with open(f'{title2}/err0mean.pickle', 'rb') as f:
        err0meant = pickle.load(f)
    with open(f'{title2}/hist.pickle', 'rb') as f:
        histt = pickle.load(f)
    with open(f'{title2}/errhrms.pickle', 'rb') as f:
        errhrmst = pickle.load(f)
    with open(f'{title2}/errhmean.pickle', 'rb') as f:
        errhmeant = pickle.load(f)

# Adaptive Data
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
if(title2):
    mft = []
    e0rt = []
    e0mt = []
    hit = []
    ehrt = []
    ehmt = []
nprocs = len(modelf)
for i in range(nprocs):
    xk = xk + xtrainK[i][:]
    fk = fk + ftrainK[i][:]
    gk = gk + gtrainK[i][:]
    # ekr = ekr + errkrms[i][:]
    # ekm = ekm + errkmean[i][:]
    mf = mf + modelf[i][:]
    e0r = e0r + err0rms[i]
    e0m = e0m + err0mean[i]
    hi = hi + hist[i][:]
    ehr = ehr + errhrms[i][:]
    ehm = ehm + errhmean[i][:]
    if(title2):
        mft =  mft  + modelft[i][:]
        e0rt = e0rt + err0rmst[i]
        e0mt = e0mt + err0meant[i]
        hit =  hit  + histt[i][:]
        ehrt = ehrt + errhrmst[i][:]
        ehmt = ehmt + errhmeant[i][:]

nruns = len(mf)
dim = xk[0].shape[1]


# Problem Settings
trueFunc = GetProblem(prob, dim)


for i in range(nruns):
    ehr[i] = [e0r[i]] + ehr[i] #[errf] #errh
    ehm[i] = [e0m[i]] + ehm[i]
    if(title2):
        ehrt[i] = [e0rt[i]] + ehrt[i] #[errf] #errh
        ehmt[i] = [e0mt[i]] + ehmt[i]
# ekr = [ekr]
# ekm = [ekm]

# Plot Error History
iters = len(ehr[0])
if(dim > 3):
    with open(f'{title}/intervals.pickle', 'rb') as f:
        intervals = pickle.load(f)
    #iters = intervals.shape[0] + 1
else:
    intervals = np.arange(iters)

# itersk = len(ekr[0])
if(title2):
    iterst = len(ehrt[0])
    if(iterst < iters):
        iters = iterst

samplehist = np.zeros(iters, dtype=int)
# samplehistk = np.zeros(itersk, dtype=int)

samplehist[0] = hi[0][0][0][0].shape[0] #training_points 
for i in range(1, iters-1):
    samplehist[i] = samplehist[i-1] + (intervals[1] - intervals[0])
samplehist[iters-1] = mf[0].training_points[None][0][0].shape[0]
# for i in range(itersk):
#     samplehistk[i] = len(xk[i])

if(title2):
    xt = np.linspace(0, samplehist[-1]-samplehist[0], iters, dtype=int)
    for i in range(nruns):
        ehrt[i] = [ehrt[i][j] for j in xt]
        ehmt[i] = [ehmt[i][j] for j in xt]

# Average out runs
ehrm = np.zeros(iters)
ehmm = np.zeros(iters)
ehsm = np.zeros(iters) 
# ekrm = np.zeros(itersk)
# ekmm = np.zeros(itersk)
# eksm = np.zeros(itersk)
if(title2):
    ehrmt = np.zeros(iters)
    ehmmt = np.zeros(iters)
    ehsmt = np.zeros(iters) 

for i in range(nruns):
    ehrm += np.array(ehr[i]).T[0]/nruns
    ehmm += np.array(ehm[i]).T[0][0]/nruns
    ehsm += np.array(ehm[i]).T[0][1]/nruns
    # ekrm += np.array(ekr[i]).T[0]/nruns
    # ekmm += np.array(ekm[i]).T[0][0]/nruns
    # eksm += np.array(ekm[i]).T[0][1]/nruns
    if(title2):
        ehrmt += np.array(ehrt[i]).T[0]/nruns
        ehmmt += np.array(ehmt[i]).T[0][0]/nruns
        ehsmt += np.array(ehmt[i]).T[0][1]/nruns





#NRMSE
ax = plt.gca()
plt.loglog(samplehist, ehrm, "b-", label=f'Adaptive')
# plt.loglog(samplehistk, ekrm, 'k-', label='LHS')
if(title2):
    plt.loglog(samplehist, ehrmt, "r-", label=f'TEAD NRMSE')
plt.xlabel("Number of samples")
plt.ylabel("NRMSE")
plt.gca().set_ylim(top=10 ** math.ceil(math.log10(ehrm[0])))
plt.gca().set_ylim(bottom=10 ** math.floor(math.log10(ehrm[-1])))
plt.xticks(ticks=np.arange(min(samplehist), max(samplehist), 40), labels=np.arange(min(samplehist), max(samplehist), 40) )
plt.grid()
ax.xaxis.set_minor_formatter(mticker.ScalarFormatter())
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
plt.legend(loc=3)
plt.savefig(f"{title}/err_nrmse_ensemble.pdf", bbox_inches="tight")
plt.clf()

ax = plt.gca()
plt.loglog(samplehist, ehmm, "b-", label='Adaptive' )
# plt.loglog(samplehistk, ekmm, 'k-', label='LHS')
plt.xlabel("Number of samples")
plt.ylabel("Mean Error")
plt.gca().set_ylim(top=10 ** math.ceil(math.log10(ehmm[0])))
plt.gca().set_ylim(bottom=10 ** math.floor(math.log10(ehmm[-1])))
plt.xticks(ticks=np.arange(min(samplehist), max(samplehist), 40), labels=np.arange(min(samplehist), max(samplehist), 40) )
plt.grid()
ax.xaxis.set_minor_formatter(mticker.ScalarFormatter())
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())

plt.legend(loc=3)
plt.savefig(f"{title}/err_mean_ensemble.pdf", bbox_inches="tight")
plt.clf()

ax = plt.gca()
plt.loglog(samplehist, ehsm, "b-", label='Adaptive' )
# plt.loglog(samplehistk, eksm, 'k-', label='LHS')
plt.xlabel("Number of samples")
plt.ylabel(r"$\sigma$ Error")
plt.gca().set_ylim(top=10 ** math.ceil(math.log10(ehsm[0])))
plt.gca().set_ylim(bottom=10 ** math.floor(math.log10(ehsm[-1])))
plt.xticks(ticks=np.arange(min(samplehist), max(samplehist), 40), labels=np.arange(min(samplehist), max(samplehist), 40) )
plt.grid()
ax.xaxis.set_minor_formatter(mticker.ScalarFormatter())
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())

plt.legend(loc=3)
plt.savefig(f"{title}/err_stdv_ensemble.pdf", bbox_inches="tight")
plt.clf()

trx = mf[0].training_points[None][0][0]
trf = mf[0].training_points[None][0][1]
m, n = trx.shape
normal = np.ones(dim)
normal /= np.linalg.norm(normal)

planedists = np.zeros(m)
for i in range(m):
    planedists[i] = abs(np.dot(trx[i,:],normal))

mk = copy.deepcopy(mf[0])
mk.set_training_values(xk[-1], fk[-1])
if(isinstance(mk, GEKPLS) or isinstance(mk, POUSurrogate) or isinstance(mk, POUHessian)):
    for j in range(dim):
        mk.set_training_derivatives(xk[-1], gk[-1][:,j:j+1], j)
mk.train()
trxk = mk.training_points[None][0][0]
trfk = mk.training_points[None][0][1]
mk.options.update({"print_global":False})
mf[0].options.update({"print_global":False})
pmod = mf[0]

#snapshot is proc, iteration
#snapshot = [30, 60] 
snapshot = 0
if(snapshot):
    trx = hi[snapshot[0]][snapshot[1]][0][0]
    trf = hi[snapshot[0]][snapshot[1]][0][1]
    trg = np.zeros_like(trx)
    if(isinstance(mf[0], GEKPLS) or isinstance(mf[0], POUSurrogate) or isinstance(mf[0], POUHessian)):
        for j in range(dim):
            trg[:,j:j+1] = hi[snapshot[0]][snapshot[1]][j+1][1]
    m, n = trx.shape


    pmod = copy.deepcopy(mf[0])
    pmod.set_training_values(trx, trf)
    if(isinstance(mk, GEKPLS) or isinstance(mk, POUSurrogate) or isinstance(mk, POUHessian)):
        for j in range(dim):
            pmod.set_training_derivatives(trx, trg[:,j:j+1], j)
    pmod.train()
    pmod.options.update({"print_global":False})

    import pdb; pdb.set_trace()

# # Plot points

if(dim == 1):
    plt.clf()
    nt0 = samplehist[0]
    # Plot Training Points
    plt.plot(trx[0:nt0,0], np.zeros(trx[0:nt0,0].shape[0]), "bo", label='Initial Samples')
    plt.plot(trx[nt0:,0], np.zeros(trx[nt0:,0].shape[0]), "ro", label='Adaptive Samples')
    plt.xlabel(r"$x$")
    plt.ylabel(r"$f$")
    #plt.legend(loc=1)
    plt.savefig(f"{title}/1d_adaptive_pts.pdf", bbox_inches="tight")#"tight")
    plt.clf()

    ndir = 75
    xlimits = trueFunc.xlimits
    x = np.linspace(xlimits[0][0], xlimits[0][1], ndir)

    Z = np.zeros([ndir])
    Zh = np.zeros([ndir])
    F = np.zeros([ndir])
    Fh = np.zeros([ndir])
    TF = np.zeros([ndir])

    for i in range(ndir):
        xi = np.zeros([1,1])
        xi[0] = x[i]
        TF[i] = trueFunc(xi)
        F[i] = pmod.predict_values(xi)
        Fh[i] = mk.predict_values(xi)
        Z[i] = abs(F[i] - TF[i])
        Zh[i] = abs(Fh[i] - TF[i])

    # Plot the target function
    plt.plot(x, TF, "-k", label=f'True')
    plt.xlabel(r"$x$")
    plt.ylabel(r"$f$")
    #plt.legend(loc=1)
    plt.savefig(f"{title}/1dtrue.pdf", bbox_inches="tight")
    plt.clf()

    # Plot Non-Adaptive Error
    plt.plot(x, TF, "-k", label=f'True')
    # plt.plot(x, Fgek, "-m", label=f'IGEK')
    plt.plot(x, F, "-b", label=f'AS')
    plt.xlabel(r"$x$")
    plt.ylabel(r"$f$")
    #plt.legend(loc=1)
    plt.plot(trx[0:nt0,0], trf[0:nt0,0], "bo", label='Initial Samples')
    plt.plot(trx[nt0:,0], trf[nt0:,0], "ro", label='Adaptive Samples')
    plt.savefig(f"{title}/1dplot.pdf", bbox_inches="tight")
    plt.clf()

    # Plot Non-Adaptive Error
    # plt.plot(x, Zgek, "-m", label=f'IGEK')
    plt.plot(x, Z, "-k", label=f'Adaptive (POU)')
    plt.plot(x, Zh, "--k", label=f'LHS (POU)')
    plt.xlabel(r"$x$")
    plt.ylabel(r"$|\hat{f}(\mathbf{x}) - f(\mathbf{x})|$")
    plt.plot(trx[0:nt0,0], np.zeros_like(trf[0:nt0,0]), "bo", label='Initial Samples')
    plt.plot(trx[nt0:,0], np.zeros_like(trf[nt0:,0]), "ro", label='Added Samples')
    plt.plot(trxk, max(np.max(Z), np.max(Zh))*np.ones_like(trxk), "ko", label='LHS Samples')
    plt.legend(fontsize='13')
    plt.savefig(f"{title}/1derr.pdf", bbox_inches="tight")

    plt.clf()


if(dim == 2):

    plt.clf()
    nt0 = samplehist[0]
    # Plot Training Points
    plt.plot(trx[0:nt0,0], trx[0:nt0,1], "bo", label='Initial Samples')
    plt.plot(trx[nt0:,0], trx[nt0:,1], "ro", label='Adaptive Samples')
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    #plt.legend(loc=1)
    plt.savefig(f"{title}/2d_adaptive_pts.pdf", bbox_inches="tight")#"tight")
    plt.clf()
    
    plt.clf()
    # Plot Training Points
    plt.plot(trxk[:,0], trxk[:,1], "bo", label='LHS Points')
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    #plt.legend(loc=1)
    plt.savefig(f"{title}/2d_lhs_pts.pdf", bbox_inches="tight")#"tight")
    plt.clf()

    # Plot Error contour
    #contour
    ndir = 150
    xlimits = trueFunc.xlimits
    x = np.linspace(xlimits[0][0], xlimits[0][1], ndir)
    y = np.linspace(xlimits[1][0], xlimits[1][1], ndir)

    X, Y = np.meshgrid(x, y)
    Za = np.zeros([ndir, ndir])
    Zk = np.zeros([ndir, ndir])
    F  = np.zeros([ndir, ndir])
    FK  = np.zeros([ndir, ndir])
    TF = np.zeros([ndir, ndir])

    for i in range(ndir):
        for j in range(ndir):
            xi = np.zeros([1,2])
            xi[0,0] = x[i]
            xi[0,1] = y[j]
            F[j,i]  = pmod.predict_values(xi)
            FK[j,i] = mk.predict_values(xi)
            TF[j,i] = trueFunc(xi)
            Za[j,i] = abs(F[j,i] - TF[j,i])
            Zk[j,i] = abs(FK[j,i] - TF[j,i])

    # Plot original function
    cs = plt.contourf(X, Y, TF, levels = 40)
    plt.colorbar(cs, aspect=20)
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    #plt.legend(loc=1)
    plt.savefig(f"{title}/2d_true.pdf", bbox_inches="tight")

    plt.clf()

    cs = plt.contourf(X, Y, Za, levels = 40)
    plt.colorbar(cs, aspect=20, label = r"$|\hat{f}(\mathbf{x}) - f(\mathbf{x})|$")
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    #plt.legend(loc=1)
    plt.plot(trx[0:nt0,0], trx[0:nt0,1], "o", fillstyle='full', markerfacecolor='b', markeredgecolor='b', label='Initial Samples')
    plt.plot(trx[nt0:,0], trx[nt0:,1], "o", fillstyle='full', markerfacecolor='r', markeredgecolor='r', label='Adaptive Samples')
    plt.savefig(f"{title}/2d_errcon_a.pdf", bbox_inches="tight")

    plt.clf()

    # Plot Non-Adaptive Error
    tk = mk.training_points[None][0][0]
    plt.contourf(X, Y, Zk, levels = cs.levels, label = r"$|\hat{f}(\mathbf{x}) - f(\mathbf{x})|$")
    plt.colorbar(cs, )
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.plot(tk[:,0], tk[:,1], "o", fillstyle='full', markerfacecolor='b', markeredgecolor='b', label='LHS Samples')
    plt.savefig(f"{title}/2d_errcon_k.pdf", bbox_inches="tight")

    plt.clf()








# Nerr = 5000
# sampling = LHS(xlimits=trueFunc.xlimits, criterion='m')
# xtest = sampling(Nerr)
# ftest = trueFunc(xtest)
# meantrue = sum(ftest)/Nerr
# stdtrue = np.sqrt((sum(ftest*ftest)/Nerr) - (sum(ftest)/Nerr)**2)

# meanlhstrue = sum(fk[0])/fk[0].shape[0]
# stdlhstrue = np.sqrt((sum(fk[0]*fk[0])/fk[0].shape[0]) - (sum(fk[0])/fk[0].shape[0])**2)

# faiges = mf[0].predict_values(xtest)
# meanaiges = sum(faiges)/Nerr
# stdaiges = np.sqrt((sum(faiges*faiges)/Nerr) - (sum(faiges)/Nerr)**2)

# ftead = mft[0].predict_values(xtest)
# meantead = sum(ftead)/Nerr
# stdtead = np.sqrt((sum(ftead*ftead)/Nerr) - (sum(ftead)/Nerr)**2)


# mf[0].set_training_values(xk[0], fk[0])
# if(isinstance(mf[0], GEKPLS) or isinstance(mf[0], POUSurrogate)):
#     for i in range(dim):
#         mf[0].set_training_derivatives(xk[0], gk[0][:,i:i+1], i)
# mf[0].train()
# flhs = mf[0].predict_values(xtest)

# meanlhs = sum(flhs)/Nerr
# stdlhs  = np.sqrt((sum(flhs*flhs)/Nerr) - (sum(flhs)/Nerr)**2)


# print("True Mean: ", meantrue)
# print("True LHS Mean: ", meanlhstrue)
# print("LHS Mean: ", meanlhs)
# print("AIGES Mean: ", meanaiges)
# print("TEAD Mean: ", meantead)
# print("True std: ", stdtrue)
# print("True LHS std: ", stdlhstrue)
# print("LHS std: ", stdlhs)
# print("AIGES std: ", stdaiges)
# print("TEAD std: ", stdtead)