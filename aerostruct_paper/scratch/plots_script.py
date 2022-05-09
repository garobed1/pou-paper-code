import sys, os
import copy
import pickle
from matplotlib.transforms import Bbox
sys.path.insert(1,"../surrogate")

import numpy as np
import matplotlib.pyplot as plt
from refinecriteria import looCV, HessianFit
from aniso_criteria import AnisotropicRefine
from getxnew import getxnew, adaptivesampling
from defaults import DefaultOptOptions
from sutils import divide_cases
from error import rmse, meane

from example_problems import  QuadHadamard, MultiDimJump, MultiDimJumpTaper, FuhgP8, FuhgP9, FuhgP10
from smt.problems import Branin, Sphere, LpNorm, Rosenbrock, WaterFlow, WeldedBeam, RobotArm, CantileverBeam
from smt.surrogate_models import KPLS, GEKPLS, KRG
#from smt.surrogate_models.rbf import RBF
from pougrad import POUSurrogate
import matplotlib as mpl
import matplotlib.ticker as mticker
from smt.sampling_methods import LHS

# Give directory with desired results as argument
usetead = True
title = sys.argv[1]
title2 = sys.argv[2]

if not os.path.isdir(title):
    os.mkdir(title)

prob = title.split("_")[0]
plt.rcParams['font.size'] = '12'

if(title2):
    
    with open(f'./{title2}/modelf.pickle', 'rb') as f:
        modelft = pickle.load(f)
    with open(f'./{title2}/err0rms.pickle', 'rb') as f:
        err0rmst = pickle.load(f)
    with open(f'./{title2}/err0mean.pickle', 'rb') as f:
        err0meant = pickle.load(f)
    with open(f'./{title2}/hist.pickle', 'rb') as f:
        histt = pickle.load(f)
    with open(f'./{title2}/errhrms.pickle', 'rb') as f:
        errhrmst = pickle.load(f)
    with open(f'./{title2}/errhmean.pickle', 'rb') as f:
        errhmeant = pickle.load(f)

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
    ekr = ekr + errkrms[i][:]
    ekm = ekm + errkmean[i][:]
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
else:
    raise ValueError("Given problem not valid.")


for i in range(nruns):
    ehr[i] = [e0r[i]] + ehr[i] #[errf] #errh
    ehm[i] = [e0m[i]] + ehm[i]
    if(title2):
        ehrt[i] = [e0rt[i]] + ehrt[i] #[errf] #errh
        ehmt[i] = [e0mt[i]] + ehmt[i]


# Plot Error History
iters = len(ehr[0])
itersk = len(ekr[0])
if(title2):
    iterst = len(ehrt[0])
    if(iterst < iters):
        iters = iterst

samplehist = np.zeros(iters, dtype=int)
samplehistk = np.zeros(itersk, dtype=int)

for i in range(iters-1):
    samplehist[i] = hi[0][i].ntr
samplehist[iters-1] = mf[0].training_points[None][0][0].shape[0]
for i in range(itersk):
    samplehistk[i] = len(xk[i])

if(title2):
    xt = np.linspace(0, samplehist[-1]-samplehist[0], iters, dtype=int)
    for i in range(nruns):
        ehrt[i] = [ehrt[i][j] for j in xt]
        ehmt[i] = [ehmt[i][j] for j in xt]

# Average out runs
ehrm = np.zeros(iters)
ehmm = np.zeros(iters)
ehsm = np.zeros(iters) 
ekrm = np.zeros(itersk)
ekmm = np.zeros(itersk)
eksm = np.zeros(itersk)
if(title2):
    ehrmt = np.zeros(iters)
    ehmmt = np.zeros(iters)
    ehsmt = np.zeros(iters) 

for i in range(nruns):
    ehrm += np.array(ehr[i]).T[0]/nruns
    ehmm += np.array(ehm[i]).T[0][0]/nruns
    ehsm += np.array(ehm[i]).T[0][1]/nruns
    ekrm += np.array(ekr[i]).T[0]/nruns
    ekmm += np.array(ekm[i]).T[0][0]/nruns
    eksm += np.array(ekm[i]).T[0][1]/nruns
    if(title2):
        ehrmt += np.array(ehrt[i]).T[0]/nruns
        ehmmt += np.array(ehmt[i]).T[0][0]/nruns
        ehsmt += np.array(ehmt[i]).T[0][1]/nruns

# plt.loglog(samplehist, ehrm, "-", label=f'Adaptive Runs Ensemble')
# plt.loglog(samplehistk, ekrm, 'k--', label='LHS Runs Ensemble')
# plt.xlabel("Number of samples")
# plt.ylabel("NRMSE")
# plt.legend(loc=1)
# # import matplotlib.ticker
# # ax = plt.gca()
# # ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
# # ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
# # ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
# # ax.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
# plt.xticks(np.arange(min(samplehist), max(samplehist), 10))
# plt.ticklabel_format(style='plain', axis='x')
# # plt.yticks(np.arange(0.04, 0.18, 0.01))
# plt.savefig(f"./{title}err_rms_ensemble.png", bbox_inches="tight")
# plt.clf()


# plt.loglog(samplehist, ehmm, "-", label='Adaptive Runs Ensemble' )
# plt.loglog(samplehistk, ekmm, 'k--', label='LHS Runs Ensemble')
# plt.xlabel("Number of samples")
# plt.ylabel("Relative Mean Error")
# plt.legend(loc=1)
# plt.savefig(f"./{title}err_mean_ensemble.png", bbox_inches="tight")
# plt.clf()

#NRMSE
ax = plt.gca()
plt.loglog(samplehist, ehrm, "b-", label=f'AIGES NRMSE')
plt.loglog(samplehistk, ekrm, 'k-', label='LHS NRMSE')
if(title2):
    plt.loglog(samplehist, ehrmt, "r-", label=f'TEAD NRMSE')
plt.xlabel("Number of samples")
plt.ylabel("Relative error")
plt.xticks(ticks=np.arange(min(samplehist), max(samplehist), 40), labels=np.arange(min(samplehist), max(samplehist), 40) )
plt.grid()
ax.xaxis.set_minor_formatter(mticker.ScalarFormatter())
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.ticklabel_format(style='plain', axis='x')

plt.legend(loc=1)
plt.savefig(f"./{title}err_nrmse_ensemble.png", bbox_inches="tight")
plt.clf()

ax = plt.gca()
plt.loglog(samplehist, ehmm, "b--", label='AIGES Mean' )
plt.loglog(samplehistk, ekmm, 'k--', label='LHS Mean')
plt.loglog(samplehist, ehsm, "b-.", label='AIGES Std. Dev.' )
plt.loglog(samplehistk, eksm, 'k-.', label='LHS Std. Dev.')
if(title2):
    plt.loglog(samplehist, ehmmt, "r--", label='AIGES Mean' )
    plt.loglog(samplehist, ehsmt, "r-.", label='AIGES Std. Dev.' )

plt.xlabel("Number of samples")
plt.ylabel("Relative error")
plt.xticks(ticks=np.arange(min(samplehist), max(samplehist), 40), labels=np.arange(min(samplehist), max(samplehist), 40) )
plt.grid()
ax.xaxis.set_minor_formatter(mticker.ScalarFormatter())
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.ticklabel_format(style='plain', axis='x')

plt.legend(loc=1)
plt.savefig(f"./{title}err_uq_ensemble.png", bbox_inches="tight")
plt.clf()

trx = mf[0].training_points[None][0][0]
m, n = trx.shape
normal = np.ones(dim)
normal /= np.linalg.norm(normal)

planedists = np.zeros(m)
for i in range(m):
    planedists[i] = abs(np.dot(trx[i,:],normal))


# # Plot points
if(dim == 2):
    bbox = Bbox([[0.0, 0], [6.5, 4.3]])
    mk = copy.deepcopy(mf[0])
    mk.set_training_values(xtrainK[0][-1], ftrainK[0][-1])
    if(isinstance(mk, GEKPLS) or isinstance(mk, POUSurrogate)):
        for i in range(dim):
            mk.set_training_derivatives(xtrainK[0][-1], gtrainK[0][-1][:,i:i+1], i)
    mk.train()
    plt.clf()
    nt0 = samplehist[0]
    # Plot Training Points
    plt.plot(trx[0:nt0,0], trx[0:nt0,1], "bo", label='Initial Samples')
    plt.plot(trx[nt0:,0], trx[nt0:,1], "ro", label='Adaptive Samples')
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    #plt.legend(loc=1)
    plt.savefig(f"./{title}2d_aniso_pts.png", bbox_inches="tight")#"tight")
    plt.clf()
    
    # Plot Error Contour
    #Contour
    ndir = 150
    xlimits = trueFunc.xlimits
    x = np.linspace(xlimits[0][0], xlimits[0][1], ndir)
    y = np.linspace(xlimits[1][0], xlimits[1][1], ndir)

    X, Y = np.meshgrid(x, y)
    Za = np.zeros([ndir, ndir])
    Va = np.zeros([ndir, ndir])
    V0 = np.zeros([ndir, ndir])
    Zk = np.zeros([ndir, ndir])
    Vk = np.zeros([ndir, ndir])
    Z0 = np.zeros([ndir, ndir])
    F  = np.zeros([ndir, ndir])
    TF = np.zeros([ndir, ndir])

    for i in range(ndir):
        for j in range(ndir):
            xi = np.zeros([1,2])
            xi[0,0] = x[i]
            xi[0,1] = y[j]
            F[j,i]  = mf[0].predict_values(xi)
            TF[j,i] = trueFunc(xi)
            Za[j,i] = abs(F[j,i] - TF[j,i])
            Zk[j,i] = abs(mk.predict_values(xi) - TF[j,i])


    cs = plt.contour(X, Y, Za, levels = 40)
    plt.colorbar(cs, aspect=20)
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    #plt.legend(loc=1)
    plt.plot(trx[0:nt0,0], trx[0:nt0,1], "o", fillstyle='full', markerfacecolor='b', markeredgecolor='b', label='Initial Samples')
    plt.plot(trx[nt0:,0], trx[nt0:,1], "o", fillstyle='full', markerfacecolor='r', markeredgecolor='r', label='Adaptive Samples')
    plt.savefig(f"./{title}2d_errcona.png", bbox_inches="tight")

    plt.clf()

    # Plot Non-Adaptive Error
    tk = mk.training_points[None][0][0]
    plt.contour(X, Y, Zk, levels = cs.levels)
    plt.colorbar(cs, )
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.plot(tk[:,0], tk[:,1], "o", fillstyle='full', markerfacecolor='b', markeredgecolor='b', label='LHS Samples')
    plt.savefig(f"./{title}2d_errconk.png", bbox_inches="tight")

    plt.clf()

Nerr = 5000
sampling = LHS(xlimits=trueFunc.xlimits, criterion='m')
xtest = sampling(Nerr)
ftest = trueFunc(xtest)
meantrue = sum(ftest)/Nerr
stdtrue = np.sqrt((sum(ftest*ftest)/Nerr) - (sum(ftest)/Nerr)**2)

meanlhstrue = sum(fk[0])/fk[0].shape[0]
stdlhstrue = np.sqrt((sum(fk[0]*fk[0])/fk[0].shape[0]) - (sum(fk[0])/fk[0].shape[0])**2)

faiges = mf[0].predict_values(xtest)
meanaiges = sum(faiges)/Nerr
stdaiges = np.sqrt((sum(faiges*faiges)/Nerr) - (sum(faiges)/Nerr)**2)

ftead = mft[0].predict_values(xtest)
meantead = sum(ftead)/Nerr
stdtead = np.sqrt((sum(ftead*ftead)/Nerr) - (sum(ftead)/Nerr)**2)


mf[0].set_training_values(xk[0], fk[0])
if(isinstance(mf[0], GEKPLS) or isinstance(mf[0], POUSurrogate)):
    for i in range(dim):
        mf[0].set_training_derivatives(xk[0], gk[0][:,i:i+1], i)
mf[0].train()
flhs = mf[0].predict_values(xtest)

meanlhs = sum(flhs)/Nerr
stdlhs  = np.sqrt((sum(flhs*flhs)/Nerr) - (sum(flhs)/Nerr)**2)


print("True Mean: ", meantrue)
print("True LHS Mean: ", meanlhstrue)
print("LHS Mean: ", meanlhs)
print("AIGES Mean: ", meanaiges)
print("TEAD Mean: ", meantead)
print("True std: ", stdtrue)
print("True LHS std: ", stdlhstrue)
print("LHS std: ", stdlhs)
print("AIGES std: ", stdaiges)
print("TEAD std: ", stdtead)