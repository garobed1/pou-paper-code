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

from scipy.spatial.distance import pdist, cdist
from example_problems import  QuadHadamard, MultiDimJump, MultiDimJumpTaper, FuhgP8, FuhgP9, FuhgP10
from smt.problems import Branin, Sphere, LpNorm, Rosenbrock, WaterFlow, WeldedBeam, RobotArm, CantileverBeam
from shock_problem import ImpingingShock
from smt.surrogate_models import KPLS, GEKPLS, KRG
#from smt.surrogate_models.rbf import RBF
from pougrad import POUSurrogate
import matplotlib as mpl
import matplotlib.ticker as mticker
from smt.sampling_methods import LHS

dim = 2

# Give directory with desired results as argument
iters = 11
itersk = 3
titleref = "5000_shock_results"
titlelhs1 = ["70_shock_results_1","70_shock_results_2","70_shock_results_3","70_shock_results_4"]
nrunsk1 = len(titlelhs1)
titlelhs2 = ["120_shock_results_1","120_shock_results_3","120_shock_results_4","120_shock_results_5"]
nrunsk2 = len(titlelhs2)
titleaiges = ["SHOCK_aniso_1","SHOCK_aniso_2","SHOCK_aniso_3","SHOCK_aniso_4","SHOCK_aniso_5"]
title = "shock_plots"
if not os.path.isdir(title):
    os.mkdir(title)

# reference data
with open(f'./{titleref}/xref.pickle', 'rb') as f:
    xref = pickle.load(f)
with open(f'./{titleref}/fref.pickle', 'rb') as f:
    fref = pickle.load(f)
with open(f'./{titleref}/gref.pickle', 'rb') as f:
    gref = pickle.load(f)

xref = xref.astype(float)

xlimits = np.zeros([2,2])
xlimits[0,:] = [23., 27.]
xlimits[1,:] = [0.36, 0.51]
inputs = ["shock_angle", "rsak"]
dummy = ImpingingShock(ndim=2, input_bounds=xlimits, inputs=inputs)

xtest = np.array(xref, dtype=float)
ftest = np.array(fref, dtype=float)
gtest = np.array(gref, dtype=float)
testdata = [xtest, ftest]
SCALE = 1
ftest = SCALE*ftest
gtest = SCALE*gtest

ekr = []
ekm = []
mf = []
e0r = []
e0m = []
hi = []
ehr = []
ehm = []
# AIGES data
co = 0
for key in titleaiges:
    with open(f'./{key}/hist.pickle', 'rb') as f:
        hi = hi + [pickle.load(f)]
    with open(f'./{key}/modelf.pickle', 'rb') as f:
        mf = mf + [pickle.load(f)]

    

    # # 
    # trx = mf[co].training_points[None][0][0]
    # trf = mf[co].training_points[None][0][1]
    # trg = np.zeros_like(trx)
    # for j in range(dim):
    #     trg[:,j] = mf[co].training_points[None][j+1][1].flatten()
    # trxc = np.zeros_like(trx)
    # trfc = np.zeros_like(trf)
    # trgc = np.zeros_like(trg)
    # dists = cdist(trx, xtest)
    # for k in range(trx.shape[0]):
    #     ind = np.argmin(dists[k,:])
    #     trxc[k] = xtest[ind]
    #     trfc[k] = ftest[ind]
    #     trgc[k] = gtest[ind]
    # modelbase.set_training_values(trxc, trfc)
    # for j in range(dim):
    #     modelbase.set_training_derivatives(trxc, trgc[:,j:j+1], j)
    # modelbase.train()

    modelbase = GEKPLS(xlimits=xlimits)
    modelbase.options.update({"extra_points":1})
    modelbase.options.update({"corr":'abs_exp'})
    modelbase.options.update({"poly":'linear'})
    modelbase.options.update({"n_start":5})
    modelbase.options.update({"print_global":False})
    trx = hi[co][0].model.training_points[None][0][0]
    trf = hi[co][0].model.training_points[None][0][1]
    trg = np.zeros_like(trx)
    for j in range(dim):
        trg[:,j] = hi[co][0].model.training_points[None][j+1][1].flatten()
    trxc = np.zeros_like(trx)
    trfc = np.zeros_like(trf)
    trgc = np.zeros_like(trg)
    dists = cdist(trx, xtest)
    for k in range(trx.shape[0]):
        ind = np.argmin(dists[k,:])
        trxc[k] = xtest[ind]
        trfc[k] = ftest[ind]
        trgc[k] = gtest[ind]
    modelbase.set_training_values(trxc, trfc)
    for j in range(dim):
        modelbase.set_training_derivatives(trxc, trgc[:,j:j+1], j)
    modelbase.train()
    # compute error
    e0r = e0r + [rmse(modelbase, dummy, xdata=xtest, fdata=ftest)]
    e0m = e0m +  [meane(modelbase, dummy, xdata=xtest, fdata=ftest)]

    err1 = []
    err2 = []
    for i in range(0, len(hi[co])):

        # c
        modelbase = GEKPLS(xlimits=xlimits)
        modelbase.options.update({"extra_points":1})
        modelbase.options.update({"corr":'abs_exp'})
        modelbase.options.update({"poly":'linear'})
        modelbase.options.update({"n_start":5})
        modelbase.options.update({"print_global":False})
        trx = hi[co][i].model.training_points[None][0][0]
        trf = hi[co][i].model.training_points[None][0][1]
        trg = np.zeros_like(trx)
        for j in range(dim):
            trg[:,j] = hi[co][i].model.training_points[None][j+1][1].flatten()
        trxc = np.zeros_like(trx)
        trfc = np.zeros_like(trf)
        trgc = np.zeros_like(trg)
        dists = cdist(trx, xtest)
        for k in range(trx.shape[0]):
            ind = np.argmin(dists[k,:])
            trxc[k] = xtest[ind]
            trfc[k] = ftest[ind]
            trgc[k] = gtest[ind]
        modelbase.set_training_values(trxc, trfc)
        for j in range(dim):
            modelbase.set_training_derivatives(trxc, trgc[:,j:j+1], j)
        modelbase.train()



        err1 = err1 + [rmse(modelbase, dummy, xdata=xtest, fdata=ftest)]
        err2 = err2 + [meane(modelbase, dummy, xdata=xtest, fdata=ftest)]
    
    modelbase = GEKPLS(xlimits=xlimits)
    modelbase.options.update({"extra_points":1})
    modelbase.options.update({"corr":'abs_exp'})
    modelbase.options.update({"poly":'linear'})
    modelbase.options.update({"n_start":5})
    modelbase.options.update({"print_global":False})

    # 
    trx = mf[co].training_points[None][0][0]
    trf = mf[co].training_points[None][0][1]
    trg = np.zeros_like(trx)
    for j in range(dim):
        trg[:,j] = mf[co].training_points[None][j+1][1].flatten()
    trxc = np.zeros_like(trx)
    trfc = np.zeros_like(trf)
    trgc = np.zeros_like(trg)
    dists = cdist(trx, xtest)
    for k in range(trx.shape[0]):
        ind = np.argmin(dists[k,:])
        trxc[k] = xtest[ind]
        trfc[k] = ftest[ind]
        trgc[k] = gtest[ind]
    modelbase.set_training_values(trxc, trfc)
    for j in range(dim):
        modelbase.set_training_derivatives(trxc, trgc[:,j:j+1], j)
    modelbase.train()
    err1 = err1 + [rmse(modelbase, dummy, xdata=xtest, fdata=ftest)]
    err2 = err2 + [meane(modelbase, dummy, xdata=xtest, fdata=ftest)]
    
    ehr = ehr + [err1]
    ehm = ehm + [err2]
    
    co += 1

modelbasea = copy.deepcopy(modelbase)
# LHS data
modelbase = GEKPLS(xlimits=xlimits)
modelbase.options.update({"extra_points":1})
modelbase.options.update({"corr":'abs_exp'})
modelbase.options.update({"poly":'linear'})
modelbase.options.update({"n_start":5})
modelbase.options.update({"print_global":False})

co = 0
x1 = []
f1 = []
g1 = []
m1 = []
for key in titlelhs1:
    with open(f'./{key}/x0to120.pickle', 'rb') as f:
        x1 = x1 + [pickle.load(f)]
    with open(f'./{key}/y0to120.pickle', 'rb') as f:
        f1 = f1 + [SCALE*pickle.load(f)]
    with open(f'./{key}/g0to120.pickle', 'rb') as f:
        g1 = g1 + [SCALE*pickle.load(f)]

    
    ekr.append([e0r[co]])
    ekm.append([e0m[co]])

    # train models
    m1.append(copy.deepcopy(modelbase))
    m1[co].set_training_values(x1[co], f1[co])
    for i in range(dim):
        m1[co].set_training_derivatives(x1[co], g1[co][:,i:i+1], i)
    m1[co].train()

    ekr[co] = ekr[co] + [rmse(m1[co], dummy, xdata=xtest, fdata=ftest)]
    ekm[co] = ekm[co] + [meane(m1[co], dummy, xdata=xtest, fdata=ftest)]
    
    co += 1

# # don't count last one
# ekr.append([e0r[co]])
# ekm.append([e0m[co]])
# ekr[co] = ekr[co] + [np.array([0])]
# ekm[co] = ekm[co] + [(np.array([0]),np.array([0]))]

co = 0
x2 = []
f2 = []
g2 = []
m2 = []
for key in titlelhs2:
    with open(f'./{key}/x0to120.pickle', 'rb') as f:
        x2 = x2 + [pickle.load(f)]
    with open(f'./{key}/y0to120.pickle', 'rb') as f:
        f2 = f2 + [pickle.load(f)]
    with open(f'./{key}/g0to120.pickle', 'rb') as f:
        g2 = g2 + [pickle.load(f)]

    # train models
    m2.append(copy.deepcopy(modelbase))
    m2[co].set_training_values(x2[co], f2[co])
    for i in range(dim):
        m2[co].set_training_derivatives(x2[co], g2[co][:,i:i+1], i)
    m2[co].train()

    ekr[co] = ekr[co] + [rmse(m2[co], dummy, xdata=xtest, fdata=ftest)]
    ekm[co] = ekm[co] + [meane(m2[co], dummy, xdata=xtest, fdata=ftest)]
    
    co += 1



# Average out runs
ehrm = np.zeros(iters)
ehmm = np.zeros(iters)
ehsm = np.zeros(iters) 
ekrm = np.zeros(itersk)
ekmm = np.zeros(itersk)
eksm = np.zeros(itersk)

for i in range(nrunsk2):
    ehrm += np.array(ehr[i]).T[0]/nrunsk2
    ehmm += np.array(ehm[i]).T[0][0]/nrunsk2
    ehsm += np.array(ehm[i]).T[0][1]/nrunsk2
    for j in range(itersk):
        # if j == 1:
        #     if i >= nrunsk1:
        #         print("blah")
        #     else:
        #         ekrm[j] += np.array(ekr[i]).T[0][j]/nrunsk1
        #         ekmm[j] += np.array(ekm[i]).T[0][0][j]/nrunsk1
        #         eksm[j] += np.array(ekm[i]).T[0][1][j]/nrunsk1
        # else: 
        ekrm[j] += np.array(ekr[i]).T[0][j]/nrunsk2
        ekmm[j] += np.array(ekm[i]).T[0][0][j]/nrunsk2
        eksm[j] += np.array(ekm[i]).T[0][1][j]/nrunsk2


samplehist = np.zeros(iters, dtype=int)
samplehistk = np.zeros(itersk, dtype=int)

for i in range(iters-1):
    samplehist[i] = hi[0][i].ntr
samplehist[iters-1] = mf[0].training_points[None][0][0].shape[0]
for i in range(itersk):
    samplehistk[i] = 20 + i*50

plt.rcParams['font.size'] = '14'

plt.tricontour(xtest[:,0], xtest[:,1], ftest, levels=40)
plt.colorbar()
plt.xticks([23,27])
plt.yticks([0.36,0.51])
plt.xlabel(r'$\theta_s$')
plt.ylabel(r'$\kappa$')
plt.savefig(f"./{title}/trishock.png", bbox_inches="tight")
plt.clf()

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
plt.xlabel("Number of samples")
plt.ylabel("Relative error")
plt.xticks(ticks=np.arange(min(samplehist), max(samplehist), 40), labels=np.arange(min(samplehist), max(samplehist), 40) )
plt.grid()
ax.xaxis.set_minor_formatter(mticker.ScalarFormatter())
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.ticklabel_format(style='plain', axis='x')
#plt.legend(loc=1)
plt.savefig(f"./{title}/err_nrmse_ensemble.png", bbox_inches="tight")
plt.clf()

ax = plt.gca()
plt.loglog(samplehist, ehmm, "b--", label='AIGES Mean' )
plt.loglog(samplehistk, ekmm, 'k--', label='LHS Mean')
plt.loglog(samplehist, ehsm, "b-.", label='AIGES Std. Dev.' )
plt.loglog(samplehistk, eksm, 'k-.', label='LHS Std. Dev.')
plt.xlabel("Number of samples")
plt.ylabel("Relative error")
plt.xticks(ticks=np.arange(min(samplehist), max(samplehist), 40), labels=np.arange(min(samplehist), max(samplehist), 40) )
plt.grid()
ax.xaxis.set_minor_formatter(mticker.ScalarFormatter())
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.ticklabel_format(style='plain', axis='x')
#plt.legend(loc=1)
plt.savefig(f"./{title}/err_uq_ensemble.png", bbox_inches="tight")
plt.clf()

trx = mf[4].training_points[None][0][0]
# trf = modelbase.training_points[None][0][1]
# trf = np.squeeze(trf)
# trg = np.zeros_like(trx)
# for j in range(dim):
#     trg[:,j] = modelbase.training_points[None][j+1][1].flatten()
# plt.tricontour(trx[:,0], trx[:,1], trf, levels = 20)
# plt.colorbar()
# plt.savefig(f"./{title}/tritrain.png", bbox_inches="tight")#"tight")
# plt.clf()

m, n = trx.shape
normal = np.ones(dim)
normal /= np.linalg.norm(normal)


planedists = np.zeros(m)
for i in range(m):
    planedists[i] = abs(np.dot(trx[i,:],normal))
nref = xref.shape[0]


# # Plot points
if(dim == 2):
    bbox = Bbox([[0.0, 0], [6.5, 4.3]])
    # mk = copy.deepcopy(mf[0])
    # mk.set_training_values(x2[0], f2[0])
    # if(isinstance(mk, GEKPLS) or isinstance(mk, POUSurrogate)):
    #     for i in range(dim):
    #         mk.set_training_derivatives(x2[0], g2[0][:,i:i+1], i)
    # mk.train()
    plt.clf()
    nt0 = samplehist[0]
    # Plot Training Points
    plt.plot(trx[0:nt0,0], trx[0:nt0,1], "bo", label='Initial Samples')
    plt.plot(trx[nt0:,0], trx[nt0:,1], "ro", label='Adaptive Samples')
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    #plt.legend(loc=1)
    plt.savefig(f"./{title}/2d_aniso_pts.png", bbox_inches="tight")#"tight")
    plt.clf()
    
    # Plot Error Contour
    #Contour
    nref = xref.shape[0]
    xlimits = dummy.xlimits

    Za = np.zeros(nref)
    Zk = np.zeros(nref)
    F = np.zeros(nref)

    for i in range(nref):
        xi = np.zeros([1,2])
        xi[0,0] = xref[i,0]
        xi[0,1] = xref[i,1]
        F[i]  = modelbasea.predict_values(xi)
        Za[i] = abs(F[i] - ftest[i])
        Zk[i] = abs(m2[0].predict_values(xi) - ftest[i])


    cs = plt.tricontour(xtest[:,0], xtest[:,1], Za, levels = 12)
    plt.colorbar(cs, aspect=20)
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    #plt.legend(loc=1)
    plt.plot(trx[0:nt0,0], trx[0:nt0,1], "o", fillstyle='full', markerfacecolor='b', markeredgecolor='b', label='Initial Samples')
    plt.plot(trx[nt0:,0], trx[nt0:,1], "o", fillstyle='full', markerfacecolor='r', markeredgecolor='r', label='Adaptive Samples')
    plt.xlim([23.,27.])
    plt.ylim([0.36,0.51])
    plt.xticks([23,24,25,26,27])
    plt.savefig(f"./{title}/2d_errcona.png", bbox_inches="tight")

    plt.clf()

    # Plot Non-Adaptive Error
    plt.tricontour(xtest[:,0], xtest[:,1], Zk, levels = 12)
    plt.colorbar(cs, aspect=20)
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.plot(x2[0][:,0], x2[0][:,1], "o", fillstyle='full', markerfacecolor='b', markeredgecolor='b', label='LHS Samples')
    plt.xticks([23,24,25,26,27])
    plt.savefig(f"./{title}/2d_errconk.png", bbox_inches="tight")

    plt.clf()


meantrue = sum(fref)/nref
stdtrue = np.sqrt((sum(fref*fref)/nref) - (sum(fref)/nref)**2)

meanlhstrue = sum(f2[0])/f2[0].shape[0]
stdlhstrue = np.sqrt((sum(f2[0]*f2[0])/f2[0].shape[0]) - (sum(f2[0])/f2[0].shape[0])**2)

faiges = modelbasea.predict_values(xtest)
meanaiges = sum(faiges)/nref
stdaiges = np.sqrt((sum(faiges*faiges)/nref) - (sum(faiges)/nref)**2)

flhs = m2[0].predict_values(xtest)
meanlhs = sum(flhs)/nref
stdlhs  = np.sqrt((sum(flhs*flhs)/nref) - (sum(flhs)/nref)**2)


print("True Mean: ", meantrue)
print("True LHS Mean: ", meanlhstrue[0])
print("LHS Mean: ", meanlhs[0])
print("AIGES Mean: ", meanaiges[0])
print("True std: ", stdtrue)
print("True LHS std: ", stdlhstrue[0])
print("LHS std: ", stdlhs[0])
print("AIGES std: ", stdaiges[0])
