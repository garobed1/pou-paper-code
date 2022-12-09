from ast import Mult
import sys, os
import copy
import pickle
from matplotlib.transforms import Bbox
sys.path.insert(1,"../surrogate")

import numpy as np
import matplotlib.pyplot as plt
from sutils import divide_cases
from error import rmse, meane

from hess_criteria import HessianRefine
from getxnew import getxnew, adaptivesampling
from example_problems import Peaks2D, QuadHadamard, MultiDimJump, MultiDimJumpTaper, FuhgP8, FuhgP9, FuhgP10, FakeShock, Sine1D
from smt.problems import TensorProduct, Rosenbrock, WingWeight
from smt.surrogate_models import KPLS, GEKPLS, KRG
from direct_gek import DGEK
#from smt.surrogate_models.rbf import RBF
from pougrad import POUSurrogate, POUHessian
import matplotlib as mpl
import matplotlib.ticker as mticker
from smt.sampling_methods import LHS
from scipy.stats.qmc import Halton, scale
from defaults import DefaultOptOptions

adapt = True
plt.rcParams['font.size'] = '13'

trueFunc = [MultiDimJump(ndim=1), Rosenbrock(ndim=2), MultiDimJump(ndim=6), WingWeight(ndim=10)]
dim = [1, 2, 6, 10]
nprob = len(trueFunc)
xlimits = []
for i in range(nprob):
    xlimits.append(trueFunc[i].xlimits)

rlimits = [0, 3]
numtest = 20
rho_sample = np.logspace(rlimits[0], rlimits[1], num = numtest)
sampling = [Halton(dim[0]), Halton(dim[1]), Halton(dim[2]), Halton(dim[3])]

nt = [[item * 10 for item in dim], [item * 20 for item in dim], [item * 30 for item in dim]]
nsets = len(nt)

xk = []
fk = []
gk = []
model0 = [[POUHessian(bounds=xlimits[0]), POUHessian(bounds=xlimits[0]), POUHessian(bounds=xlimits[0])], \
          [POUHessian(bounds=xlimits[1]), POUHessian(bounds=xlimits[1]), POUHessian(bounds=xlimits[1])], \
          [POUHessian(bounds=xlimits[2]), POUHessian(bounds=xlimits[2]), POUHessian(bounds=xlimits[2])], \
          [POUHessian(bounds=xlimits[3]), POUHessian(bounds=xlimits[3]), POUHessian(bounds=xlimits[3])]]


Nerr = 3000
xtest = [scale(sampling[0].random(Nerr), xlimits[0][:,0], xlimits[0][:,1]), 
        scale(sampling[1].random(Nerr), xlimits[1][:,0], xlimits[1][:,1]), 
        scale(sampling[2].random(Nerr), xlimits[2][:,0], xlimits[2][:,1]), 
        scale(sampling[3].random(Nerr), xlimits[3][:,0], xlimits[3][:,1])]
ftest = [trueFunc[0](xtest[0]), trueFunc[1](xtest[1]), trueFunc[2](xtest[2]), trueFunc[3](xtest[3])]

sampling[0].reset()
sampling[1].reset()
sampling[2].reset()
sampling[3].reset()

for i in range(nprob):
    xk.append([])
    fk.append([])
    gk.append([])
    for j in range(nsets):
        xk[i].append(scale(sampling[i].random(nt[j][i]), xlimits[i][:,0], xlimits[i][:,1]))
        sampling[i].reset()
        fk[i].append(trueFunc[i](xk[i][j]))
        gk[i].append(np.zeros([nt[j][i],dim[i]]))
        for k in range(dim[i]):
            gk[i][j][:,k:k+1] = trueFunc[i](xk[i][j], k)

        model0[i][j].options.update({"neval":int(np.ceil((dim[i]+2)))+1})
        model0[i][j].set_training_values(xk[i][j], fk[i][j])
        model0[i][j].options.update({"print_global":False})
        for k in range(dim[i]):
            model0[i][j].set_training_derivatives(xk[i][j], gk[i][j][:,k:k+1], k)
        model0[i][j].train()



ndir = 150
x = np.linspace(xlimits[1][0][0], xlimits[1][0][1], ndir)
y = np.linspace(xlimits[1][1][0], xlimits[1][1][1], ndir)
X, Y = np.meshgrid(x, y)
F = np.zeros([ndir, ndir])
Zk = np.zeros([ndir, ndir])
TF = np.zeros([ndir, ndir])
model0[1][1].options.update({"rho":120})
for i in range(ndir):
    for j in range(ndir):
        xi = np.zeros([1,2])
        xi[0,0] = x[i]
        xi[0,1] = y[j]
        F[j,i]  = model0[1][1].predict_values(xi)
        TF[j,i] = trueFunc[1](xi)
        Zk[j,i] = abs(F[j,i] - TF[j,i])

plt.contourf(X, Y, Zk, levels = 40)
plt.colorbar()
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.plot(xk[1][1][:,0], xk[1][1][:,1], "o", fillstyle='full', markerfacecolor='b', markeredgecolor='b', label='LHS Samples')
plt.savefig(f"./pou_rho_plots_issue.pdf", bbox_inches="tight")
plt.clf()







errs = []
for i in range(nprob):
    errs.append([])
    for j in range(nsets):
        errs[i].append([])
        for k in range(numtest):
            model0[i][j].options.update({"rho": rho_sample[k]})
            errs[i][j].append(rmse(model0[i][j], trueFunc[i], N=Nerr, xdata=xtest[i], fdata=ftest[i]))
            print(f"{i}, {j}, {k}")
    # Plot Non-Adaptive Error
    ax1 = plt.subplot(211)
    for j in range(nsets):
        ropt = 5.5*pow(nt[j][i], 1./dim[i])
        p = plt.plot(rho_sample, errs[i][j], marker='s', linewidth=1.4, label=fr'${nt[j][i]}$ Samples')
        plt.axvline(ropt, color=p[0].get_color(), linestyle='--', linewidth=1.2)

    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    
    plt.xlabel(r"$\rho$")
    plt.ylabel(r"NRMSE")

    plt.savefig(f"./pou_rho_example_{i}.pdf", bbox_inches="tight")
    plt.clf()


with open(f'./rho_sample.pickle', 'wb') as f:
    pickle.dump(rho_sample, f)
with open(f'./rho_errs.pickle', 'wb') as f:
    pickle.dump(errs, f)



