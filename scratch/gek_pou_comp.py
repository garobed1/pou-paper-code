import sys, os
import copy
import pickle
import math
from matplotlib.transforms import Bbox
sys.path.insert(1,"../surrogate")

import numpy as np
import matplotlib.pyplot as plt
from sutils import divide_cases
from error import rmse, meane

from hess_criteria import HessianRefine
from getxnew import getxnew, adaptivesampling
from example_problems import Peaks2D, QuadHadamard, MultiDimJump, MultiDimJumpTaper, FuhgP8, FuhgP9, FuhgP10, FakeShock
from smt.problems import TensorProduct
from smt.surrogate_models import KPLS, GEKPLS, KRG
from direct_gek import DGEK
#from smt.surrogate_models.rbf import RBF
from pougrad import POUSurrogate, POUHessian
import matplotlib as mpl
import matplotlib.ticker as mticker
from smt.sampling_methods import LHS
from defaults import DefaultOptOptions

D = False
adapt = True
loaddata = True
plt.rcParams['font.size'] = '14'
prob = "arctan"
corr = "squar_exp"
dim = 1

# Problem Settings
alpha = 8.       #arctangent jump strength
if(prob == "arctan"):
    trueFunc = MultiDimJump(ndim=dim, alpha=alpha)
elif(prob == "arctantaper"):
    trueFunc = MultiDimJumpTaper(ndim=dim, alpha=alpha)
elif(prob == "tensorexp"):
    trueFunc = TensorProduct(ndim=dim, func="gaussian")
else:
    raise ValueError("Given problem not valid.")
xlimits = trueFunc.xlimits
sampling = LHS(xlimits=xlimits, criterion='m')

Nerr = 5000
xtest = sampling(Nerr)
ftest = trueFunc(xtest)
testdata = [xtest, ftest]
numtest = 80
limits = [-5, 2]
theta_sample = np.logspace(limits[0], limits[1], num = numtest)

if(not loaddata):
    nt0 = 5
    ntr = 35
    xk = sampling(nt0)
    fk = trueFunc(xk)
    gk = np.zeros([nt0,dim])
    for j in range(dim):
        gk[:,j:j+1] = trueFunc(xk, j)

    if(adapt):
        model0 = POUHessian(bounds=xlimits)
        model0.set_training_values(xk, fk)
        model0.options.update({"print_global":False})
        neval=2
        rscale=2.5
        for i in range(dim):
            model0.set_training_derivatives(xk, gk[:,i:i+1], i)
        model0.train()
        options = DefaultOptOptions
        options["local"] = False
        options["errorcheck"] = None
        options["multistart"] = 2
        options["lmethod"] = 'L-BFGS-B'
        RC0 = HessianRefine(model0, gk, xlimits, neval=neval, rscale=rscale, improve=1, multistart=5)
        mf, rF, hf, ef, ef2 = adaptivesampling(trueFunc, model0, RC0, xlimits, ntr, options=options)
        nt0 = nt0 + ntr
        xk = mf.training_points[None][0][0]
        fk = mf.training_points[None][0][1]
        gk = np.zeros([nt0,dim])
        for j in range(dim):
            gk[:,j:j+1] = trueFunc(xk, j)

    else:
        nt0 += ntr
        xk = sampling(nt0)
        fk = trueFunc(xk)
        gk = np.zeros([nt0,dim])
        for j in range(dim):
            gk[:,j:j+1] = trueFunc(xk, j)

    with open(f'./cond_data.pickle', 'wb') as f:
        pickle.dump((xk, fk, gk), f)

else:
    with open(f'./cond_data.pickle', 'rb') as f:
        xk, fk, gk = pickle.load(f)
    nt0 = xk.shape[0]

if(not os.path.isfile("./thopt.pickle")):
    if(D):
        modelbase = DGEK(xlimits=xlimits)
        modelbase.options.update({"corr":corr})
        modelbase.options.update({"poly":"linear"})
        modelbase.options.update({"theta_bounds":[10**limits[0], 10**limits[1]]})
        # modelbase.options.update({"n_start":5})
        modelbase.set_training_values(xk, fk)
        for j in range(dim):
            modelbase.set_training_derivatives(xk, gk[:,j:j+1], j)

    else:
        modelbase = KRG()
        dx = 1e-4
        #modelgek.options.update({"hyper_opt":"TNC"})
        modelbase.options.update({"theta_bounds":[10**limits[0], 10**limits[1]]})
        modelbase.options.update({"corr":corr})
        modelbase.options.update({"poly":"linear"})
        nex = xk.shape[0]
        xaug = np.zeros([nex, 1])
        faug = np.zeros([nex, 1])
        for k in range(nex):
            xaug[k] = xk[k] + dx
            faug[k] = fk[k] + dx*gk[k]
        xtot = np.append(xk, xaug, axis=0)
        ftot = np.append(fk, faug, axis=0)
        modelbase.set_training_values(xtot, ftot)

    modelbase.train()
    modelbase.options.update({"print_global":False})

    modelbase2 = KRG()
    modelbase2.options.update({"corr":corr})
    modelbase2.options.update({"poly":"linear"})
    modelbase2.options.update({"theta_bounds":[10**limits[0], 10**limits[1]]})
    # modelbase.options.update({"n_start":5})
    modelbase2.set_training_values(xk, fk)
    modelbase2.train()
    modelbase2.options.update({"print_global":False})


modelpou = POUHessian(bounds=xlimits)
modelpou.options.update({"rho":100})
modelpou.set_training_values(xk, fk)
for j in range(dim):
    modelpou.set_training_derivatives(xk, gk[:,j:j+1], j)
modelpou.train()
modelpou.options.update({"print_global":False})

ndir = 150
xlimits = trueFunc.xlimits
x = np.linspace(xlimits[0][0], xlimits[0][1], ndir)
ZDGEK = np.zeros([ndir])
ZKRG = np.zeros([ndir])
FDGEK = np.zeros([ndir])
FKRG = np.zeros([ndir])
FPOU = np.zeros([ndir])
TF = np.zeros([ndir])
for i in range(ndir):
    xi = np.zeros([1,1])
    xi[0] = x[i]
    TF[i] = trueFunc(xi)
    FDGEK[i] = modelbase.predict_values(xi)
    FKRG[i] = modelbase2.predict_values(xi)
    FPOU[i] = modelpou.predict_values(xi)
    ZDGEK[i] = abs(FDGEK[i] - TF[i])
    ZKRG[i] = abs(FKRG[i] - TF[i])

trx = modelbase.training_points[None][0][0]
trf = modelbase.training_points[None][0][1]

# Plot Non-Adaptive Error

plt.plot(x, TF, 'k-', linewidth=1.2, label=f'Original')
plt.plot(x, FDGEK, linewidth=1.4, label=f'GEK')
# plt.yscale("log")
plt.xlabel(r"$x$")
plt.ylabel(r"$f(\mathbf{x})$")
plt.grid()
#plt.legend(loc=1)
plt.plot(trx[0:nt0,0], trf[0:nt0,0], "bo", ms=4 , label=f'Sample Locations')#min(np.min(ZKRG),np.min(ZDGEK))*np.ones_like(
plt.legend(loc=0)
plt.ylim(-2,2)
plt.savefig(f"./arctan_gek.pdf", bbox_inches="tight")
plt.clf()

plt.plot(x, TF, 'k-', linewidth=1.2, label=f'Original')
plt.plot(x, FPOU, linewidth=1.4, label=f'POU')
plt.xlabel(r"$x$")
plt.ylabel(r"$f(\mathbf{x})$")
plt.grid()
#plt.legend(loc=1)
plt.plot(trx[0:nt0,0], trf[0:nt0,0], "bo", ms=4 , label=f'Sample Locations')#min(np.min(ZKRG),np.min(ZDGEK))*np.ones_like(
plt.legend(loc=0)
plt.ylim(-2,2)
plt.savefig(f"./arctan_pou.pdf", bbox_inches="tight")
plt.clf()
# plt.savefig(f"./gek_issues.{save_format}", bbox_inches="tight")





