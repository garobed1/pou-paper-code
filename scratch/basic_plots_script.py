import sys, os
import copy
import pickle
sys.path.insert(1,"../surrogate")

import numpy as np
import matplotlib.pyplot as plt
from refinecriteria import looCV, HessianFit, TEAD
from hess_criteria import HessianRefine, POUSSA
from aniso_criteria import AnisotropicRefine
from getxnew import getxnew, adaptivesampling
from defaults import DefaultOptOptions
from sutils import divide_cases
from error import rmse, meane
from pougrad import POUSurrogate, POUHessian


from example_problems import  QuadHadamard, MultiDimJump, MultiDimJumpTaper,FuhgP3, FuhgSingleHump, FuhgP8, FuhgP9, FuhgP10
from smt.problems import Branin, Sphere, LpNorm, Rosenbrock, WaterFlow, WeldedBeam, RobotArm, CantileverBeam
from smt.surrogate_models import KPLS, GEKPLS, KRG
#from smt.surrogate_models.rbf import RBF
from pougrad import POUSurrogate
from smt.sampling_methods import LHS

# 
dim = 1
prob = "arctan"


# Problem Settings
alpha = 8.       #arctangent jump strength
#trueFunc = MultiDimJumpTaper(ndim=dim, alpha=alpha)
trueFunc = FuhgSingleHump(ndim=dim)


ndir = 500
xlimits = trueFunc.xlimits
sampling = LHS(xlimits=xlimits, criterion='m')
#x = sampling(ndir)
x = np.linspace(xlimits[0][0], xlimits[0][1], ndir)
F = trueFunc(x)

m0 = 6
mf = 12

#x0 = np.linspace(xlimits[0][0], xlimits[0][1], m0)
x0 = sampling(m0)
y0dum = -1.5*np.ones(m0)
y0 = trueFunc(x0)
g0 = np.zeros([m0,dim])
g0[:,0:1] = trueFunc(x0,0)

model0 = POUHessian(bounds=xlimits, rscale=5.5)
#model0 = KRG()
model0.set_training_values(x0, y0)
model0.set_training_derivatives(x0, g0, 0)
model0.train()
f0 = model0.predict_values(x)

#xk = np.linspace(xlimits[0][0], xlimits[0][1], mf)
xk = sampling(mf)
ykdum = -1.5*np.ones(mf)
yk = trueFunc(xk)
gk = np.zeros([mf,dim])
gk[:,0:1] = trueFunc(xk,0)
modelk = POUHessian(bounds=xlimits, rscale=5.5)
# modelk = KRG()
modelk.set_training_values(xk, yk)
modelk.set_training_derivatives(xk, gk, 0)
modelk.train()
fk = modelk.predict_values(x)
ek = rmse(modelk, trueFunc, N=5000)

RC = HessianRefine(model0, grad=g0, bounds=xlimits, rscale=5.5, print_rc_plots=False)
#RC = TEAD(model0, grad=g0, bounds=xlimits)

options = DefaultOptOptions
options["local"] = False
options["localswitch"] = True
options["multistart"] = 2
options["lmethod"] = 'L-BFGS-B'

modelf, RF, hf, ef, ef2 = adaptivesampling(trueFunc, model0, RC, xlimits, mf-m0, options=options)

xf = modelf.training_points[None][0][0]
yf = modelf.training_points[None][0][1]
yfdum = -1.5*np.ones(mf)
ff = modelf.predict_values(x)
ef = rmse(modelf, trueFunc, N=5000)

#plot 
plt.rcParams['font.size'] = '13'
plt.figure(figsize=(6.4, 3.4))

ax = plt.gca()
plt.plot(x, F, "k-",  label="Original")
plt.plot(x, fk, "b--",  label="POU")
plt.plot(xk, ykdum, "bo",  label="Non-Adaptive Points")
plt.plot(xk, yk, "bo")
plt.ylim(top = 21.5, bottom= -2.5)
plt.grid()
plt.xlabel(r"$x$")
plt.ylabel(r"$f(x)$")
plt.xticks(xlimits[:][0])
#plt.yticks([-1.0, -0.5, -0.0, 0.5, 1.0])
plt.legend()

ax.text(0.0, 1.25, f'NRMSE = {ek}', color='black', 
        bbox=dict(facecolor='none', edgecolor='black'))

plt.savefig(f"./nonadapt.pdf", bbox_inches="tight")
plt.clf()

# import pdb; pdb.set_trace()
ax = plt.gca()
plt.plot(x, F, "k-",  label="Original")
plt.plot(x, ff, "b--",  label="POU")
plt.plot(x0, y0dum, "bo",  label="Initial Points")
plt.plot(x0, y0, "bo")
plt.plot(xf[m0:mf], yfdum[m0:mf], "ro",  label="Adaptive Points")
plt.plot(xf[m0:mf], yf[m0:mf], "ro")
plt.ylim(top = 21.5, bottom= -2.5)
plt.grid()
plt.xlabel(r"$x$")
plt.ylabel(r"$f(x)$")
plt.xticks(xlimits[:][0])
#plt.yticks([-1.0, -0.5, -0.0, 0.5, 1.0])
plt.legend()
ax.text(0.0, 1.25, f'NRMSE = {ef}', color='black', 
        bbox=dict(facecolor='none', edgecolor='black'))
plt.savefig(f"./adapt.pdf", bbox_inches="tight")
plt.clf()


#import pdb; pdb.set_trace()
