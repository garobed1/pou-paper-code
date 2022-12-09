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
plt.rcParams['font.size'] = '13'
prob = "sine"
dim = 1

# Problem Settings
if(prob == "sine"):
    trueFunc = Sine1D(ndim=dim)
else:
    raise ValueError("Given problem not valid.")
xlimits = trueFunc.xlimits

limits = [1, 3]
numtest = limits[1] - limits[0] + 1
numtest = 5
rho_sample = np.logspace(limits[0], limits[1], num = numtest)
numtest = 4
rho_sample = [10, 50, 200, 1000]

nt1 = 2
nt2 = 2*nt1
nt0 = nt1 + nt2

xk1 = np.linspace(xlimits[0][0], 0.5*xlimits[0][1], num=nt1, endpoint=False)
xk2 = np.linspace(0.5*xlimits[0][1], xlimits[0][1], num=nt2)
xk = np.append(xk1, xk2)
xb = np.zeros(nt0 - 1)

for i in range(nt0 - 1):
    xb[i] = xk[i] + 0.5*(xk[i+1] - xk[i])


fk = trueFunc(xk)
gk = np.zeros([nt0,dim])
for j in range(dim):
    gk[:,j:j+1] = trueFunc(xk, j)

model0 = POUHessian(bounds=xlimits)
model0.set_training_values(xk, fk)
model0.options.update({"neval":int(np.ceil((dim+2)/2))})
model0.options.update({"print_global":False})
neval=2
for i in range(dim):
    model0.set_training_derivatives(xk, gk[:,i:i+1], i)
model0.train()


ndir = 2250
ys = []
x = np.linspace(xlimits[0][0], xlimits[0][1], ndir)
for i in range(numtest):
    rhoi = rho_sample[i]
    model0.options.update({"rho":rhoi})
    ys.append(model0.predict_values(x))

TF = trueFunc(x)

trx = model0.training_points[None][0][0]
trf = model0.training_points[None][0][1]

# Plot Non-Adaptive Error
ax = plt.gca()
plt.plot(x, TF, color='k', label=r'Original')[0]
for i in range(numtest):
    plt.plot(x, ys[i], linewidth=1.4, label=fr'POU, $\rho = {rho_sample[i]:4.1f}$')
plt.plot(trx, trf,"bo", ms=5, label="Sample Points" )
plt.legend()

for j in range(nt0 - 1):
    plt.axvline(xb[j], color='k', linestyle='--', linewidth=0.7)

plt.axhline(0., color='k', linestyle='-', linewidth=0.5)

plt.xlabel(r"$x$")
plt.ylabel(r"$f(x)$")



# inset axes
axins = ax.inset_axes([0.55, 0.55, 0.43, 0.43])

# replot
axins.plot(x, TF, color='k')
for i in range(numtest):
    axins.plot(x, ys[i], linewidth=1.4)
for j in range(nt0 - 1):
    axins.axvline(xb[j], color='k', linestyle='--', linewidth=0.7)
# sub region of the original image
#x1, x2, y1, y2 = 0.6, 1.0, 0.5, 0.9
x1, x2, y1, y2 = 2.2, 2.5, 0.6, 1.0
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_xticklabels([])
axins.set_yticklabels([])

ax.indicate_inset_zoom(axins, edgecolor="black")


#plt.legend(loc=1)
plt.savefig(f"./pou_rho_example.pdf", bbox_inches="tight")
plt.clf()

# plt.savefig(f"./gek_issues.{save_format}", bbox_inches="tight")





