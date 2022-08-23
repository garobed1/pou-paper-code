import sys, os
import copy
import pickle
from matplotlib.transforms import Bbox
sys.path.insert(1,"../surrogate")

import numpy as np
import matplotlib.pyplot as plt
from sutils import divide_cases
from error import rmse, meane

from example_problems import Peaks2D, QuadHadamard, MultiDimJump, MultiDimJumpTaper, FuhgP8, FuhgP9, FuhgP10, FakeShock
from smt.problems import Branin, Sphere, LpNorm, Rosenbrock, WaterFlow, WeldedBeam, RobotArm, CantileverBeam, WingWeight
from smt.surrogate_models import KPLS, GEKPLS, KRG
from direct_gek import DGEK
#from smt.surrogate_models.rbf import RBF
from pougrad import POUSurrogate, POUHessian
import matplotlib as mpl
import matplotlib.ticker as mticker
from smt.sampling_methods import LHS

save_format = "svg"
plt.rcParams['font.size'] = '14'
prob = "arctan"
dim = 1

# Problem Settings
alpha = 8.       #arctangent jump strength
if(prob == "arctan"):
    trueFunc = MultiDimJump(ndim=dim, alpha=alpha)
elif(prob == "arctantaper"):
    trueFunc = MultiDimJumpTaper(ndim=dim, alpha=alpha)
else:
    raise ValueError("Given problem not valid.")
xlimits = trueFunc.xlimits
sampling = LHS(xlimits=xlimits, criterion='m')

nt0 = 35
numtest = 50
limits = [-5, 2]
theta_sample = np.logspace(limits[0], limits[1], num = numtest)
xk = sampling(nt0)
fk = trueFunc(xk)
gk = np.zeros([nt0,dim])
for j in range(dim):
    gk[:,j:j+1] = trueFunc(xk, j)

if(not os.path.isfile("./thopt.pickle")):
    modelbase = DGEK(xlimits=xlimits)
    modelbase.options.update({"corr":"matern32"})
    modelbase.options.update({"poly":"linear"})
    modelbase.options.update({"theta_bounds":[10**limits[0], 10**limits[1]]})
    # modelbase.options.update({"n_start":5})
    modelbase.set_training_values(xk, fk)
    for j in range(dim):
        modelbase.set_training_derivatives(xk, gk[:,j:j+1], j)
    modelbase.train()

    thopt = modelbase.optimal_theta

    # get condition numbers and maximum likelihood values
    rlike = np.zeros_like(theta_sample)
    conds = np.zeros_like(theta_sample)
    for i in range(numtest):
        th = theta_sample[i]
        # modelbase.options.update({"theta0":[th]})
        rlike[i], par = modelbase._reduced_likelihood_function(np.array([th]))
        conds[i] = par["cond"]
    
    with open(f'./thopt.pickle', 'wb') as f:
        pickle.dump(thopt, f)
    with open(f'./rlike.pickle', 'wb') as f:
        pickle.dump(rlike, f)
    with open(f'./conds.pickle', 'wb') as f:
        pickle.dump(conds, f)


else: 
    with open(f'./thopt.pickle', 'rb') as f:
        thopt = pickle.load(f)
    with open(f'./rlike.pickle', 'rb') as f:
        rlike = pickle.load(f)
    with open(f'./conds.pickle', 'rb') as f:
        conds = pickle.load(f)

import pdb; pdb.set_trace()

#NRMSE
ax1 = plt.subplot(211)
plt.plot(theta_sample, rlike)
plt.axvline(thopt, color='r', linestyle='--')
plt.xscale("log")
plt.ylabel(r"Reduced Likelihood")
plt.xlim(theta_sample[0], theta_sample[-1])
plt.tick_params('x', labelsize=14)
plt.tick_params('x', labelbottom=False)

# share x only
ax2 = plt.subplot(212, sharex=ax1)
plt.loglog(theta_sample, conds)
plt.axvline(thopt, color='r', linestyle='--')
plt.xscale("log")
theta_ticks = np.logspace(limits[0], limits[1], num = limits[1] - limits[0] + 1)
plt.xticks(theta_ticks)
plt.ylabel(r"$\mathsf{R}$ Condition No.")
plt.xlabel(r"Kriging $\theta$")

plt.savefig(f"./gek_issues.{save_format}", bbox_inches="tight")
plt.savefig(f"./gek_issues.png", bbox_inches="tight")




