import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
"""
run a mean plus variance optimization over the 1D-1D test function, pure MC
for now
"""

from optimization.optimizers import optimize
from functions.problem_picker import GetProblem
from utils.error import stat_comp
from optimization.robust_objective import RobustSampler
from optimization.defaults import DefaultOptOptions

plt.rcParams['font.size'] = '16'

# set up robust objective UQ comp parameters
u_dim = 1
eta_use = 1.0
N = 5000*u_dim


# set up beta test problem parameters
dim = 2
prob = 'betatestex'
pdfs = [['beta', 3., 1.], 0.] # replace 2nd arg with the current design var
# pdfs = ['uniform', 0.] # replace 2nd arg with the current design var

func = GetProblem(prob, dim)
xlimits = func.xlimits
# start at just one point for now
x_init = 5.

sampler = RobustSampler(np.array([x_init]), N, xlimits=xlimits, probability_functions=pdfs, retain_uncertain_points=True)

xds = []
func_calls = []
objs = []


# eta*mean + (1-eta)stdev
def objRobust(x, func, eta = 0.5):

    pdfs[1] = x
    sampler.set_design(np.array([x]))
    sampler.generate_uncertain_points(N)
    res = stat_comp(None, func, stat_type="mu_sigma", pdfs=pdfs, xdata=sampler)
    fm = res[0]
    fs = res[1]

    xds.append(x)
    objs.append(fm)
    if len(func_calls):
        func_calls.append(N + func_calls[-1])
    else:
        func_calls.append(N)

    return eta*fm + (1-eta)*fs
# eta*mean + (1-eta)stdev


# need to get grad
def objRobustGrad(x, func, eta = 0.5):

    pdfs[1] = x
    sampler.set_design(np.array([x]))
    sampler.generate_uncertain_points(N)
    gres = stat_comp(None, func, stat_type="mu_sigma", pdfs=pdfs, xdata=sampler, get_grad=True)
    gm = gres[0]
    gs = gres[1]

    return eta*gm + (1-eta)*gs


# test deriv
# h = 1e-8
# eta_use = 0.5
# fd0 = objRobust(x_init, func, eta_use)
# ad = objRobustGrad(x_init, func, eta_use)
# fd1 = objRobust(x_init+h, func, eta_use)

# fd = (fd1-fd0)/h
# import pdb; pdb.set_trace()


# run optimizations
options = DefaultOptOptions
options['lmethod'] = 'L-BFGS-B'

x_init = 5.
args = (func, eta_use)
xlimits_d = np.zeros([1,2])
xlimits_d[:,1] = 10.

results1 = optimize(objRobust, args=args, bounds=xlimits_d, type="local", jac=objRobustGrad, x0=x_init, options=options)

# plot robust func


plt.plot(func_calls, objs, linestyle='-', marker='s', label='convergence')
plt.xlabel(r"Function Calls")
plt.ylabel(r"$\mu_f(x_d)$")

plt.savefig(f"./robust_opt_plots/convrobust1_true.pdf", bbox_inches="tight")
plt.clf()

cs = plt.plot(xds, objs, 'b-', marker='s', label='convergence')

plt.legend()
plt.savefig(f"./robust_opt_plots/objrobustwconv1_true.pdf", bbox_inches="tight")

ndir = 150
x = np.linspace(xlimits[1][0], xlimits[1][1], ndir)
y = np.zeros([ndir])
for j in range(ndir):
    y[j] = objRobust(x[j], func, eta_use)
# Plot original function
plt.plot(x, y, '-k', label='objective')

plt.xlabel(r"$x_d$")
plt.ylabel(r"$\mu_f(x_d)$")

plt.axvline(x_init, color='k', linestyle='--', linewidth=1.2)
plt.axvline(results1.x, color='r', linestyle='--', linewidth=1.2)


plt.legend()
plt.savefig(f"./robust_opt_plots/objrobust1_true.pdf", bbox_inches="tight")



plt.clf()





# plot beta dist
x = np.linspace(xlimits[0][0], xlimits[0][1], ndir)
y = np.zeros([ndir])
beta_handle = beta(pdfs[0][1],pdfs[0][2])
for j in range(ndir):
    y[j] = beta_handle.pdf(x[j])
cs = plt.plot(x, y)
plt.xlabel(r"$x_u$")
plt.ylabel(r"$P(x_u;3,1)$")
plt.axvline(0.75, color='r', linestyle='--', linewidth=1.2)
#plt.legend(loc=1)
plt.savefig(f"./robust_opt_plots/betadist1_true.pdf", bbox_inches="tight")
plt.clf()



import pdb; pdb.set_trace()




# change distribution a bit
pdfs = [['beta', 1., 3.], 0.]
sampler = RobustSampler(np.array([x_init]), N, xlimits=xlimits, probability_functions=pdfs)

results2 = optimize(objRobust, args=args, bounds=xlimits_d, type="local", jac=objRobustGrad, x0=x_init, options=options)

# fun plots
ndir = 150
x = np.linspace(xlimits[0][0], xlimits[0][1], ndir)
y = np.linspace(xlimits[1][0], xlimits[1][1], ndir)
X, Y = np.meshgrid(x, y)
TF = np.zeros([ndir, ndir])
for i in range(ndir):
    for j in range(ndir):
        xi = np.zeros([1,2])
        xi[0,0] = x[i]
        xi[0,1] = y[j]
        TF[j,i] = func(xi)
# Plot original function
cs = plt.contourf(X, Y, TF, levels = 40)
plt.colorbar(cs, aspect=20, label = r"$f(x_u, x_d)$")
plt.xlabel(r"$x_u$")
plt.ylabel(r"$x_d$")
#plt.legend(loc=1)
plt.savefig(f"./robust_opt_plots/betarobust_true.pdf", bbox_inches="tight")
plt.clf()

# plot robust func
x = np.linspace(xlimits[1][0], xlimits[1][1], ndir)
y = np.zeros([ndir])
for j in range(ndir):
    y[j] = objRobust(x[j], func, eta_use)
# Plot original function
cs = plt.plot(x, y)
plt.xlabel(r"$x_d$")
plt.ylabel(r"$\mu_f(x_d)$")
plt.axvline(x_init, color='k', linestyle='--', linewidth=1.2)
plt.axvline(results2.x, color='r', linestyle='--', linewidth=1.2)
#plt.legend(loc=1)
plt.savefig(f"./robust_opt_plots/objrobust2_true.pdf", bbox_inches="tight")
plt.clf()

x = np.linspace(xlimits[0][0], xlimits[0][1], ndir)
y = np.zeros([ndir])
beta_handle = beta(pdfs[0][1],pdfs[0][2])
for j in range(ndir):
    y[j] = beta_handle.pdf(x[j])
cs = plt.plot(x, y)

plt.xlabel(r"$x_d$")
plt.ylabel(r"$\mu_f(x_d)$")
plt.axvline(0.25, color='r', linestyle='--', linewidth=1.2)
#plt.legend(loc=1)
plt.savefig(f"./robust_opt_plots/betadist2_true.pdf", bbox_inches="tight")
plt.clf()

import pdb; pdb.set_trace()