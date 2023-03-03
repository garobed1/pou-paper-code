import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import openmdao.api as om
from scratch.stat_comp_comp import StatCompComponent
from optimization.opt_subproblem import SequentialFullSolve

"""
run a mean plus variance optimization over the 1D-1D test function, pure LHS
for now using the subproblem idea
"""

# from optimization.optimizers import optimize
from pyoptsparse import Optimization, SLSQP
from functions.problem_picker import GetProblem
from utils.error import stat_comp
from optimization.robust_objective import RobustSampler
from optimization.defaults import DefaultOptOptions

plt.rcParams['font.size'] = '22'

# set up robust objective UQ comp parameters
u_dim = 1
eta_use = 1.0
N_t = 5000*u_dim
N_m = 50
jump = 50

# set up beta test problem parameters
dim = 2
prob = 'betatestex'
# pdfs = ['uniform', 0.] # replace 2nd arg with the current design var

func = GetProblem(prob, dim)
xlimits = func.xlimits
# start at just one point for now
x_init = 5.


# run optimizations
x_init = 5.
# args = (func, eta_use)
xlimits_d = np.zeros([1,2])
xlimits_d[:,1] = 10.

pdfs = [['beta', 3., 1.], 0.] # replace 2nd arg with the current design var

max_outer = 10
opt_settings = {}
opt_settings['ACC'] = 1e-8

# start distinguishing between model and truth here

### TRUTH ###
sampler_t = RobustSampler(np.array([x_init]), N=N_t, xlimits=xlimits, probability_functions=pdfs, retain_uncertain_points=True)
probt = om.Problem()
probt.model.add_subsystem("stat", StatCompComponent(sampler=sampler_t,
                                 stat_type="mu_sigma", 
                                 pdfs=pdfs, 
                                 eta=eta_use, 
                                 func=func))
# doesn't need a driver

probt.driver = om.pyOptSparseDriver() #Default: SLSQP
probt.driver.opt_settings = opt_settings
probt.model.add_design_var("stat.x_d", lower=xlimits_d[0,0], upper=xlimits_d[0,1])
probt.model.add_objective("stat.musigma")
probt.setup()
probt.set_val("stat.x_d", x_init)
probt.run_model()

### MODEL ###
sampler_m = RobustSampler(np.array([x_init]), N=N_m, xlimits=xlimits, probability_functions=pdfs, retain_uncertain_points=True)
probm = om.Problem()
probm.model.add_subsystem("stat", StatCompComponent(sampler=sampler_m,
                                 stat_type="mu_sigma", 
                                 pdfs=pdfs, 
                                 eta=eta_use, 
                                 func=func))
probm.driver = om.pyOptSparseDriver() #Default: SLSQP
probt.driver.opt_settings = opt_settings
probm.model.add_design_var("stat.x_d", lower=xlimits_d[0,0], upper=xlimits_d[0,1])
probm.model.add_objective("stat.musigma")
probm.setup()
probm.set_val("stat.x_d", x_init)
probm.run_model()

sub_optimizer = SequentialFullSolve(prob_model=probm, prob_truth=probt, flat_refinement=jump, max_iter=max_outer)
sub_optimizer.setup_optimization()
sub_optimizer.solve_full()

x_opt_1 = sub_optimizer.zk #prob.get_val("stat.x_d")[0]

# plot robust func
ndir = 150
x = np.linspace(xlimits[1][0], xlimits[1][1], ndir)
y = np.zeros([ndir])
for j in range(ndir):
    prob.set_val("stat.x_d", x[j])
    prob.run_model()
    y[j] = prob.get_val("stat.musigma")

# Plot original function
cs = plt.plot(x, y)
plt.xlabel(r"$x_d$")
plt.ylabel(r"$\mu_f(x_d)$")
plt.axvline(x_init, color='k', linestyle='--', linewidth=1.2)
plt.axvline(x_opt_1, color='r', linestyle='--', linewidth=1.2)
#plt.legend(loc=1)
plt.savefig(f"./robust_opt_subopt_plots/objrobust1_true.pdf", bbox_inches="tight")
plt.clf()

# plot beta dist
x = np.linspace(xlimits[0][0], xlimits[0][1], ndir)
y = np.zeros([ndir])
beta_handle = beta(pdfs[0][1],pdfs[0][2])
for j in range(ndir):
    y[j] = beta_handle.pdf(x[j])
cs = plt.plot(x, y)
plt.xlabel(r"$x_d$")
plt.ylabel(r"$\mu_f(x_d)$")
plt.axvline(0.75, color='r', linestyle='--', linewidth=1.2)
#plt.legend(loc=1)
plt.savefig(f"./robust_opt_subopt_plots/betadist1_true.pdf", bbox_inches="tight")
plt.clf()




