import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import openmdao.api as om
from scratch.stat_comp_comp import StatCompComponent
from optimization.opt_subproblem import SequentialFullSolve
from surrogate.pougrad import POUSurrogate, POUHessian
import os, copy
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

name = 'betaex_no_surrogate'

if not os.path.isdir(f"{name}"):
    os.mkdir(f"{name}")

# surrogate
use_surrogate = False
full_surrogate = True
use_truth_to_train = False #NEW
print_plots = True

# set up robust objective UQ comp parameters
u_dim = 1
eta_use = 1.0
N_t = 5000*u_dim
# N_t = 100*u_dim
N_m = 10
jump = 10
retain_uncertain_points = False
external_only = use_truth_to_train #NEW

# set up beta test problem parameters
dim = 2
prob = 'betatestex'
# pdfs = ['uniform', 0.] # replace 2nd arg with the current design var

func = GetProblem(prob, dim)
xlimits = func.xlimits
# start at just one point for now


# optimization
x_init = 5.
# args = (func, eta_use)
xlimits_d = np.zeros([1,2])
xlimits_d[:,0] = xlimits[1,0]
xlimits_d[:,1] = xlimits[1,1]
xlimits_u = np.zeros([u_dim,2])
xlimits_u[:,1] = xlimits[0,0]
xlimits_u[:,1] = xlimits[0,1]

# set up surrogates #NOTE: not doing it for truth for now
msur = None
if use_surrogate:
    rscale = 5.5
    rho = 10 

    if(full_surrogate):
        neval = 1+(dim+2)
        msur = POUHessian(bounds=xlimits)
    else:
        neval = 1+(u_dim+2)
        msur = POUHessian(bounds=xlimits_u)

    msur.options.update({"rscale":rscale})
    msur.options.update({"rho":rho})
    msur.options.update({"neval":neval})
    msur.options.update({"print_prediction":False})

pdfs = [['beta', 3., 1.], 0.] # replace 2nd arg with the current design var

max_outer = 20
opt_settings = {}
opt_settings['ACC'] = 1e-6

# start distinguishing between model and truth here

### TRUTH ###
sampler_t = RobustSampler(np.array([x_init]), N=N_t, 
                          name='truth', 
                          xlimits=xlimits, 
                          probability_functions=pdfs, 
                          retain_uncertain_points=retain_uncertain_points)
probt = om.Problem()
probt.model.add_subsystem("stat", StatCompComponent(sampler=sampler_t,
                                 stat_type="mu_sigma", 
                                 pdfs=pdfs, 
                                 eta=eta_use, 
                                 func=func,
                                 name=name))
# doesn't need a driver

# probt.driver = om.pyOptSparseDriver(optimizer= 'SNOPT') #Default: SLSQP
# probt.driver.opt_settings = opt_settings
probt.driver = om.ScipyOptimizeDriver(optimizer='L-BFGS-B') 
probt.model.add_design_var("stat.x_d", lower=xlimits_d[0,0], upper=xlimits_d[0,1])
probt.model.add_objective("stat.musigma")
probt.setup()
probt.set_val("stat.x_d", x_init)
# probt.run_model()
probt.run_driver()

x_opt_true = copy.deepcopy(probt.get_val("stat.x_d")[0])

# plot conv
cs = plt.plot(probt.model.stat.func_calls, probt.model.stat.objs)
plt.xlabel(r"Number of function calls")
plt.ylabel(r"$\mu_f(x_d)$")
#plt.legend(loc=1)
plt.savefig(f"./{name}/convergence_truth.pdf", bbox_inches="tight")
plt.clf()

true_fm = copy.deepcopy(probt.model.stat.objs[-1])

probt.set_val("stat.x_d", x_init)

### MODEL ###
sampler_m = RobustSampler(np.array([x_init]), N=N_m, 
                          name='model', 
                          xlimits=xlimits, 
                          probability_functions=pdfs, 
                          retain_uncertain_points=retain_uncertain_points,
                          external_only=external_only)
probm = om.Problem()
probm.model.add_subsystem("stat", StatCompComponent(sampler=sampler_m,
                                 stat_type="mu_sigma", 
                                 pdfs=pdfs, 
                                 eta=eta_use, 
                                 func=func,
                                 surrogate=msur,
                                 full_space=full_surrogate,
                                 name=name,
                                 print_surr_plots=print_plots))
# probm.driver = om.pyOptSparseDriver(optimizer='SNOPT') #Default: SLSQP
# probm.driver.opt_settings = opt_settings
probm.driver = om.ScipyOptimizeDriver(optimizer='L-BFGS-B') 
probm.model.add_design_var("stat.x_d", lower=xlimits_d[0,0], upper=xlimits_d[0,1])
probm.model.add_objective("stat.musigma")
probm.setup()
probm.set_val("stat.x_d", x_init)
probm.run_model()

sub_optimizer = SequentialFullSolve(prob_model=probm, prob_truth=probt, 
                                    flat_refinement=jump, 
                                    max_iter=max_outer,
                                    use_truth_to_train=use_truth_to_train,)
sub_optimizer.setup_optimization()
sub_optimizer.solve_full()

# x_opt_1 = sub_optimizer.result_cur['stat.xd'][0] #prob.get_val("stat.x_d")[0]

# plot conv
cs = plt.plot(probm.model.stat.func_calls, probm.model.stat.objs)
plt.xlabel(r"Number of function calls")
plt.ylabel(r"$\mu_f(x_d)$")
#plt.legend(loc=1)
plt.savefig(f"./{name}/convergence_model_nosurr.pdf", bbox_inches="tight")
plt.clf()

import pdb; pdb.set_trace()
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




