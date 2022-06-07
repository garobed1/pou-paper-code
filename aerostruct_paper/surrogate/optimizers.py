from scipy.optimize import minimize, differential_evolution, NonlinearConstraint
from defaults import DefaultOptOptions
import numpy as np

"""
Wrapper for scipy optimizers
"""
def optimize(func, args, bounds, type="global", x0=None, jac=None, hess=None, constraints=(), options=None):
    
    # if no options are provided
    if(options == None):
        options = DefaultOptOptions

    method = options["method"]
    strategy = options["strategy"]
    lmethod = options["lmethod"]
    giter = options["giter"]
    gpop = options["gpop"]
    gtol = options["gtol"]
    liter = options["liter"]
    ltol = options["ltol"]

    # check if using a global or local method
    if(type == "global"):
        if(method == "ga"):
            #constraints = NonlinearConstraint(constraints["fun"], lb=0., ub=np.inf)
            results = differential_evolution(func, bounds, args, strategy, maxiter=giter, popsize=gpop, tol=gtol)
        else:
            return

    # if local, use minimize
    else:
        results = minimize(func, x0, args, method=lmethod, jac=jac, hess=hess, bounds=bounds, constraints=constraints, tol=ltol, options={"maxiter":liter,"disp":True, "iprint":1})
        #import pdb; pdb.set_trace()

    return results