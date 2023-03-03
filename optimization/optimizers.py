from scipy.optimize import minimize, differential_evolution, NonlinearConstraint
from pyoptsparse import Optimization
from optimization.defaults import DefaultOptOptions
import numpy as np

"""
Wrapper for scipy optimizers
"""
def optimize(func, args, bounds, type="global", x0=None, jac=None, hess=None, constraints=(), callback=None, options=None):
    
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
            # gcon = NonlinearConstraint(lambda x: constraints["fun"](x, constraints["args"]), lb=0., ub=np.inf)
            results = differential_evolution(func, bounds, args, strategy, maxiter=giter, popsize=gpop, tol=gtol, callback=callback, disp=False)
        else:
            return

        if(options["localswitch"]):
            results = minimize(func, results.x, args, method=lmethod, jac=jac, hess=hess, bounds=bounds, constraints=constraints, tol=ltol, callback=callback, options={"maxiter":liter,"disp":False, "iprint":1})

    # if local, use minimize
    else:
        results = minimize(func, x0, args, method=lmethod, jac=jac, hess=hess, bounds=bounds, constraints=constraints, tol=ltol, callback=callback, options={"maxiter":liter,"disp":True, "iprint":2})
        # print(results.x)
        #import pdb; pdb.set_trace()

    return results

"""
Wrapper for pyOptSparse
"""