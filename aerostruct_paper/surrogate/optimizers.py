from scipy.optimize import minimize, differential_evolution
from defaults import DefaultOptOptions

"""
Wrapper for scipy optimizers
"""
def optimize(func, args, bounds, type="global", x0=None, jac=None, hess=None, constraints=None, options=None):
    
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
            results = differential_evolution(func, bounds, args, strategy, maxiter=giter, popsize=gpop, tol=gtol, constraints=constraints)
        else:
            return

    # if local, use minimize
    else:
        results = minimize(func, x0, args, maxiter=liter, method=lmethod, jac=jac, hess=hess, bounds=bounds, constraints=constraints, tol=ltol)

    return results