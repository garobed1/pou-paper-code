from optimizers import optimize
import copy
from defaults import DefaultOptOptions
import numpy as np
from smt.surrogate_models import GEKPLS
from pougrad import POUSurrogate


"""
The adaptive sampling routine. Given a refinement criteria, find its corresponding 
optimal observation point.

Parameters
----------
rcrit : ASCriteria object
    Refinement criteria function
options : dictionary object
    Optimization settings

Returns
----------
xnew : ndarray
    New adaptive sampling point


"""

def getxnew(rcrit, x0, bounds, options=None):
    
    # set default options if None is provided
    if(options == None):
        options = DefaultOptOptions
    
    xnew = []
    bounds_used = bounds

    #gresults = optimize(rcrit.evaluate, args=(), bounds=bounds, type="global")
    for i in range(rcrit.nnew):
        x0, lbounds = rcrit.pre_asopt(bounds, dir=i)
        if(lbounds is not None):
            bounds_used = lbounds
        args=(i,)
        if(options["localswitch"]):
            results = optimize(rcrit.evaluate, args=args, bounds=bounds_used, type="local", x0=x0)
        else:
            results = optimize(rcrit.evaluate, args=args, bounds=bounds_used, type="global")
    #    results = gresults
        xnew.append(rcrit.post_asopt(results.x, dir=i))

    return xnew


def adaptivesampling(func, model, rcrit, bounds, ntr, options=None):

    #TODO: Alternate Stopping Criteria
    count = int(ntr/rcrit.nnew)
    hist = []
    
    for i in range(count):
        t0 = model.training_points[None][0][0]
        f0 = model.training_points[None][0][1]
        g0 = rcrit.grad
        nt, dim = t0.shape
        x0 = np.zeros([1, dim])
        # if(isinstance(model, GEKPLS) or isinstance(model, POUSurrogate)):
        #     for i in range(dim):
        #         g0.append(model.training_points[None][i+1][1])

        # get the new points
        xnew = np.array(getxnew(rcrit, x0, bounds, options))

        # add the new points to the model
        t0 = np.append(t0, xnew, axis=0)
        f0 = np.append(f0, func(xnew), axis=0)
        g0 = np.append(g0, np.zeros([xnew.shape[0], xnew.shape[1]]), axis=0)
        for i in range(dim):
            g0[nt:,i] = func(xnew, i)[:,0]

        model.set_training_values(t0, f0)
        if(isinstance(model, GEKPLS) or isinstance(model, POUSurrogate)):
            for i in range(dim):
                model.set_training_derivatives(t0, g0[i], i)
        model.train()

        hist.append(copy.deepcopy(rcrit))

        # replace criteria
        rcrit.initialize(model, g0)

    return model, rcrit, hist
