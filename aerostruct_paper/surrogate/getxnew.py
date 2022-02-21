from optimizers import optimize
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

    #gresults = optimize(rcrit.evaluate, args=(), bounds=bounds, type="global")
    for i in range(rcrit.nnew):
        x0 = rcrit.pre_asopt()
        args=(i,)
        if(options["localswitch"]):
            results = optimize(rcrit.evaluate, args=args, bounds=bounds, type="local", x0=x0)
        else:
            results = optimize(rcrit.evaluate, args=args, bounds=bounds, type="global")
    #    results = gresults
        xnew.append(rcrit.post_asopt(results.x))

    return xnew


def adaptivesampling(func, model, rcrit, bounds, ntr, options=None):

    #TODO: Alternate Stopping Criteria
    for i in range(ntr):
        x0 = np.array([[0.25, 0]])
        t0 = model.training_points[None][0][0]
        f0 = model.training_points[None][0][1]
        g0 = []
        nt, dim = t0.shape
        if(isinstance(model, GEKPLS) or isinstance(model, POUSurrogate)):
            for i in range(dim):
                g0.append(model.training_points[None][i+1][1])
        xnew = np.array(getxnew(rcrit, x0, bounds, options))
        #import pdb; pdb.set_trace()
        t0 = np.append(t0, xnew, axis=0)
        f0 = np.append(f0, func(xnew), axis=0)
        if(isinstance(model, GEKPLS) or isinstance(model, POUSurrogate)):
            for i in range(dim):
                g0[i] = np.append(g0[i], func(xnew, i), axis=0)

        model.set_training_values(t0, f0)
        if(isinstance(model, GEKPLS) or isinstance(model, POUSurrogate)):
            for i in range(dim):
                model.set_training_derivatives(t0, g0[i], i)
        model.train()

        #zs = model.predict_values(x)

        #replace criteria
        rcrit.__init__(model, approx=False)# = looCV(model, approx=False)
        #zlv = criteria.evaluate(x)

    return model, rcrit
