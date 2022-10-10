from optimizers import optimize
from mpi4py import MPI
import copy
from defaults import DefaultOptOptions
import numpy as np
from smt.surrogate_models import GEKPLS
from direct_gek import DGEK
from scipy.stats import qmc
from pougrad import POUSurrogate
from error import rmse, meane, full_error

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

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
    m, n = x0.shape
    xnew = []
    bounds_used = bounds
    unit_bounds = np.zeros([n,2])
    unit_bounds[:,1] = 1.

    # loop over batch
    for i in range(rcrit.nnew):
        rx = None
        if(rcrit.opt): #for methods that don't use optimization
            x0, lbounds = rcrit.pre_asopt(bounds, dir=i)
            x0 = qmc.scale(x0, bounds_used[:,0], bounds_used[:,1], reverse=True)
            m, n = x0.shape
            if(lbounds is not None):
                bounds_used = lbounds
            args=(bounds_used, i,)
            if(rcrit.condict is not None):
                rcrit.condict["args"] = [bounds_used, i]
            jac = None
            if(rcrit.supports["obj_derivatives"]):
                jac = rcrit.eval_grad
            if(options["local"]):

                # proper multistart
                if(options["multistart"] == 2):
                    resx = np.zeros([m,n])
                    resy = np.zeros(m)
                    succ = np.full(m, True)
                    for j in range(m):
                        
                        results = optimize(rcrit.evaluate, args=args, bounds=unit_bounds, type="local", constraints=rcrit.condict, jac=jac, x0=x0[j,:])
                        resx[j,:] = results.x
                        resy[j] = results.fun
                        succ[j] = results.success
                    valid = np.where(succ)[0]
                    try:
                        rx = resx[valid[np.argmin(resy[valid])]]
                    except:
                        rx = resx[np.argmin(resy)]
                    # print(rx)

                # start at best point
                elif(options["multistart"] == 1):
                    x0b = None
                    y0 = np.zeros(m)
                    for j in range(m):
                        y0[j] = rcrit.evaluate(x0[j], bounds_used, i)
                    ind = np.argmin(y0)
                    x0b = x0[0]
                    results = optimize(rcrit.evaluate, args=args, bounds=unit_bounds, type="local", constraints=rcrit.condict, jac=jac, x0=x0b)
                    rx = results.x

                # perform one optimization
                else:
                    x0b = x0[0]
                    results = optimize(rcrit.evaluate, args=args, bounds=unit_bounds, type="local", constraints=rcrit.condict, jac=jac, x0=x0b)
                    rx = results.x

            else:
                results = optimize(rcrit.evaluate, args=args, bounds=unit_bounds, type="global", constraints=rcrit.condict)
                rx = results.x
            
            rx = qmc.scale(np.array([rx]), bounds_used[:,0], bounds_used[:,1])
            rx = rx[0]
        else:
            rx = None

        xnew.append(rcrit.post_asopt(rx, bounds, dir=i))

    return xnew


def adaptivesampling(func, model0, rcrit, bounds, ntr, options=None):

    count = int(ntr/rcrit.nnew)
    hist = []
    errh = []
    errh2 = []
    model = copy.deepcopy(model0)
    

    for i in range(count):
        t0 = model.training_points[None][0][0]
        f0 = model.training_points[None][0][1]
        g0 = rcrit.grad
        nt, dim = t0.shape
        x0 = np.zeros([1, dim])

        # get the new points
        xnew = np.array(getxnew(rcrit, x0, bounds, options))
        # import pdb; pdb.set_trace()
        # add the new points to the model
        t0 = np.append(t0, xnew, axis=0)
        f0 = np.append(f0, func(xnew), axis=0)
        g0 = np.append(g0, np.zeros([xnew.shape[0], xnew.shape[1]]), axis=0)
        for j in range(dim):
            g0[nt:,j] = func(xnew, j)[:,0]
        model.set_training_values(t0, f0)
        if(isinstance(model, GEKPLS) or isinstance(model, POUSurrogate) or isinstance(model0, DGEK)):
            for j in range(dim):
                model.set_training_derivatives(t0, g0[:,j], j)
        model.train()

        # evaluate errors
        if(options["errorcheck"] is not None):
            xdata, fdata, intervals = options["errorcheck"]
            # err = rmse(model, func, xdata=xdata, fdata=fdata)
            # err2 = meane(model, func, xdata=xdata, fdata=fdata)
            if(i in intervals.tolist() and i !=0):
                err = full_error(model, func, xdata=xdata, fdata=fdata)
                errh.append(err[0])
                errh2.append(err[1:])
                #print("yes")
        else:
            errh = None
            errh2 = None

        if i in intervals.tolist():
            hist.append(copy.deepcopy(rcrit.model.training_points[None]))

        if(rcrit.options["print_iter"] and rank == 0):
            print("Iteration: ", i)

        # replace criteria
        rcrit.initialize(model, g0)

    return model, rcrit, hist, errh, errh2
