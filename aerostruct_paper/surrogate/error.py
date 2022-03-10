import numpy as np
from smt.sampling_methods import LHS



def rmse(model, prob, N=5000, xdata=None, fdata=None):
    """
    Compute the root mean square error of a surrogate model, either with provided 
    data or data sampled through LHS.
    
    Inputs:
        model : smt surrogate modeling object
        prob : smt problem object
        N : number of points to evaluate the error
        xdata : predefined set of points to evaluate the error
        fdata : predefined true function data

    Outputs:
        err : root mean square error
    """
    err = 0
    xlimits = prob.xlimits
     
    if(xdata is not None):
        tx = xdata
        N = xdata.shape[0]
    else:
        sampling = LHS(xlimits=xlimits, criterion='maximin')
        tx = sampling(N)

    if(fdata is not None and xdata is not None):
        tf = fdata
    else:
        tf = prob(tx)

    # compute RMSE
    vals = model.predict_values(tx)
    for i in range(N):
        work = tf[i] - vals[i]
        err += work*work
    
    err = np.sqrt(err/N)
    
    return err