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

    # scale
    err /= (max(tf) - min(tf))
    
    return err


def meane(model, prob, N=5000, xdata=None, fdata=None):
    """
    Compute the error of the mean value of the function.
    
    Inputs:
        model : smt surrogate modeling object
        prob : smt problem object
        N : number of points to evaluate the error
        xdata : predefined set of points to evaluate the error
        fdata : predefined true function data

    Outputs:
        err : error of the mean
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

    # compute error of mean
    vals = model.predict_values(tx)
    tmean = sum(tf)/N
    mmean = sum(vals)/N
    tstdev = np.sqrt((sum(tf*tf)/N) - (sum(tf)/N)**2)
    mstdev = np.sqrt((sum(vals*vals)/N) - (sum(vals)/N)**2)

    serr = abs(tstdev - mstdev)
    merr = abs(tmean - mmean)
    # scale
    #err /= abs(tmean)
    

    return merr, serr

# def stdeve(model, prob, N=5000, xdata=None, fdata=None):
#     """
#     Compute the error of the stdev value of the function.
    
#     Inputs:
#         model : smt surrogate modeling object
#         prob : smt problem object
#         N : number of points to evaluate the error
#         xdata : predefined set of points to evaluate the error
#         fdata : predefined true function data

#     Outputs:
#         err : error of the stdev
#     """
#     err = 0
#     xlimits = prob.xlimits
     
#     if(xdata is not None):
#         tx = xdata
#         N = xdata.shape[0]
#     else:
#         sampling = LHS(xlimits=xlimits, criterion='maximin')
#         tx = sampling(N)

#     if(fdata is not None and xdata is not None):
#         tf = fdata
#     else:
#         tf = prob(tx)

#     # compute error of stdev
#     vals = model.predict_values(tx)
    

#     # scale
#     #err /= abs(tmean)
    

#     return err