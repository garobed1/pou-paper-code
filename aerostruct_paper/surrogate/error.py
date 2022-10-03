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
    dim = tx.shape[1]
    vals = np.zeros([N,1])
    if(N > 5000):
        arrs = np.array_split(tx, dim)
        l1 = 0
        l2 = 0
        for k in range(dim):
            l2 += arrs[k].shape[0]
            vals[l1:l2,:] = model.predict_values(arrs[k])
            l1 += arrs[k].shape[0]
    else:
        vals = model.predict_values(tx)
    for i in range(N):
        work = tf[i] - vals[i]
        err += work*work
    
    err = np.sqrt(err/N)

    # scale
    err /= (max(tf) - min(tf))
    
    return err


def meane(model, prob, N=5000, xdata=None, fdata=None, return_values=False):
    """
    Compute the error of the mean value of the function.
    
    Inputs:
        model : smt surrogate modeling object
        prob : smt problem object
        N : number of points to evaluate the error
        xdata : predefined set of points to evaluate the error
        fdata : predefined true function data
        return_values : return statistics, or their error

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

    # get benchmark values for mean and stdev
    if("mean" in prob.__dict__):
        tmean = prob.mean
        tstdev = prob.stdev
    else:
        tmean = sum(tf)/N
        tstdev = np.sqrt((sum(tf*tf)/N) - (sum(tf)/N)**2)


    # compute error of mean
    dim = tx.shape[1]
    vals = np.zeros([N,1])
    if(N > 5000):
        arrs = np.array_split(tx, dim)
        l1 = 0
        l2 = 0
        for k in range(dim):
            l2 += arrs[k].shape[0]
            vals[l1:l2,:] = model.predict_values(arrs[k])
            l1 += arrs[k].shape[0]
    else:
        vals = model.predict_values(tx)
    mmean = sum(vals)/N
    mstdev = np.sqrt((sum(vals*vals)/N) - (sum(vals)/N)**2)

    serr = abs(tstdev - mstdev)
    merr = abs(tmean - mmean)

    # scale
    #err /= abs(tmean)
    
    if(return_values):
        return mmean, mstdev, tmean, tstdev

    return merr, serr


def full_error(model, prob, N=5000, xdata=None, fdata=None , return_values=False):
    """
    Compute the root mean square error of a surrogate model, either with provided 
    data or data sampled through LHS.
    
    Inputs:
        model : smt surrogate modeling object
        prob : smt problem object
        N : number of points to evaluate the error
        xdata : predefined set of points to evaluate the error
        fdata : predefined true function data
        return_values : return statistics, or their error


    Outputs:
        nrmse : root mean square error
        meane : error of the mean
        stdve : error of the std dev
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

    # compute NRMSE
    dim = tx.shape[1]
    vals = np.zeros([N,1])
    if(N > 5000):
        arrs = np.array_split(tx, dim)
        l1 = 0
        l2 = 0
        for k in range(dim):
            l2 += arrs[k].shape[0]
            vals[l1:l2,:] = model.predict_values(arrs[k])
            l1 += arrs[k].shape[0]
    else:
        vals = model.predict_values(tx)
    for i in range(N):
        work = tf[i] - vals[i]
        err += work*work
    nrmse = np.sqrt(err/N)
    # scale
    nrmse /= (max(tf) - min(tf))
    
    # compute mean error
    # get benchmark values for mean and stdev
    if("mean" in prob.__dict__):
        tmean = prob.mean
        tstdev = prob.stdev
    else:
        tmean = sum(tf)/N
        tstdev = np.sqrt((sum(tf*tf)/N) - (sum(tf)/N)**2)

    # compute error of mean
    mmean = sum(vals)/N
    mstdev = np.sqrt((sum(vals*vals)/N) - (sum(vals)/N)**2)

    serr = abs(tstdev - mstdev)
    merr = abs(tmean - mmean)

    if(return_values):
        return mmean, mstdev, tmean, tstdev

    return nrmse, merr, serr
