import numpy as np
from smt.sampling_methods import LHS
from scipy.stats import uniform, norm, beta, truncnorm


_pdf_handle = {
    "uniform":uniform,
    "norm":norm,
    "beta":beta
}

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

# keeping func name to retain compatibility
def meane(model, prob, N=5000, xdata=None, fdata=None, return_values=False):

    rval1, rval2, rval3, rval4, rval5, rval6 = stat_comp(model, prob, N=N, xdata=xdata, fdata=fdata, pdfs=["uniform"], compute_error = not return_values)
    return rval1, rval2

def stat_comp(model, prob, N=5000, xdata=None, fdata=None, pdfs=["uniform"], compute_error=False):
    """
    Compute the mean and standard deviation of a function using either the function itself or a surrogate, along with its error
    if reference values or data are available.

    TODO: Implement other statistical quantities (e.g. higher moments, probability of failure, etc.)
    
    Inputs:
        model : smt.Surrogate or None
            smt surrogate modeling object. If None, stats are computed from the prob, and xdata and fdata must be provided to compute error

        prob : smt problem object
        N : number of points to evaluate the error
        xdata : predefined set of points to compute statistics
            NOTE: If xdata is provided, static input dimensions should not vary and should match the corresponding value in pdfs
        fdata : predefined true function data for the given xdata

        TODO: separate set of args for error comparison?

        pdfs : str or list
            probability distribution functions associated with each uncertain variable 
            if str, consider all variables in that way
        compute_error : return error in addition to statistics

    Outputs:
        
    """

    #NOTE: NEED TO SPEED THIS UP BY CACHING LOTS OF SETTINGS, POTENTIALLY
    # SPLIT INTO A .setup() AND .comp() CLASS

    xlimits = prob.xlimits

    dim = len(xlimits)#tx.shape[1]

    # if pdfs is of length 1, treat each variable the same
    if isinstance(pdfs, str):
        pdfs = [pdfs]*dim
    elif len(pdfs) == 1:
        pdfs = pdfs*dim
    elif len(pdfs) != dim:
        Exception(f'pdfs arg should be a list of length 1 or dim={dim}, but is {len(pdfs)} instead.')

    # get scales for uncertain variables, and preset pdf funcs for each
    #TODO: find a way to deal with normal dist/truncated normal at least
    pdf_list, uncert_list, static_list, scales = _gen_var_lists(pdfs, xlimits)

    u_dim = len(uncert_list)
    d_dim = len(static_list)

    assert(u_dim + d_dim == dim, f'{u_dim} uncertain and {d_dim} static vars do not sum to the total {dim} vars!')

    # generate sample points, with fixed static values
    if(xdata is not None):
        tx = xdata
        N = xdata.shape[0]
    else:
        # generate points
        u_xlimits = xlimits[uncert_list] 
        sampling = LHS(xlimits=u_xlimits, criterion='maximin') # different sampling techniques?
        u_tx = sampling(N)
        tx = np.zeros([N, dim])
        tx[:, uncert_list] = u_tx
        tx[:, static_list] = [pdf_list[i] for i in static_list]



    # compute statistics
    if(model):
        func_handle = model.predict_values
    else:
        func_handle = prob

    mmean, mstdev = _actual_stat_comp(func_handle, N, tx, xlimits, scales, pdf_list)


    # Error comp if requested
    if(compute_error):
        #TODO: refactor this
        # if("mean" in prob.__dict__ and all(isinstance(pdfs == 'uniform'))):
        #     tmean = prob.mean
        #     tstdev = prob.stdev
        # else:
        if(fdata is not None and xdata is not None):
            tf = fdata
            Nt = tf.shape[0]
        else:
            tf = None # prob(tx) # the expensive option
            Nt = N
        tmean, tstdev = _actual_stat_comp(prob, Nt, tx, xlimits, scales, pdf_list, tf)


        serr = abs(tstdev - mstdev)
        merr = abs(tmean - mmean)
    
        # return errors first if requested
        return merr, serr, mmean, mstdev, tmean, tstdev # to scale error if you want
        
    return mmean, mstdev



def _actual_stat_comp(func_handle, N, tx, xlimits, scales, pdf_list, tf = None):

    dim = tx.shape[1]
    
    vals = np.zeros([N,1])
    dens = np.ones([N,1])
    summand = np.zeros([N,1])
    


    arrs = np.array_split(tx, dim)
    l1 = 0
    l2 = 0
    for k in range(dim):
        l2 += arrs[k].shape[0]
        if(tf is not None): #data
            vals[l1:l2,:] = tf[l1:l2,:]
        else:   #function
            vals[l1:l2,:] = func_handle(arrs[k])
        for j in range(dim):
            #import pdb; pdb.set_trace()
            # do i need to divide by N?
            if not isinstance(pdf_list[j], float):
                dens[l1:l2,:] = np.multiply(dens[l1:l2,:], pdf_list[j].pdf(arrs[k][:,j]).reshape((l2-l1, 1))) #TODO: loc must be different for certain dists
        summand[l1:l2,:] = dens[l1:l2,:]*vals[l1:l2,:]
        l1 += arrs[k].shape[0]

    area = np.prod(scales) #just a hypercube
    mean = area*sum(summand)/N
    stdev = np.sqrt(((area*sum(summand*vals))/N - (mean)**2))#/N

    return mean, stdev




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


def _gen_var_lists(pdfs, xlimits):

    dim = len(pdfs)
    pdf_list = []
    uncert_list = []
    static_list = []
    scales = np.zeros(dim)
    for j in range(dim):
        scales[j] = (xlimits[j][1] - xlimits[j][0]) # not necessarily the case
        # if dist needs more args, (e.g. beta shape params) pass the dist as a
        # list with the name as the first argument and remaining args as the next ones
        if isinstance(pdfs[j], list):   
            args = pdfs[j][1:]
            args.append(xlimits[j][0])#loc=
            args.append(scales[j])#scale=
            pdf_list.append(_pdf_handle[pdfs[j][0]](*args))
            uncert_list.append(j)
        elif isinstance(pdfs[j], float): # consider these variables to be fixed (e.g. design vars)
            pdf_list.append(pdfs[j])
            static_list.append(j)
        elif pdfs[j] == None: # treat as if uniform
            args = []
            args.append(xlimits[j][0])#loc=
            args.append(scales[j])#scale=
            pdf_list.append(_pdf_handle['uniform'])
            uncert_list.append(j)
        else:
            args = []
            args.append(xlimits[j][0])#loc=
            args.append(scales[j])#scale=
            pdf_list.append(_pdf_handle[pdfs[j]])
            uncert_list.append(j)

    return pdf_list, uncert_list, static_list, scales
