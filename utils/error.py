import numpy as np
from smt.sampling_methods import LHS
from scipy.stats import uniform, norm, beta, truncnorm

from smt.utils.options_dictionary import OptionsDictionary
from utils.stat_comps import _mu_sigma_comp, _mu_sigma_grad

#TODO: Need a way to overload this for stochastic collocation weights, e.g.
_pdf_handle = {
    "uniform":uniform,
    "norm":norm,
    "beta":beta
}

_stat_handle = {
    "mu_sigma":_mu_sigma_comp,
}

_stat_handle_grad = {
    "mu_sigma":_mu_sigma_grad,
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

    err_tup, t_tup, m_tup = stat_comp(model, prob, stat_type="mu_sigma", N=N, xdata=xdata, fdata=fdata, pdfs=["uniform"], compute_error = not return_values)
    rval1 = err_tup[0]
    rval2 = err_tup[1]
    return rval1, rval2

def stat_comp(model, prob, stat_type="mu_sigma", N=5000, xdata=None, fdata=None, pdfs=["uniform"], compute_error=False, get_grad=False):
    """
    Compute the mean and standard deviation of a function using either the function itself or a surrogate, along with its error
    if reference values or data are available.

    TODO: Implement other statistical quantities (e.g. higher moments, probability of failure, etc.)
    
    Inputs:
        model : smt.Surrogate or None
            smt surrogate modeling object. If None, stats are computed from the prob, and xdata and fdata must be provided to compute error

        prob : smt problem object
        N : if sampler is not present, use this number of (LHS) points to evaluate the func or surrogate
        xdata : ndarray or RobustSampler
            predefined set of points to compute statistics
            NOTE: If xdata is provided as an array, static input dimensions should not vary and should match the corresponding value in pdfs
        fdata : predefined true function data for the given xdata

        TODO: separate set of args for error comparison?

        pdfs : str or list
            probability distribution functions associated with each uncertain variable 
            if str, consider all variables in that way
        compute_error : return error in addition to statistics

        get_grad : return gradient of outputs of interest

    Outputs:
        tuple of outputs or output gradients
    """
    
    xlimits = prob.xlimits

    dim = len(xlimits)#tx.shape[1]

    using_sampler= False

    from optimization.robust_objective import RobustSampler
    if isinstance(xdata, RobustSampler):
        using_sampler = True


    # if pdfs is of length 1, treat each variable the same
    if isinstance(pdfs, str):
        pdfs = [pdfs]*dim
    elif len(pdfs) == 1:
        pdfs = pdfs*dim
    elif len(pdfs) != dim:
        Exception(f'pdfs arg should be a list of length 1 or dim={dim}, but is {len(pdfs)} instead.')

    # get scales for uncertain variables, and preset pdf funcs for each
    #TODO: find a way to deal with normal dist/truncated normal at least
    pdf_list, uncert_list, static_list, scales, pdf_name_list = _gen_var_lists(pdfs, xlimits)
    u_scales = scales[uncert_list]

    u_dim = len(uncert_list)
    d_dim = len(static_list)

    assert(u_dim + d_dim == dim, f'{u_dim} uncertain and {d_dim} static vars do not sum to the total {dim} vars!')

    # generate sample points, with fixed static values
    # TODO: need some kind of all-encompassing sampling function
    weights=None # for collocation method with weights provided by sampler
    if(xdata is not None):
        if using_sampler:
            tx = xdata.current_samples['x']
            tf = xdata.current_samples['f'] #probably None but this should work
            tg = xdata.current_samples['g']
            N = tx.shape[0]

            if hasattr(xdata, 'weights'):
                weights = xdata.weights
        else:
            tx = xdata
            tf = None
            tg = None
            N = xdata.shape[0]
    else:
        # generate points
        u_xlimits = xlimits[uncert_list] 
        sampling = LHS(xlimits=u_xlimits, criterion='maximin') # different sampling techniques?
        u_tx = sampling(N)
        tx = np.zeros([N, dim])
        tx[:, uncert_list] = u_tx
        tx[:, static_list] = [pdf_list[i] for i in static_list]
        tf = None
        tg = None

    # compute statistics
    if(model):
        func_handle = model.predict_values
        if(get_grad):
            grad_handle = model.predict_derivatives
    else:
        func_handle = prob
        grad_handle = prob

    if get_grad:
        if tf is None:
            out_tup, vals = _stat_handle[stat_type](func_handle, N, tx, xlimits, u_scales, pdf_list, tf=tf, weights=weights)
            if using_sampler:
                xdata.set_evaluated_func(vals)

            tf = vals

        grad_tup, grads = _stat_handle_grad[stat_type](grad_handle, N, tx, xlimits, u_scales, static_list, pdf_list, tf=tf, tg=tg, weights=weights)

        if using_sampler:
            xdata.set_evaluated_grad(grads)

        return grad_tup

    out_tup, vals = _stat_handle[stat_type](func_handle, N, tx, xlimits, u_scales, pdf_list, tf=tf)
    
    if using_sampler:
        xdata.set_evaluated_func(vals)

    
    # TODO: this needs all sorts of separate options to handle validation
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
        t_tup, t_vals = _stat_handle[stat_type](prob, Nt, tx, xlimits, u_scales, pdf_list, tf)

        err_tup = []
        for i in range(len(t_tup)):
            err_tup.append(abs(t_tup[i] - out_tup[i]))

        # return errors first if requested
        return err_tup, out_tup, t_tup # to scale error if you want 
    
    return out_tup








def full_error(model, prob, N=5000, xdata=None, fdata=None, pdfs=["uniform"], return_values=False):
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

    # merr, serr = def meane(model, prob, N=5000, xdata=None, fdata=None, return_values=False):
    # merr, serr, mmean, mstdev, tmean, tstdev = stat_comp(model, prob, N=N, xdata=xdata, fdata=fdata, pdfs=pdfs, compute_error=True)
    err_tup, t_tup, m_tup = stat_comp(model, prob, N=N, xdata=xdata, fdata=fdata, pdfs=pdfs, compute_error=True)
    merr = err_tup[0]
    serr = err_tup[1]

    if(return_values):
        return m_tup[0], m_tup[1], t_tup[0], t_tup[1]

    return nrmse, merr, serr


def _gen_var_lists(pdfs, xlimits):
    """
    Based on variable designation in pdfs, return useful lists

    N total variables, N = n_u + n_d

    Parameters:
        pdfs : list
            list of pdf and deterministic information

        xlimits : list
            parameter bounds for each variable

    Returns:
        pdf_list : list
            list of pdf function handles and deterministic values, length N
        uncert_list : list
            list of indices in pdf_list corresponding to uncertain vars, length n_u
        static_list : list
            list of indices in pdf_list corresponding to deterministic vars, length n_d
        scales : list
            list of indices in pdf_list corresponding to deterministic vars, length N
        pdf_name_list : list
            list of pdf name strings, length n_u


    """

    dim = len(pdfs)
    pdf_list = []
    pdf_name_list = [] # same length 
    uncert_list = []   # same length
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
            pdf_name_list.append(pdfs[j][0])
            uncert_list.append(j)

        # pdf with no arguments e.g. uniform
        elif isinstance(pdfs[j], str):   
            args = []
            args.append(xlimits[j][0])#loc=
            args.append(scales[j])#scale=
            pdf_list.append(_pdf_handle[pdfs[j]](*args))
            pdf_name_list.append(pdfs[j])
            uncert_list.append(j)
        
        # no pdf, fixed at the float given
        elif isinstance(pdfs[j], float): # consider these variables to be fixed (e.g. design vars)
            pdf_list.append(pdfs[j])
            static_list.append(j)

        # treat as if uniform
        elif pdfs[j] == None: 
            args = []
            args.append(xlimits[j][0])#loc=
            args.append(scales[j])#scale=
            pdf_list.append(_pdf_handle['uniform'])
            pdf_name_list.append('uniform')
            uncert_list.append(j)
        else: # ndarray, should only have one element
            pdf_list.append(pdfs[j][0])
            static_list.append(j)

    return pdf_list, uncert_list, static_list, scales, pdf_name_list
