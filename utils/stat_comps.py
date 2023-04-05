import numpy as np
from utils.sutils import convert_to_smt_grads







def _mu_sigma_comp(func_handle, N, tx, xlimits, scales, pdf_list, tf = None):

    dim = tx.shape[1]
    
    vals = np.zeros([N,1])
    dens = np.ones([N,1])
    summand = np.zeros([N,1])
    

    # uniform importance sampled monte carlo integration
    #NOTE: REMEMBER TO CITE/MENTION THIS
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
            if not isinstance(pdf_list[j], float):
                dens[l1:l2,:] = np.multiply(dens[l1:l2,:], pdf_list[j].pdf(arrs[k][:,j]).reshape((l2-l1, 1))) #TODO: loc must be different for certain dists
        summand[l1:l2,:] = dens[l1:l2,:]*vals[l1:l2,:]
        l1 += arrs[k].shape[0]

    area = np.prod(scales) #just a hypercube
    mean = area*sum(summand)/N
    #stdev = np.sqrt(((area*sum(summand*vals))/N - (mean)**2))#/N
    A = sum(dens)/N
    stdev = np.sqrt(((area*sum(summand*vals))/N - (2-A)*(mean)**2 ))#/N

    return (mean, stdev), vals


#TODO: need to not recompute functions if not needed, right now it will rerun analyses already completed
def _mu_sigma_grad(func_handle, N, tx, xlimits, scales, static_list, pdf_list, tf, tg = None):

    dim = tx.shape[1]
    dim_d = len(static_list)
    
    grads = np.zeros([N,dim])
    vals = np.zeros([N,1])
    dens = np.ones([N,1])
    summand = np.zeros([N,1])
    gsummand = np.zeros([N,dim_d])
    
    arrs = np.array_split(tx, dim)
    l1 = 0
    l2 = 0
    for k in range(dim):
        l2 += arrs[k].shape[0]
        
        # tf is needed
        vals[l1:l2,:] = tf[l1:l2,:]
        
        if tg is not None:
            grads[l1:l2,:] = tg[l1:l2,:]
        else:
            # grads[l1:l2,:] = convert_to_smt_grads(func_handle, arrs[k])
            for ki in range(dim):
                grads[l1:l2,ki] = func_handle(arrs[k], kx=ki)[:,0]
        
        for j in range(dim):
            #import pdb; pdb.set_trace()
            if not isinstance(pdf_list[j], float):
                dens[l1:l2,:] = np.multiply(dens[l1:l2,:], pdf_list[j].pdf(arrs[k][:,j]).reshape((l2-l1, 1))) #TODO: loc must be different for certain dists
        summand[l1:l2,:] = dens[l1:l2,:]*vals[l1:l2,:]
        
        for j in range(dim_d):
            gsummand[l1:l2,j] = dens[l1:l2,:][:,0]*grads[l1:l2,static_list[j]]
        
        l1 += arrs[k].shape[0]

    area = np.prod(scales) #just a hypercube
    mean = area*sum(summand)/N
    gmean = (area/N)*np.sum(gsummand, axis=0 )

    #stdev = np.sqrt(((area*sum(summand*vals))/N - (mean)**2))#/N
    A = sum(dens)/N
    stdev = np.sqrt(((area*sum(summand*vals))/N - (2-A)*(mean)**2 ))#/N
    work = 0.5*(1./stdev)
    work2 = 2.*(2.-A)*mean*gmean
    work3 = np.multiply(2*summand, grads[:,static_list])
    work3 = (area/N)*np.sum(work3, axis=0)
    gstdev = work*(work3 - work2)

    #return full gradients, but gmean and gstdev are only with respect to dvs
    return (gmean, gstdev), grads

