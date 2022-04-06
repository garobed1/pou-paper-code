import numpy as np
import copy

from matplotlib import pyplot as plt
from smt.utils.options_dictionary import OptionsDictionary
from smt.sampling_methods import LHS
from smt.surrogate_models import GEKPLS
from pougrad import POUMetric, POUSurrogate
from refinecriteria import ASCriteria
from scipy.linalg import lstsq, eig
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.optimize import Bounds, least_squares, root
from scipy.stats import qmc
from utils import linear, quadratic, quadraticSolve, quadraticSolveHOnly, symMatfromVec, maxEigenEstimate, boxIntersect


# Hessian estimation to generate an anisotropic mapping of the space

# Places points in a space filling manner, but mapped from an isotropic to an anisotropic space
class AnisotropicTransform(ASCriteria):
    def __init__(self, model, init_sequence, grad, **kwargs):

        self.bads = None
        self.bad_list = None
        self.nbhd = None
        self.eigvals = None

        self.trx = None # use for sequential optimization for batches of points
        self.trxi = model.training_points[None][0][0] # only for initial sampling
        self.dminmax = None
        self.grad = grad
        self.bounds = None
        self.bnorms = None
        self.bpts = None

        self.metric = None
        self.mmodel = None #POU model of the anisotropy metric
        self.sequence = init_sequence #Quasi Monte Carlo sequence for space-filling design

        super().__init__(model, **kwargs)

        self.opt = False
        #self.supports["obj_derivatives"] = True  
        
    def _init_options(self):
        #options: neighborhood, surrogate, exact
        self.options.declare("hessian", "neighborhood", types=str)

        #options: honly, full, arnoldi
        self.options.declare("interp", "arnoldi", types=str)

        #options: linear, quadratic
        self.options.declare("error", "linear", types=str)

        self.options.declare("improve", 0, types=int)

        #number of closest points to evaluate hessian eigenpair
        self.options.declare("neval", self.dim*2+1, types=int)

        #number of closest points to match differential distance
        self.options.declare("nmatch", self.dim*2, types=int)
        
    def initialize(self, model=None, grad=None):

        if(model is not None):
            self.model = copy.deepcopy(model)
            kx = 0
            self.dim = self.model.training_points[None][kx][0].shape[1]
            self.ntr = self.model.training_points[None][kx][0].shape[0]

        if(grad is not None):
            self.grad = grad

        self.nnew = self.options["improve"]#int(self.ntr*self.options["improve"])
        if(self.nnew == 0):
            self.nnew = 1

        trx = self.model.training_points[None][0][0]
        trf = self.model.training_points[None][0][1]
        trg = np.zeros_like(trx)
        trg = self.grad
        if(isinstance(self.model, GEKPLS)):
            for j in range(self.dim):
                trg[:,j] = self.model.training_points[None][j+1][1].flatten()

        self.trx = trx
        
        dists = pdist(trx)
        dists = squareform(dists)

        neval = self.options["neval"]
        if(self.options["interp"] == "arnoldi"):
            neval = self.dim


        # 1. Estimate the Hessian/the principal Hessian eigenpair about each point
        hess = []
        metric = np.zeros([self.ntr, self.dim, self.dim])
        nbhd = []
        indn = []

        # 1a. Determine the neighborhood to fit the Hessian for each point/evaluate the error
        # along with minimum distances
        mins = np.zeros(self.ntr)
        for i in range(self.ntr):
            ind = dists[i,:]
            ind = np.argsort(ind)
            indn.append(ind)
            mins[i] = dists[i,ind[1]]
        self.dminmax = max(mins)
        lmax = np.amax(dists)

        if(self.options["hessian"] == "neighborhood"):        
            for i in range(self.ntr):
                if(self.options["interp"] == "full"):
                    fh, gh, Hh = quadraticSolve(trx[i,:], trx[indn[i][1:neval+1],:], \
                                            trf[i], trf[indn[i][1:neval+1]], \
                                            trg[i,:], trg[indn[i][1:neval+1],:])

                if(self.options["interp"] == "honly"):
                    Hh = quadraticSolveHOnly(trx[i,:], trx[indn[i][1:neval+1],:], \
                                            trf[i], trf[indn[i][1:neval+1]], \
                                            trg[i,:], trg[indn[i][1:neval+1],:])
                    fh = trf[i]
                    gh = trg[i,:]

                if(self.options["interp"] == "full" or self.options["interp"] == "honly"):
                    hess.append(np.zeros([self.dim, self.dim]))
                    for j in range(self.dim):
                        for k in range(self.dim):
                            hess[i][j,k] = Hh[symMatfromVec(j,k,self.dim)]
                
                else: #arnoldi
                    evalm, evecm = maxEigenEstimate(trx[i,:], trx[indn[i][1:neval],:], \
                                                    trg[i,:], trg[indn[i][1:neval],:])

                    hess.append([evalm, evecm])

        if(self.options["hessian"] == "surrogate"):
            # 1a. Get the hessian as determined by the surrogate
            # central difference scheme
            h = 1e-5
            for i in range(self.ntr):
                hess.append(np.zeros((self.dim, self.dim)))
            
            for j in range(self.dim):
                xsp = np.copy(trx)
                xsm = np.copy(trx)
                xsp[:,j] += h
                xsm[:,j] -= h

                for k in range(self.dim):
                    hj = np.zeros(self.dim)
                    hj = self.model.predict_derivatives(xsp, k)
                    hj -= self.model.predict_derivatives(xsm, k)
                    for l in range(len(hess)):
                        hess[l][j,k] = hj[l]/h




        


        # 2. Take the highest eigenvalue/vector of each Hessian and generate the metric M
        opt_dir = []
        opt_val = []
        min_eig = [] # for symmetry
        if(self.options["interp"] == "arnoldi"):
            for i in range(self.ntr):
                opt_dir.append(hess[i][1])
                opt_val.append(hess[i][0])

        else:
            for i in range(self.ntr):
                H = hess[i]
                eigvals, eigvecs = eig(H)
                o = np.argsort(abs(eigvals))
                opt_dir.append(eigvecs[:,o[-1]])
                opt_val.append(eigvals[o[-1]])
                min_eig.append(eigvals[o[0]])


        for i in range(self.ntr):
            work = np.outer(opt_dir[i], opt_dir[i])
            metric[i,:,:] = np.abs(opt_val[i])*work
            metric[i,:,:] += np.abs(np.real(min_eig[i]))*(np.eye(self.dim) - work)
        metric /= sum(np.abs(np.real(opt_val)))/(len(opt_val)*1.)
        # Now train a POU surrogate on the metric

        # increase rho based on number of points, minimum distance
        rho = 1.#*self.dim*self.ntr#100.#self.ntr*100.#/(self.dim*1.)

        self.mmodel = POUMetric(rho=rho, metric=metric) # the metric are the actual training outputs, this is a bad workaround
        self.mmodel.set_training_values(trx, np.ones(trx.shape[0]))
        self.mmodel = metric
        #import pdb; pdb.set_trace()
        # h = 1e-6
        # zero = np.zeros([1,2])
        # step = np.zeros([1,2])
        # step[0,0] += h
        # ad = self.mmodel.predict_derivatives(zero)
        # fd = (self.mmodel.predict_values(step) - self.mmodel.predict_values(zero))/h

        self.bad_eigs = opt_val
        self.bad_dirs = opt_dir
        #import pdb; pdb.set_trace()

    def pre_asopt(self, bounds, dir=0):
        
        xc = self.bads[dir]
        # xdir = self.bad_dirs[dir]
        trx = self.trx
        m, n = trx.shape

        # # Store points at centroids of the bounds
        # center = np.zeros(n)
        # for i in range(n):
        #     center[i] = (bounds[i,1]-bounds[i,0])/2. + bounds[i,0]
        # #self.bounds = bounds
        # self.bounds = np.zeros([2*n, n])
        # for i in range(n):
        #     self.bounds[i,:] = center
        #     self.bounds[i,i] = bounds[i,0] 
        #     self.bounds[i+n,:] = center
        #     self.bounds[i+n,i] = bounds[i,1] 
        
        # # Store boundary normals
        # self.bnorms = np.zeros([2*n, n])
        # for i in range(n):
        #     self.bnorms[i,i] = -1.
        #     self.bnorms[i+n,i] = 1.



        # # need to update this for batches
        # #dists = pdist(trx, lambda u, v: np.matmul(np.matmul((u-v), self.mmodel.predict_values(np.array([u]))), (u-v)))
        # dists = pdist(trx)
        # dists = squareform(dists)
        # mins = np.zeros(m)
        # for i in range(m):
        #     ind = dists[i,:]
        #     ind = np.argsort(ind)
        #     mins[i] = dists[i,ind[1]]
        # self.dminmax = max(mins)
        # # S = 0.5*self.dminmax



        # # find which points are on boundaries 
        # self.bpts = []
        # self.bnorms = []
            
        # co = 0
        # for i in range(m):
        #     on_bound = False
        #     work = abs(trx[i,:] - bounds[:,0]) #lb
        #     work2 = abs(trx[i,:] - bounds[:,1]) #ub
        #     if((work < 1e-8).any() or (work2 < 1e-8).any()):
        #         on_bound = True

        #     if(on_bound):
        #         self.bpts.append(trx[i,:])
        #         self.bnorms.append(np.zeros([n]))
        #         for j in range(n):
        #             if(abs(trx[i,j] - bounds[j ,0]) < 1e-8):
        #                 self.bnorms[co][j] = -1.
        #             if(abs(trx[i,j] - bounds[j ,1]) < 1e-8):
        #                 self.bnorms[co][j] = 1.
        #         co += 1


        sampling = LHS(xlimits=bounds, criterion='m')
        ntries = self.options["multistart"]
        if(ntries > 1):
            xc = sampling(ntries)
        else: 
            xc = np.random.rand(n)*(bounds[:,1] - bounds[:,0]) + bounds[:,0]
            xc = np.array([xc])

        # for i in range(n):
        #     xc[i] = xc[i]*(bounds[i,1] - bounds[i,0]) + bounds[i,0]
        return xc, bounds# + 0.001*self.dminmax+randvec, bounds



    def post_asopt(self, x, bounds, dir=0):

        # we're only transforming from the isotropic to the anisotropic space, since we can't transform the existing points in the sequence
        
        # generate new point from the sequence
        xinew = self.sequence.random(1)
        xinew = qmc.scale(xinew, bounds[:,0], bounds[:,1])
        nmatch = self.options["nmatch"]

        # find the corresponding anisotropic point
        dists = cdist(xinew, self.trxi)
        #dists = cdist(xinew, self.trx)
        ind = np.argsort(dists)
        nbhdxi = self.trxi[ind[0:nmatch]]
        nbhdx = self.trx[ind[0:nmatch]]

        # solve quadratic set of equations for xnew
        # rhs and lhs
        x0 = copy.deepcopy(xinew)
        # M = np.zeros([nmatch, self.dim, self.dim])
        # M = self.mmodel.predict_values(nbhdx[0])
        M = self.mmodel[ind[0:nmatch]][0]
        r0 = get_residual(x0[0], xinew, nbhdx[0], nbhdxi[0], M, nmatch)

        # h = 1e-6
        # step = copy.deepcopy(x0[0])
        # step[0] += h
        # ad = get_res_jac(x0[0], xinew, nbhdx[0], nbhdxi[0], M, nmatch)
        # fd = ( get_residual(step, xinew, nbhdx[0], nbhdxi[0], M, nmatch) - r0)/h

        args = (xinew, nbhdx[0], nbhdxi[0], M, nmatch)
        # if(nmatch == self.dim):
        #     results = root(get_residual, x0[0], jac=get_res_jac, args=args, options={"disp":True})
        # else:
        results = least_squares(get_residual, x0[0], get_res_jac, bounds=(bounds[:,0],bounds[:,1]), args=args, verbose=0, xtol=None, ftol=None)
        
        # keep the solution in bounds if needed
        xnew = results.x
        for i in range(self.dim):
            if(xnew[i] > bounds[i,1]):
                xnew[i] = bounds[i,1]
            if(xnew[i] < bounds[i,0]):
                xnew[i] = bounds[i,0]


        self.trxi = np.append(self.trxi, xinew, axis=0)
        self.trx = np.append(self.trx, np.array([results.x]), axis=0)

        return results.x




def get_residual(x, xinew, nbhdx, nbhdxi, M, nmatch):
    d2r = np.zeros(nmatch)
    d2l = np.zeros(nmatch)
    for i in range(nmatch):
        workxi = xinew - nbhdxi[i]
        d2r[i] = np.dot(workxi, workxi.T)
        workx = x - nbhdx[i]
        d2l[i] = np.matmul(np.matmul(workx, M[i]), workx.T)
    res = d2r - d2l
    return res
        
def get_res_jac(x, xinew, nbhdx, nbhdxi, M, nmatch):
    # no dependence on rhs
    d2l = np.zeros(nmatch)
    dd2l = np.zeros([nmatch, x.shape[0]])
    for i in range(nmatch):
        workx = x - nbhdx[i]
        d2l[i] = np.matmul(np.matmul(workx, M[i]), workx.T)
        dd2l[i,:] = 2*np.matmul(workx, M[i])
    dres = -dd2l
    return dres



