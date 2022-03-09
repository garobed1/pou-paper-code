import numpy as np
import copy

from smt.utils.options_dictionary import OptionsDictionary
from smt.surrogate_models import GEKPLS
from pougrad import POUSurrogate
from scipy.linalg import lstsq, eig
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import Bounds
from utils import linear, quadratic, quadraticSolve, quadraticSolveHOnly, symMatfromVec, maxEigenEstimate, boxIntersect

"""Base Class for Adaptive Sampling Criteria Functions"""
class ASCriteria():
    def __init__(self, model, **kwargs):
        """
        Constructor for a class encompassing a refinement criteria function for adaptive sampling or cross-validation

        Parameters
        ----------
        model : smt SurrogateModel object
            Surrogate model as needed to evaluate the criteria

        **kwargs : named arguments
            Set of options that can be optionally set; each option must have been declared.
        """
        # copy the surrogate model object
        self.model = copy.deepcopy(model)

        # get the size of the training set
        kx = 0
        self.dim = self.model.training_points[None][kx][0].shape[1]
        self.ntr = self.model.training_points[None][kx][0].shape[0]

        # set options
        self.options = OptionsDictionary()
        self._init_options()
        self.options.update(kwargs)
        
        self.nnew = 1

        self.initialize()

    def _init_options(self):
        pass

    def initialize(self, model=None):
        pass

    def evaluate(self, x, dir=0):
        pass

    def pre_asopt(self, bounds, dir=0):
        pass

    def post_asopt(self, x, dir=0):
        pass

    
"""
A Continuous Leave-One-Out Cross Validation function
"""
class looCV(ASCriteria):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)


        #TODO: Add more options for this
    
    def _init_options(self):
        self.options.declare("approx", False, types=bool)

    def initialize(self, model=None):

        if(model is not None):
            self.model = copy.deepcopy(model)
            kx = 0
            self.dim = self.model.training_points[None][kx][0].shape[1]
            self.ntr = self.model.training_points[None][kx][0].shape[0]

        # Create a list of LOO surrogate models
        self.loosm = []
        for i in range(self.ntr):
            self.loosm.append(copy.deepcopy(self.model))
            self.loosm[i].options.update({"print_global":False})

            kx = 0

            # Give each LOO model its training data, and retrain if not approximating
            trx = self.loosm[i].training_points[None][kx][0]
            trf = self.loosm[i].training_points[None][kx][1]
            trg = []
                
            trx = np.delete(trx, i, 0)
            trf = np.delete(trf, i, 0)
            

            self.loosm[i].set_training_values(trx, trf)
            if(isinstance(self.model, GEKPLS) or isinstance(self.model, POUSurrogate)):
                for j in range(self.dim):
                    trg.append(self.loosm[i].training_points[None][j+1][1]) #TODO:make this optional
                for j in range(self.dim):
                    trg[j] = np.delete(trg[j], i, 0)
                for j in range(self.dim):
                    self.loosm[i].set_training_derivatives(trx, trg[j][:], j)

            if(self.options["approx"] == False):
                self.loosm[i].train()

    #TODO: This could be a variety of possible LOO-averaging functions
    def evaluate(self, x, dir=0):
        
        if(len(x.shape) != 2):
            x = np.array([x])

        # evaluate the point for the original model
        #import pdb; pdb.set_trace()
        M = self.model.predict_values(x).flatten()

        # now evaluate the point for each LOO model and sum
        y = 0
        for i in range(self.ntr):
            Mm = self.loosm[i].predict_values(x).flatten()
            y += (1/self.ntr)*((M-Mm)**2)
        
        ans = -np.sqrt(y)

        return ans # to work with optimizers

    # if only using local optimization, start the optimizer at the worst LOO point
    def pre_asopt(self, bounds, dir=0):
        t0 = self.model.training_points[None][0][0]
        #import pdb; pdb.set_trace()
        diff = np.zeros(self.ntr)

        for i in range(self.ntr):
            M = self.model.predict_values(t0[[i]]).flatten()
            Mm = self.loosm[i].predict_values(t0[[i]]).flatten()
            diff[i] = abs(M - Mm)

        ind = np.argmax(diff)

        return np.array([t0[ind]]), None

    def post_asopt(self, x, dir=0):

        return x
        


# Hessian estimation and direction criteria

class HessianFit(ASCriteria):
    def __init__(self, model, grad, **kwargs):

        self.bads = None
        self.bad_eigs = None
        self.bad_list = None
        self.bad_nbhd = None
        self.bad_dirs = None
        self.grad = grad

        super().__init__(model, **kwargs)

        
        
    def _init_options(self):
        #options: neighborhood, surrogate, exact
        self.options.declare("hessian", "neighborhood", types=str)

        #options: honly, full, arnoldi
        self.options.declare("interp", "arnoldi", types=str)

        #options: distance, variance, random
        self.options.declare("criteria", "distance", types=str)

        #options: linear, quadratic
        self.options.declare("error", "linear", types=str)

        self.options.declare("improve", 0, types=int)

        #number of closest points to evaluate nonlinearity measure
        self.options.declare("neval", self.dim*2, types=int)
        
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
        dists = pdist(trx)
        dists = squareform(dists)

        neval = self.options["neval"]
        if(self.options["interp"] == "arnoldi"):
            neval = self.dim

        # 1. Estimate the Hessian (or its principal eigenpair) about each point
        hess = []
        nbhd = []


        if(self.options["hessian"] == "neighborhood"):
            pts = []
            indn = []

            # 1a. Determine the neighborhood to fit the Hessian for each point
            for i in range(self.ntr):
                ind = dists[i,:]
                ind = np.argsort(ind)
                pts.append(np.array(trx[ind,:]))
                indn.append(ind)
                #for key in ind[1:neval]:

                    
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


        # 2. For every point, sum the discrepancies between the linear (quadratic)
        # prediction in the neighborhood and the observation value
        err = np.zeros(self.ntr)

        # using the original neval here
        for i in range(self.ntr):
            #ind = indn[i]
            ind = dists[i,:]
            ind = np.argsort(ind)
            for key in ind[1:self.options["neval"]]:
                if(self.options["error"] == "linear" or self.options["interp"] == "arnoldi"):
                    fh = linear(trx[key], trx[i], trf[i], trg[:][i])
                else:
                    fh = quadratic(trx[key], trx[i], trf[i], trg[:][i], hess[i])
                err[i] += abs(trf[key] - fh)
            err[i] /= self.options["neval"]
            nbhd.append(ind[1:neval])

        # 2a. Pick some percentage of the "worst" points, and their principal Hessian directions
        badlist = np.argsort(err)
        badlist = badlist[-self.nnew:]
        bads = trx[badlist]
        bad_nbhd = np.zeros([bads.shape[0], self.options["neval"]-1], dtype=int)
        for i in range(bads.shape[0]):
            bad_nbhd[i,:] = nbhd[badlist[i]]

        # 3. Generate a criteria for each bad point

        # 3a. Take the highest eigenvalue/vector of each Hessian
        opt_dir = []
        opt_val = []
        if(self.options["interp"] == "arnoldi"):
            for i in badlist:
                opt_dir.append(hess[i][1])
                opt_val.append(hess[i][0])
        else:
            for i in badlist:
                H = hess[i]
                eigvals, eigvecs = eig(H)
                o = np.argsort(abs(eigvals))
                opt_dir.append(eigvecs[:,o[-1]])
                opt_val.append(eigvals[o[-1]])

        # we have what we need
        self.bad_list = badlist
        self.bad_eigs = opt_val
        self.bads = bads
        self.bad_nbhd = bad_nbhd
        self.bad_dirs = opt_dir
        

    def evaluate(self, x, dir=0):
        
        xc = self.bads[dir]
        xdir = self.bad_dirs[dir]
        trx = self.model.training_points[None][0][0]
        m, n = trx.shape

        # x is alpha, the distance from xc along xdir, x(alpha) = xc + alpha*xdir

        # we can use either distance, or kriging variance

        xeval = xc + x*xdir

        if(self.options["criteria"] == "distance"):
            sum = 0
            for i in range(m):
                sum += np.linalg.norm(xeval-trx[i])**2
            
            ans = -sum

        elif(self.options["criteria"] == "variance"):
            
            ans = -self.model.predict_variances(np.array([xeval]))[0,0]

        else:
            print("Invalid Criteria Option")

        return ans 

    def pre_asopt(self, bounds, dir=0):
        xc = self.bads[dir]
        gc = self.grad[self.bad_list[dir],:]
        eig = self.bad_eigs[dir]
        xdir = self.bad_dirs[dir]
        trx = self.model.training_points[None][0][0]
        nbhd = trx[self.bad_nbhd[dir],:]
        dists = pdist(np.append(np.array([xc]), nbhd, axis=0))
        dists = squareform(dists)
        B = max(dists[0,:])

        # check if we need to limit further based on bounds
        p0, p1 = boxIntersect(xc, xdir, bounds)
        bp = p1# min(B, p1)
        bm = p0# max(-B, p0)

        # choose the direction to go

        # if concave up, move up the gradient, if concave down, move down the gradient
        work = np.dot(gc, xdir)
        adir = np.sign(work)*np.sign(np.real(eig))
        if(adir > 0):
            bm = 0
        else:
            bp = 0
        return np.array([0.+adir*0.01*B]), Bounds(bm, bp)

    def post_asopt(self, x, dir=0):

        # transform back to regular coordinates

        xc = self.bads[dir]
        xdir = self.bad_dirs[dir]
        xeval = xc + x*xdir

        return xeval


        






