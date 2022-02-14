import numpy as np
import copy

from smt.utils.options_dictionary import OptionsDictionary
from smt.surrogate_models import GEKPLS
from pougrad import POUSurrogate
from scipy.linalg import lstsq, eig
from scipy.spatial.distance import pdist, squareform

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

    def initialize(self):
        pass

    def evaluate(self, x, dir=0):
        pass

    def pre_asopt(self):
        pass

    def post_asopt(self):
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

    def initialize(self):

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
        
        # if(len(x.shape) != 2):
        #     x = np.array([x])

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
    def pre_asopt(self, dir=0):
        t0 = self.model.training_points[None][0][0]
        #import pdb; pdb.set_trace()
        diff = np.zeros(self.ntr)

        for i in range(self.ntr):
            M = self.model.predict_values(t0[[i]]).flatten()
            Mm = self.loosm[i].predict_values(t0[[i]]).flatten()
            diff[i] = abs(M - Mm)

        ind = np.argmax(diff)

        return t0[ind]

    def post_asopt(self, x, dir=0):

        return x
        


# Hessian estimation and direction criteria

class HessianFit(ASCriteria):
    def __init__(self, model, **kwargs):

        self.bads = None
        self.bad_dirs = None

        super().__init__(model, **kwargs)

        
        
    def _init_options(self):
        #options: neighborhood, surrogate, exact
        self.options.declare("hessian", "surrogate", types=str)

        #options: distance, variance, random
        self.options.declare("criteria", "distance", types=str)
        self.options.declare("improve", 0.05, types=float)

        #number of closest points to evaluate nonlinearity measure
        self.options.declare("neval", self.dim*2, types=int)
        
    def initialize(self):
        


        self.nnew = int(self.ntr*self.options["improve"])
        if(self.nnew == 0):
            self.nnew = 1
        
        trx = self.model.training_points[None][0][0]
        trf = self.model.training_points[None][0][1]
        trg = np.zeros_like(trx)
        for j in range(self.dim):
            trg[:,j] = self.model.training_points[None][j+1][1].flatten()

        # 1. Estimate the Hessian about each point
        hess = []

        if(self.options["hessian"] == "neighborhood"):
            pts = []
            indn = []

            # 1a. Determine the neighborhood to fit the Hessian for each point
            for i in range(self.ntr):
                ind = neighborhood(i, trx)
                pts.append(np.array(trx[ind,:]))
                indn.append(ind)

            # 1b. Fit a Hessian over the points using least squares
            for i in range(self.ntr):
                hess.append(np.zeros((self.dim, self.dim)))

                # Solve P*h_j = g_j for each direction in a least-squares sense
                P = pts[i] - trx[i]
                for j in range(self.dim):
                    hj = np.zeros(self.dim)
                    gj = trg[j]

                    hj = lstsq(P, gj)

                    hess[i][:,j] = hj

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

        # 2. For every point, sum the discrepancies between the quadratic 
        # prediction in the neighborhood and the observation value
        
        #sum contributions in this vector
        err = np.zeros(self.ntr)

        dists = pdist(trx)
        dists = squareform(dists)
        for i in range(self.ntr):
            #ind = indn[i]
            ind = dists[i,:]
            ind = np.argsort(ind)
            for key in ind[1:self.options["neval"]]:
                fh = quadratic(trx[key], trx[i], trf[i], trg[:][i], hess[i])
                err[i] += abs(trf[key] - fh)
            err[i] /= self.options["neval"]

        # 2a. Pick some percentage of the "worst" points, and their principal Hessian directions
        badlist = np.argsort(err)
        badlist = badlist[-self.nnew:]
        bads = trx[badlist]

        # 3. Generate a 1D distance/variance criteria for each bad point

        # 3a. Take the highest eigenvalue/vector of each Hessian
        opt_dir = []
        for i in badlist:
            H = hess[i]
            eigvals, eigvecs = eig(H)
            o = np.argsort(eigvals)
            opt_dir.append(eigvecs[o[-1]])

        import pdb; pdb.set_trace()
        # we have what we need
        self.bads = bads
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
            
            ans = -self.model.predict_variances(xeval)

        else:
            print("Invalid Criteria Option")

        return ans 

    def pre_asopt(self, dir=0):
        xc = self.bads[dir]

        return xc

    def post_asopt(self, x, dir=0):

        # transform back to regular coordinates

        xc = self.bads[dir]
        xdir = self.bad_dirs[dir]
        xeval = xc + x*xdir

        return xeval


        



def neighborhood(i, trx):
    """
    Determine an "optimal" neighborhood around a data point for estimating the 
    Hessian, based on the closest points that best surround the point
    
    Inputs:
        i - index of point to determine neighborhood of
        trx - full list of data points
    Outputs:
        ind - indices of points to include in the neighborhood
    """
    ind = []
    return ind


def quadratic(x, x0, f0, g, h):
    """
    Given the gradient and Hessian about a nearby point, return the quadratic
    Taylor series approximation of the function
    
    f(x) = f(x0) + g(x0)^T*(x-x0) + (1/2)*(x-x0)^T*h(x0)*(x-x0) + O((x-x0)^3)

    Inputs:
        x - point to evaluate the approximation
        x0 - center point of the Taylor series
        f0 - function value at the center
        g - gradient at the center
        h - Hessian at the center
    Outputs:
        f - quadratic Taylor series approximation at x
    """

    dx = x - x0

    Hdx = np.matmul(h,dx.T)
    dHd = np.dot(dx,Hdx)
    f = f0 + np.dot(g,dx) + 0.5*dHd

    return f

