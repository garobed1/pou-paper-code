import numpy as np
import copy

from smt.utils.options_dictionary import OptionsDictionary
from smt.surrogate_models import GEKPLS
from pougrad import POUSurrogate
from scipy.linalg import lstsq

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
        self.options = OptionsDictionary()
        self.options.update(kwargs)

        # copy the surrogate model object
        self.model = copy.deepcopy(model)

        # get the size of the training set
        kx = 0
        self.dim = self.model.training_points[None][kx][0].shape[1]
        self.ntr = self.model.training_points[None][kx][0].shape[0]

        self.nnew = 1

        self.initialize()

    def initialize(self):
        pass

    def evaluate(self, x):
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

        self.options.declare("approx", False, types=bool)

        #TODO: Add more options for this
    
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
    def evaluate(self, x):
        
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
    def pre_asopt(self):
        t0 = self.model.training_points[None][0][0]
        #import pdb; pdb.set_trace()
        diff = np.zeros(self.ntr)

        for i in range(self.ntr):
            M = self.model.predict_values(t0[[i]]).flatten()
            Mm = self.loosm[i].predict_values(t0[[i]]).flatten()
            diff[i] = abs(M - Mm)

        ind = np.argmax(diff)

        return t0[ind]

    def post_asopt(self, x):

        return x
        


# Hessian estimation and direction criteria

class HessianFit(ASCriteria):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)

        self.options.declare("criteria", "distance", types=str)
        self.options.declare("improve", 0.05, type=float)

    def initialize(self):
        
        self.nnew = int(self.ntr*self.options["improve"])
        
        trx = self.model.training_points[None][0][0]
        trf = self.model.training_points[None][0][1]
        trg = []
        for j in range(self.dim):
            trg.append(self.model.training_points[None][j+1][1]) 

        # 1. Estimate the Hessian about each point
        hess = []
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

        # 2. For every point, sum the discrepancies between the quadratic 
        # prediction in the neighborhood and the observation value
        
        #sum contributions in this vector
        err = np.zeros(self.ntr)

        for i in range(self.ntr):
            ind = indn[i]
            for key in ind:
                fh = quadratic(trx[key], trx[i], trf[i], trg[:][i], hess[i])
                err[key] += abs(trf[key] - fh)

        # 2a. Pick some percentage of the "worst" points, and their principal Hessian directions
        badlist = np.argsort(err)
        badlist = badlist[:self.nnew]

        # 3. Generate a 1D distance/variance criteria for each bad point



class EIGF(ASCriteria):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        #TODO: Add more options for this
    
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
    def evaluate(self, x):
        
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
    def pre_asopt(self):
        t0 = self.model.training_points[None][0][0]
        #import pdb; pdb.set_trace()
        diff = np.zeros(self.ntr)

        for i in range(self.ntr):
            M = self.model.predict_values(t0[[i]]).flatten()
            Mm = self.loosm[i].predict_values(t0[[i]]).flatten()
            diff[i] = abs(M - Mm)

        ind = np.argmax(diff)

        return t0[ind]

    def post_asopt(self, x):

        return x
        



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

    Hdx = h*dx
    dHd = np.dot(dx,Hdx)

    f = f0 + np.dot(g,dx) + 0.5*dHd

    return f

