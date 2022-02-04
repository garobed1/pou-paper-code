import numpy as np
from smt.surrogate_models.surrogate_model import SurrogateModel


"""
Gradient-Enhanced Partition-of-Unity Surrogate model
"""
# Might be worth making a generic base class
# Also, need to add some asserts just in case

class POUSurrogate(SurrogateModel):
    name = "POU"
    """
    Create the surrogate object

    Parameters
    ----------

    xcenter : numpy array(numsample, dim)
        Surrogate data locations
    func : numpy array(numsample)
        Surrogate data outputs
    grad : numpy array(numsample, dim)
        Surrogate data gradients
    rho : float
        Hyperparameter that controls smoothness
    delta : float
        Parameter used to regularize the distance function

    """
    def _initialize(self):#, xcenter, func, grad, rho, delta=1e-10):
        # initialize data and parameters
        super(POUSurrogate, self)._initialize()
        declare = self.options.declare

        declare(
            "rho",
            10,
            types=(int, float),
            desc="Distance scaling parameter"
        )

        declare(
            "delta",
            1e-10,
            types=(int, float),
            desc="Regularization parameter"
        )

        self.supports["training_derivatives"] = True
        # dim = len(xcenter[0]) # take the size of the first data point
        # numsample = len(xcenter)

        # self.xc = xcenter
        # self.f = func
        # self.g = grad

        # self.training_points = {}
        # self.training_points[None] = []
        # self.training_points[None].append([])
        # self.training_points[None][0].append(self.xc)
        # self.training_points[None][0].append(self.f)
        # for i in range(dim):
        #     self.training_points[None].append(self.g[i])

        # rho = rho
        # delta = delta

    """
    Add additional data to the surrogate

    Parameters
    ----------

    xcenter : numpy array(numsample, dim)
        Surrogate data locations
    func : numpy array(numsample)
        Surrogate data outputs
    grad : numpy array(numsample, dim)
        Surrogate data gradients
    """
    # def addPoints(self, xcenter, func, grad):
    #     numsample += len(xcenter)

    #     self.xc.append(xcenter)
    #     self.f.append(func)
    #     self.g.append(grad)

    #     self.training_points = {}
    #     self.training_points[None] = []
    #     self.training_points[None].append([])
    #     self.training_points[None][0].append(self.xc)
    #     self.training_points[None][0].append(self.f)
    #     for i in range(dim):
    #         self.training_points[None].append(self.g[i])

    """
    Evaluate the surrogate as-is at the point x

    Parameters
    ----------

    Parameters
        ----------
        x : np.ndarray[nt, nx]
            Input values for the prediction points.

        Returns
        -------
        y : np.ndarray[nt, ny]
            Output values at the prediction points.
        
    """
    def _predict_values(self, xt):

        xc = self.training_points[None][0][0]
        f = self.training_points[None][0][1]
        g = np.zeros([xc.shape[0],xc.shape[1]])
        numsample = xc.shape[0]
        dim = xc.shape[1]
        delta = self.options["delta"]
        rho = self.options["rho"]

        for i in range(xc.shape[1]):
            g[:,[i]] = self.training_points[None][i+1][1]
        
        # loop over rows in xt
        y = np.zeros(xt.shape[0])
        for k in range(xt.shape[0]):
            x = xt[k,:]
            # exhaustive search for closest sample point, for regularization
            mindist = 1e100
            dists = np.zeros(numsample)
            for i in range(numsample):
                dists[i] = np.sqrt(np.dot(x-xc[i],x-xc[i]) + delta)

            mindist = min(dists)

            # for i in range(numsample):
            #     dist = np.sqrt(np.dot(x-xc[i],x-xc[i]) + delta)
            #     mindist = min(mindist,dist)

            numer = 0
            denom = 0

            # evaluate the surrogate, requiring the distance from every point
            for i in range(numsample):
                dist = np.sqrt(np.dot(x-xc[i],x-xc[i]) + delta)
                local = f[i] + np.dot(g[i], x-xc[i]) # locally linear approximation
                expfac = np.exp(-rho*(dist-mindist))
                numer += local*expfac
                denom += expfac

            y[k] = numer/denom

        return y

    """
    Evaluate the gradient of the surrogate at the point x, with respect to x

    Parameters
    ----------

    x : numpy array(dim)
        Query location

    """
    def evalGrad(self, x):
        sgrad = np.zeros(dim)

        mindist = 1e100
        xc = self.xc
        f = self.f
        g = self.g
        imindist = 1e100
        

        # exhaustive search for closest sample point, for regularization
        dists = np.zeros(numsample)
        for i in range(numsample):
            dists[i] = np.sqrt(np.dot(x-xc[i],x-xc[i]) + delta)
            
        mindist = min(dists)
        imindist = np.argmin(dists)

        dmindist = (1.0/mindist)*(x-xc[imindist])

        numer = 0
        denom = 0

        dnumer = np.zeros(dim)
        ddenom = np.zeros(dim)

        sum = 0

        for i in range(numsample):
            dist = np.sqrt(np.dot(x-xc[i],x-xc[i]) + delta)
            local = f[i] + np.dot(g[i], x-xc[i]) # locally linear approximation
            expfac = np.exp(-rho*(dist-mindist))
            numer += local*expfac
            denom += expfac        

            ddist = (1.0/np.sqrt(np.dot(x-xc[i],x-xc[i])+delta))*(x-xc[i])
            dlocal = g[i]
            dexp1 = -rho*expfac
            dexp2 = rho*expfac

            dnumer += expfac*dlocal + local*(dexp1*ddist + dexp2*dmindist)
            ddenom += (dexp1*ddist + dexp2*dmindist)

            sum += (dexp1*ddist + dexp2*dmindist)
            #import pdb; pdb.set_trace()

        xgrad = (denom*dnumer - numer*ddenom)/(denom*denom)
        return xgrad