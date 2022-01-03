import numpy as np



"""
Gradient-Enhanced Partition-of-Unity Surrogate model
"""
# Might be worth making a generic base class
# Also, need to add some asserts just in case

class POUSurrogate():
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
    def __init__(self, xcenter, func, grad, rho, delta=1e-10):
        # initialize data and parameters
        self.dim = len(xcenter(0)) # take the size of the first data point
        self.numsample = len(xcenter)

        self.xc = xcenter
        self.f = func
        self.g = grad

        self.rho = rho
        self.delta = delta

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
    def addPoints(self, xcenter, func, grad):
        self.numsample += len(xcenter)

        self.xc.append(xcenter)
        self.f.append(func)
        self.g.append(grad)

    """
    Evaluate the surrogate as-is at the point x

    Parameters
    ----------

    x : numpy array(dim)
        Query location
    """
    def eval(self, x):

        mindist = 1e100
        xc = self.xc
        f = self.f
        g = self.g

        # exhaustive search for closest sample point, for regularization
        for i in range(self.numsample):
            dist = np.sqrt(np.dot(x-xc(i),x-xc(i)) + self.delta)
            mindist = min(mindist,dist)

        numer = 0
        denom = 0

        # evaluate the surrogate, requiring the distance from every point
        for i in range(self.numsample):
            dist = np.sqrt(np.dot(x-xc(i),x-xc(i)) + self.delta)
            local = f(i) + np.dot(g(i, x-xc(i))) # locally linear approximation
            expfac = np.exp(-self.rho*(dist-mindist))
            numer += local*expfac
            denom += expfac

        return numer/denom

    """
    Evaluate the gradient of the surrogate at the point x, with respect to x

    Parameters
    ----------

    x : numpy array(dim)
        Query location

    sgrad : numpy array(dim)
        Gradient output
    """
    def evalGrad(self, x, sgrad):
        sgrad = np.zeros(self.dim)
        return