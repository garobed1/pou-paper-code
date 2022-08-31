import numpy as np
import time

from smt.surrogate_models.surrogate_model import SurrogateModel
from smt.utils.options_dictionary import OptionsDictionary
from collections import defaultdict
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.stats import qmc
from sutils import estimate_pou_volume, innerMatrixProduct, quadraticSolveHOnly, symMatfromVec
from sutils import standardization2
#from pou_cython_ext import POUEval

"""
Gradient-Enhanced Partition-of-Unity Surrogate model
"""
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
            "bounds",
            None,
            types=(list, np.ndarray),
            desc="Domain boundaries"
        )

        declare(
            "rho",
            10,
            types=(int, float),
            desc="Distance scaling parameter"
        )
        declare(
            "rscale",
            None,
            types=(int, float),
            desc="Scaling factor for auto refining rho"
        )

        declare(
            "delta",
            1e-10,
            types=(int, float),
            desc="Regularization parameter"
        )

        self.supports["training_derivatives"] = True

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

        X_cont = (xt - self.X_offset) / self.X_scale
        xc = self.X_norma
        f = self.y_norma
        g = self.g_norma
        h = self.h
        numsample = xc.shape[0]
        delta = self.options["delta"]
        rho = self.options["rho"]

        if(self.options["rscale"]):
            rho = self.options["rscale"]*pow(numsample, 1./xc.shape[1])

        # y_ = POUEval(X_cont, xc, f, g, h, delta, rho)

        # loop over rows in xt
        y_ = np.zeros(xt.shape[0])
        for k in range(xt.shape[0]):
            x = X_cont[k,:]

            # exhaustive search for closest sample point, for regularization
            D = cdist(np.array([x]),xc)
            mindist = min(D[0])

            numer = 0
            denom = 0

            # evaluate the surrogate, requiring the distance from every point
            # for i in range(numsample):
            work = x - xc
            dist = D[0][:] + delta#np.sqrt(D[0][i] + delta)
            expfac = np.exp(-rho*(dist-mindist))
            local = np.zeros(numsample)
            for i in range(numsample):
                local[i] = f[i] + self.higher_terms(work[i], g[i], h[i])
            numer = np.dot(local, expfac)
            denom = np.sum(expfac)
            # t2 = time.time()

            # exec1 += t1-t0
            # exec2 += t2-t1
            

            y_[k] = numer/denom

        y = (self.y_mean + self.y_std * y_).ravel()
        # print("mindist  = ", exec1)
        # print("evaluate = ", exec2)

        return y


    def higher_terms(self, dx, g, h):
        return np.dot(g, dx)


    def _train(self):
        xc = self.training_points[None][0][0]
        f = self.training_points[None][0][1]

        self.dim = xc.shape[1]
        self.ntr = xc.shape[0]

        # Center and scale X and y
        (
            self.X_norma,
            self.y_norma,
            self.X_offset,
            self.y_mean,
            self.X_scale,
            self.y_std,
        ) = standardization2(xc, f, self.options["bounds"])

        self.g_norma = np.zeros([xc.shape[0], xc.shape[1]])
        
        self.g_norma = np.zeros([xc.shape[0], xc.shape[1]])
        
        for i in range(self.dim):
            self.g_norma[:,[i]] = self.training_points[None][i+1][1]*(self.X_scale[i]/self.y_std)

        self.h = np.zeros([xc.shape[0], self.dim, self.dim])
        
        # self.dV = estimate_pou_volume(self.training_points[None][0][0], self.options["bounds"])

'''
First-order POU surrogate with Hessian estimation
'''
class POUHessian(POUSurrogate):
    name = "POUHessian"
    def _initialize(self):
        # initialize data and parameters
        super(POUHessian, self)._initialize()
        declare = self.options.declare

        declare(
            "bounds",
            None,
            types=(list, np.ndarray),
            desc="Domain boundaries"
        )

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
        
        declare(
            "neval", 
            3, 
            types=int,
            desc="number of closest points to evaluate hessian estimate")

        self.supports["training_derivatives"] = True

    def higher_terms(self, dx, g, h):
        terms = np.dot(g, dx)
        terms += 0.5*innerMatrixProduct(h, dx)
        return terms


    def _train(self):
        xc = self.training_points[None][0][0]
        f = self.training_points[None][0][1]

        self.dim = xc.shape[1]
        self.ntr = xc.shape[0]

        # Center and scale X and y
        (
            self.X_norma,
            self.y_norma,
            self.X_offset,
            self.y_mean,
            self.X_scale,
            self.y_std,
        ) = standardization2(xc, f, self.options["bounds"])

        self.g_norma = np.zeros([xc.shape[0], xc.shape[1]])
        
        for i in range(self.dim):
            self.g_norma[:,[i]] = self.training_points[None][i+1][1]*(self.X_scale[i]/self.y_std)
        # hessian estimate
        indn = []
        nstencil = self.options["neval"]
        tree = KDTree(self.X_norma)
        # dists = pdist(self.xc)
        # dists = squareform(dists)
        # mins = np.zeros(self.ntr)
        for i in range(self.ntr):
            dists, ind = tree.query(self.X_norma[i], nstencil)
            indn.append(ind)
        hess = []

        for i in range(self.ntr):
            Hh = quadraticSolveHOnly(self.X_norma[i,:], self.X_norma[indn[i][1:nstencil],:], \
                                     self.y_norma[i], self.y_norma[indn[i][1:nstencil]], \
                                     self.g_norma[i,:], self.g_norma[indn[i][1:nstencil],:])

            hess.append(np.zeros([self.dim, self.dim]))
            for j in range(self.dim):
                for k in range(self.dim):
                    hess[i][j,k] = Hh[symMatfromVec(j,k,self.dim)]

        self.h = hess

        # self.dV = estimate_pou_volume(self.training_points[None][0][0], self.options["bounds"])






















class POUMetric():
    name = "POUMetric"
    """
    Create the surrogate object

    Parameters
    ----------

    xcenter : numpy array(numsample, dim)
        Surrogate data locations
    func : numpy array(numsample, dim, dim)
        Surrogate data outputs, in this case, the matrix that defines the anisotropic metric
    rho : float
        Hyperparameter that controls smoothness
    delta : float
        Parameter used to regularize the distance function

    """
    def __init__(self, **kwargs):
        self.options = OptionsDictionary()

        self.supports = supports = {}
        supports["training_derivatives"] = False
        supports["derivatives"] = False
        supports["output_derivatives"] = False
        supports["adjoint_api"] = False
        supports["variances"] = False
        supports["variance_derivatives"] = False

        declare = self.options.declare

        declare(
            "print_global",
            True,
            types=bool,
            desc="Global print toggle. If False, all printing is suppressed",
        )
        declare(
            "print_training",
            True,
            types=bool,
            desc="Whether to print training information",
        )
        declare(
            "print_prediction",
            True,
            types=bool,
            desc="Whether to print prediction information",
        )
        declare(
            "print_problem",
            True,
            types=bool,
            desc="Whether to print problem information",
        )
        declare(
            "print_solver", True, types=bool, desc="Whether to print solver information"
        )
        self.initialize()
        self.options.update(kwargs)
        self.training_points = defaultdict(dict)


    def initialize(self):#, xcenter, func, grad, rho, delta=1e-10):
        # initialize data and parameters
        #super(POUMetric, self)._initialize()
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

        declare(
            "metric",
            None,
            types=np.ndarray,
            desc="Actual training outputs"
        )

        self.supports["training_derivatives"] = False

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
        y : np.ndarray[nt, ny, ny]
            Output values at the prediction points.
        
    """
    def predict_values(self, xt):

        xc = self.training_points[None][0][0]
        f = self.options["metric"]
        numsample = xc.shape[0]
        dim = xc.shape[1]
        delta = self.options["delta"]
        rho = self.options["rho"]
        
        # loop over rows in xt
        y = np.zeros([xt.shape[0], dim, dim])
        for k in range(xt.shape[0]):
            x = xt[k,:]
            # exhaustive search for closest sample point, for regularization
            # mindist = 1e100
            # dists = np.zeros(numsample)
            # for i in range(numsample):
            #     dists[i] = np.sqrt(np.dot(x-xc[i],x-xc[i]) + delta)

            # mindist = min(dists)
            
            mindist = min(cdist(np.array([x]),xc)[0])

            # for i in range(numsample):
            #     dist = np.sqrt(np.dot(x-xc[i],x-xc[i]) + delta)
            #     mindist = min(mindist,dist)

            numer = np.zeros([dim, dim])
            denom = 0

            # evaluate the surrogate, requiring the distance from every point
            for i in range(numsample):
                work = x-xc[i]
                dist = np.sqrt(np.dot(work,work) + delta)
                local = f[i]
                expfac = np.exp(-rho*(dist-mindist))
                numer += local*expfac
                denom += expfac
            y[k,:,:] = numer/denom

        return y


    def predict_derivatives(self, xt):

        xc = self.training_points[None][0][0]
        f = self.options["metric"]
        numsample = xc.shape[0]
        dim = xc.shape[1]
        delta = self.options["delta"]
        rho = self.options["rho"]
        
        # loop over rows in xt
        dydx = np.zeros([xt.shape[0], dim, dim, dim])
        for k in range(xt.shape[0]):
            x = xt[k,:]

            # mindist = min(dists)
            mindist = min(cdist(np.array([x]),xc)[0])

            t1 = time.time()

            numer = np.zeros([dim, dim])
            dnumer = np.zeros([dim, dim, dim])
            denom = 0
            ddenom = np.zeros([dim])

            # evaluate the surrogate, requiring the distance from every point
            for i in range(numsample):
                work = x-xc[i]
                dist = np.sqrt(np.dot(work,work) + delta)
                ddist = (1./dist)*work
                local = f[i]
                expfac = np.exp(-rho*(dist-mindist))
                dexpfac = -rho*expfac*ddist
                for j in range(dim):
                    dnumer[j,:,:] += local*dexpfac[j]
                ddenom += dexpfac
                numer += local*expfac
                denom += expfac
            
            t2 = np.zeros([dim, dim, dim])
            for j in range(dim):
                t2[j,:,:] = ddenom[j]*numer
            dydx[k,:,:,:] = (denom*dnumer - t2)/(denom**2)
        return dydx

    def set_training_values(self, xt: np.ndarray, yt: np.ndarray, name=None) -> None:
        """
        Set training data (values).

        Parameters
        ----------

        """

        if xt.shape[0] != yt.shape[0]:
            raise ValueError(
                "the first dimension of xt and yt must have the same length"
            )

        self.nt = xt.shape[0]
        # self.nx = xt.shape[1]
        # self.ny = yt.shape[1]
        kx = 0
        self.training_points[name][kx] = [np.array(xt), np.array(yt)]



# Special POU Model that measures the difference between the FOTA and another surrogate model
class POUError():
    name = "POUError"
    """
    Create the surrogate object

    Parameters
    ----------

    xcenter : numpy array(numsample, dim)
        Surrogate data locations
    func : numpy array(numsample, dim, dim)
        Surrogate data outputs, in this case, the matrix that defines the anisotropic metric
    rho : float
        Hyperparameter that controls smoothness
    delta : float
        Parameter used to regularize the distance function

    """
    def __init__(self, **kwargs):
        self.options = OptionsDictionary()

        self.supports = supports = {}
        supports["training_derivatives"] = False
        supports["derivatives"] = False
        supports["output_derivatives"] = False
        supports["adjoint_api"] = False
        supports["variances"] = False
        supports["variance_derivatives"] = False

        declare = self.options.declare

        declare(
            "print_global",
            True,
            types=bool,
            desc="Global print toggle. If False, all printing is suppressed",
        )
        declare(
            "print_training",
            True,
            types=bool,
            desc="Whether to print training information",
        )
        declare(
            "print_prediction",
            True,
            types=bool,
            desc="Whether to print prediction information",
        )
        declare(
            "print_problem",
            True,
            types=bool,
            desc="Whether to print problem information",
        )
        declare(
            "print_solver", True, types=bool, desc="Whether to print solver information"
        )
        self.dV = None
        self.initialize()
        self.options.update(kwargs)
        self.training_points = defaultdict(dict)


    def initialize(self):#, xcenter, func, grad, rho, delta=1e-10):
        # initialize data and parameters
        #super(POUMetric, self)._initialize()
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

        declare(
            "xscale",
            np.zeros(1),
            types=np.ndarray,
            desc="Unscaled bounds of underlying surrogate"
        )

        self.supports["training_derivatives"] = True

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
        y : np.ndarray[nt, ny, ny]
            Output values at the prediction points.
        
    """
    def predict_values(self, xt, model):

        xc = self.training_points[None][0][0]
        f = self.training_points[None][0][1]
        g = np.zeros([xc.shape[0],xc.shape[1]])
        for i in range(xc.shape[1]):
            g[:,i] = self.training_points[None][i+1][1]
        numsample = xc.shape[0]
        dim = xc.shape[1]
        delta = self.options["delta"]
        rho = self.options["rho"]
        bounds = self.options["xscale"]
        
        # loop over rows in xt
        y = np.zeros([xt.shape[0]])
        for k in range(xt.shape[0]):
            x = xt[k,:]
            xscale = qmc.scale(np.array([x]), bounds[:,0], bounds[:,1])
            

            # exhaustive search for closest sample point, for regularization
            mindist = min(cdist(np.array([x]),xc)[0])

            numer = 0
            denom = 0
            xm = model.predict_values(xscale)

            # evaluate the surrogate, requiring the distance from every point
            for i in range(numsample):
                work = x-xc[i]
                dist = np.sqrt(np.dot(work,work) + delta)
                local = abs(f[i] + np.dot(g[i], work) - xm)*self.volume_weight(i)
                expfac = np.exp(-rho*(dist-mindist))
                numer += local*expfac
                denom += expfac
            y[k] = numer/denom

        return y


    def predict_derivatives(self, xt, model):

        xc = self.training_points[None][0][0]
        f = self.training_points[None][0][1]
        g = np.zeros([xc.shape[0],xc.shape[1]])
        for i in range(xc.shape[1]):
            g[:,i] = self.training_points[None][i+1][1]
        numsample = xc.shape[0]
        dim = xc.shape[1]
        delta = self.options["delta"]
        rho = self.options["rho"]
        bounds = self.options["xscale"]

        # loop over rows in xt
        dydx = np.zeros([xt.shape[0], dim])
        for k in range(xt.shape[0]):
            x = xt[k,:]
            xscale = qmc.scale(np.array([x]), bounds[:,0], bounds[:,1])
            dxscale = bounds[:,1] - bounds[:,0]

            mindist = min(cdist(np.array([x]),xc)[0])

            numer = 0
            dnumer = np.zeros([dim])
            denom = 0
            ddenom = np.zeros([dim])
            gm = np.zeros([dim])
            for j in range(dim):
                gm[j] = model.predict_derivatives(xscale, j)
            gm = np.multiply(gm, dxscale)
            plocal2 = -model.predict_values(xscale)
            # evaluate the surrogate, requiring the distance from every point
            for i in range(numsample):
                work = x-xc[i]
                dist = np.sqrt(np.dot(work,work) + delta)
                ddist = (1./dist)*work
                plocal1 = f[i] + np.dot(g[i], work) 
                local = abs(plocal1 + plocal2)*self.volume_weight(i)
                dlocal = (g[i] - gm)*np.sign(plocal1 + plocal2)*self.volume_weight(i)
                expfac = np.exp(-rho*(dist-mindist))
                dexpfac = -rho*expfac*ddist
                dnumer += (local*dexpfac + dlocal*expfac)[0]
                ddenom += dexpfac
                numer += local*expfac
                denom += expfac
            

            t2 = ddenom*numer
            dydx[k,:] = (denom*dnumer - t2)/(denom**2)
        return dydx

    def set_training_values(self, xt: np.ndarray, yt: np.ndarray, name=None) -> None:
        """
        Set training data (values).

        Parameters
        ----------

        """

        if xt.shape[0] != yt.shape[0]:
            raise ValueError(
                "the first dimension of xt and yt must have the same length"
            )

        self.nt = xt.shape[0]
        kx = 0
        self.training_points[name][kx] = [np.array(xt), np.array(yt)]

    def set_training_derivatives(self, xt: np.ndarray, yt: np.ndarray, kx: int, name=None) -> None:
        """
        Set training data (gradient).

        Parameters
        ----------

        """

        if xt.shape[0] != yt.shape[0]:
            raise ValueError(
                "the first dimension of xt and yt must have the same length"
            )

        self.nt = xt.shape[0]
        self.training_points[name][kx+1] = [np.array(xt), np.array(yt)]

    def volume_weight(self, ind):
        return 1.0


class POUErrorVol(POUError):

    def volume_weight(self, ind):
        return self.dV[ind]

    def set_training_values(self, xt: np.ndarray, yt: np.ndarray, name=None) -> None:
        """
        Set training data (values).

        Parameters
        ----------

        """

        if xt.shape[0] != yt.shape[0]:
            raise ValueError(
                "the first dimension of xt and yt must have the same length"
            )

        self.nt = xt.shape[0]
        kx = 0
        self.training_points[name][kx] = [np.array(xt), np.array(yt)]
        
        m, n = self.training_points[None][0][0].shape
        bounds = np.zeros([n,2])
        bounds[:,1] = 1.
        self.dV = estimate_pou_volume(self.training_points[None][0][0], bounds)





class POUCV(SurrogateModel): 
    name = "POU"
    """
    Computes cross-validation error of POU model

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
        super(POUCV, self)._initialize()
        declare = self.options.declare
        declare(
            "pmodel",
            desc="POU Model of Interest",
            types=(SurrogateModel),
        )


    """
    Evaluate the CV error at point x

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

        pmodel = self.options["pmodel"]

        xc = pmodel.X_norma
        f = pmodel.y_norma
        g = pmodel.g_norma
        h = pmodel.h
        numsample = xc.shape[0]
        delta = pmodel.options["delta"]
        rho = pmodel.options["rho"]

        # y_ = POUEval(X_cont, xc, f, g, h, delta, rho)

        # loop over rows in xt
        y_ = np.zeros(xt.shape[0])
        for k in range(xt.shape[0]):
            x = xt[k,:]

            # exhaustive search for closest sample point, for regularization
            D = cdist(np.array([x]),xc)
            mindist = min(D[0])

            numer = 0
            denom = 0

            # evaluate the surrogate, requiring the distance from every point
            # for i in range(numsample):
            work = x - xc
            dist = D[0][:] + delta#np.sqrt(D[0][i] + delta)
            expfac = np.exp(-rho*(dist-mindist))
            local = np.zeros(numsample)
            for i in range(numsample):
                local[i] = f[i] + pmodel.higher_terms(work[i], g[i], h[i])
            numer = np.dot(local, expfac)
            denom = np.sum(expfac)
            # t2 = time.time()

            # exec1 += t1-t0
            # exec2 += t2-t1
            

            y_base = numer/denom
            y_[k] = 0
            for i in range(numsample):
                y_i = (numer - local[i]*expfac[i])/(denom - expfac[i])
                y_[k] += (y_base - y_i)**2
            y_[k] = np.sqrt(y_[k]/numsample)

        y = y_.ravel()
        # print("mindist  = ", exec1)
        # print("evaluate = ", exec2)

        return y


