import numpy as np
from smt.surrogate_models.surrogate_model import SurrogateModel
from smt.utils.options_dictionary import OptionsDictionary
from collections import defaultdict
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.stats import qmc
from sutils import estimate_pou_volume



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
