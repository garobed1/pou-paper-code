"""
Implementation of a (gradient-enhanced) RBF method with a finite number of shifted basis centers, coefficients solved in a least-squares sense
"""
from re import L
import numpy as np
from scipy import linalg

from smt.surrogate_models.surrogate_model import SurrogateModel
from smt.utils.kriging_utils import differences, componentwise_distance
from smt.utils.kriging_utils import constant, linear, quadratic
from smt.utils.kriging_utils import (
    squar_exp,
    abs_exp,
    act_exp,
    standardization,
    cross_distances,
    matern52,
    matern32,
    gower_componentwise_distances,
    compute_X_cont,
    cross_levels,
    matrix_data_corr,
    compute_n_param,
)
from sutils import getDirectCovariance
from scipy.spatial.distance import pdist, squareform
from scipy.stats import multivariate_normal as m_norm
from smt.sampling_methods import LHS
from optimizers import optimize


class LSRBF(SurrogateModel):

    _regression_types = {"constant": constant, "linear": linear, "quadratic": quadratic}

    _basis_types = {
        "squar_exp": squar_exp
    }

    name = "LSRBF"

    def _initialize(self):
        super(LSRBF, self)._initialize()
        declare = self.options.declare
        declare(
            "t0",
            1.0,
            types=(int, float, list, np.ndarray),
            desc="basis function scaling parameter in exp(-d^2 * t0), t0 = 1/d0**2",
        )
        declare(
            "theta_bounds",
            [1e-6, 2e1],
            types=(list, np.ndarray),
            desc="bounds for hyperparameters",
        )
        declare(
            "compute_theta",
            False,
            types=(bool),
            desc="choose to compute adaptive theta depending on average distances"
        )
        declare(
            "basis_centers",
            2,
            types=(int, np.ndarray),
            desc="independent radial basis functions to use. if int, generate random centers"
        )
        declare(
            "corr",
            "squar_exp",
            values=(
                "squar_exp"
            ),
            desc="Basis function type",
            types=(str),
        )
        declare(
            "use_derivatives",
            False,
            types=(bool),
            desc="use gradient information in the least-squares problem"
        )


        declare("reg", 1e-10, types=(int, float), desc="Regularization coeff.")

        self.supports["derivatives"] = True
        self.supports["output_derivatives"] = True
        self.supports["training_derivatives"] = True

    def _setup(self):
        options = self.options

        nx = self.training_points[None][0][0].shape[1]
        if isinstance(options["t0"], (int, float)):
            options["t0"] = [options["t0"]] * nx
        options["t0"] = np.array(np.atleast_1d(options["t0"]), dtype=float)
        num = {}
        # number of inputs and outputs
        num["x"] = self.training_points[None][0][0].shape[1]
        num["y"] = self.training_points[None][0][1].shape[1]

        self.par = {}
        self.num = num

        #TODO: Random basis not implemented
        if(isinstance(self.options["basis_centers"], int)):
            return 1 #Do nothing for now
        else:
            self.xc = self.options["basis_centers"]




    def _new_train(self):
        num = self.num

        xt = self.training_points[None][0][0]
        yt = self.training_points[None][0][1]
        
        self.nt = xt.shape[0]
        self.nc = self.xc.shape[0]
        ndim = self.num["x"]

        # Center and scale X and y
        (
            self.X_norma,
            self.y_norma,
            self.X_offset,
            self.y_mean,
            self.X_scale,
            self.y_std,
        ) = standardization(xt, yt)

        # scale the centers as well
        self.Xc_norma = (self.xc - self.X_offset)/self.X_scale

        # Calculate matrix of distances D between samples and centers
        D, self.ij = cross_distances(self.X_norma, self.Xc_norma)
        self.D = self._componentwise_distance(D)
    
        if np.min(np.sum(np.abs(self.D), axis=1)) == 0.0:
            print(
                "Warning: multiple x input features have the same value (at least same row twice)."
            )
        ####

        # Determine theta, either by fixed values, or adaptively depending on average distance
        self.theta = np.zeros(ndim)
        if self.options["compute_theta"]:
            #l_adapt = np.min(self.D, axis=0)
            # for j in range(ndim):
            #     Ds = squareform(pdist(np.array([self.X_norma[:,j]]).T))
            #     ind = np.argsort(Ds, axis=1)
            #     l_adapt = 0
            #     for i in range(self.nt):
            #         l_adapt += Ds[i,ind[i,1]]/self.nt
            #     t_adapt = 1./(2*(l_adapt**2))
            #     self.theta[j] = 10*t_adapt/(10**ndim)

            # Rippa's method
            args = (D.copy(), True)
            opt = optimize(self.looEstimate, args, bounds=[self.options["theta_bounds"]]*ndim, x0=self.options["t0"], type='local')
            self.theta = opt["x"]
        else:
            self.theta = self.options["t0"]

        # get coefficients
        sol = self.looEstimate(self.theta, D)

        self.par["gamma"] = sol[0][0:-1]
        self.par["mean"] = sol[0][-1]

        #import pdb; pdb.set_trace()

    def _train(self):
        """
        Train the model
        """
        self._setup()
        self._new_train()

    def _componentwise_distance(self, dx, opt=0, theta=None, return_derivative=False):
        d = componentwise_distance(
            dx,
            self.options["corr"],
            self.nx,
            theta=theta,
            return_derivative=return_derivative,
        )
        return d


    def _predict_values(self, x):
        """
        Evaluates the model at a set of points.

        Parameters
        ----------
        x : np.ndarray [n_evals, dim]
            Evaluation point input variable values

        Returns
        -------
        y : np.ndarray
            Evaluation point output variable values
        """
        # Initialization
        n_eval, n_features_x = x.shape
        full_size = n_eval
        if(self.options["use_derivatives"]):
            full_size += n_eval*n_features_x

        X_cont = (x - self.X_offset) / self.X_scale
        # Get pairwise componentwise L1-distances to the input training set
        dx = differences(X_cont, Y=self.Xc_norma.copy())
        d = self._componentwise_distance(dx)
        dd = dx.copy()
        for j in range(dd.shape[1]):
            dd[:,j] *= 2*self.theta[j]
        derivative_dic = {"dx": dx, "dd": dd}

        # Compute the correlation function
        r, dr = self._basis_types[self.options["corr"]](self.theta, d, derivative_params=derivative_dic)
        r = r.reshape(n_eval, self.nc)
        dr = dr.reshape(n_eval, self.nc*n_features_x)
        #import pdb; pdb.set_trace()

        ra = r
        # if(self.options["use_derivatives"]):
        #     #import pdb; pdb.set_trace()
        #     ra = np.zeros([n_eval,self.nc + self.nc*n_features_x])
        #     for i in range(n_eval):
        #         ra[i,:] = np.append(r[i], dr[i,:])

        
        # Compute the regression function
        f = self.par["mean"]*np.ones(n_eval)
        # fa = np.zeros([n_eval, self.nt + self.nt*n_features_x])
        # import pdb; pdb.set_trace()
        # for i in range(n_eval):
        #     fa[i] = np.append(f[i], np.zeros([self.nt*n_features_x]))
        # Scaled predictor

        #import pdb; pdb.set_trace()
        y_ = f + np.dot(ra, self.par["gamma"])
        #y_ = np.dot(np.dot(ra, self.optimal_par["gamma"]), (ya - np.dot(f, self.optimal_par["beta"])))
        # Predictor

        y = (self.y_mean + self.y_std * y_).ravel()
        return y[0:n_eval]

    #TODO: Need to implement this
    def _predict_derivatives(self, x, kx):
        """
        Evaluates the derivatives at a set of points.

        Parameters
        ---------
        x : np.ndarray [n_evals, dim]
            Evaluation point input variable values
        kx : int
            The 0-based index of the input variable with respect to which derivatives are desired.

        Returns
        -------
        y : np.ndarray
            Derivative values.
        """
        # Initialization
        n_eval, n_features_x = x.shape

        x = (x - self.X_offset) / self.X_scale
        # Get pairwise componentwise L1-distances to the input training set
        dx = differences(x, Y=self.X_norma.copy())
        d = self._componentwise_distance(dx)
        # Compute the correlation function
        r = self._basis_types[self.options["corr"]](
            self.optimal_theta, d
        ).reshape(n_eval, self.nt)

        if self.options["corr"] != "squar_exp":
            raise ValueError(
                "The derivative is only available for squared exponential kernel"
            )
        if self.options["poly"] == "constant":
            df = np.zeros((1, self.nx))
        elif self.options["poly"] == "linear":
            df = np.zeros((self.nx + 1, self.nx))
            df[1:, :] = np.eye(self.nx)
        else:
            raise ValueError(
                "The derivative is only available for ordinary kriging or "
                + "universal kriging using a linear trend"
            )

        # Beta and gamma = R^-1(y-FBeta)
        beta = self.optimal_par["beta"]
        gamma = self.optimal_par["gamma"]
        df_dx = np.dot(df.T, beta)
        d_dx = x[:, kx].reshape((n_eval, 1)) - self.X_norma[:, kx].reshape((1, self.nt))
        if self.name != "Kriging" and "KPLSK" not in self.name:
            theta = np.sum(self.optimal_theta * self.coeff_pls ** 2, axis=1)
        else:
            theta = self.optimal_theta
        y = (
            (df_dx[kx] - 2 * theta[kx] * np.dot(d_dx * r, gamma))
            * self.y_std
            / self.X_scale[kx]
        )
        return y



    '''
    Objective function for RBF parameter tuning

    min e_{loo} w.r.t. theta

    e_{loo} \approx (ya^T R^-T R^-1 ya)/(N diag(R^-T R^-1))

    '''
    def looEstimate(self, theta, D, opt=False):

        # import pdb; pdb.set_trace()
        dx = 0
        #dd, ij = cross_distances(self.X_norma)
        dd = D.copy()
        ndim = dd.shape[1]
        for j in range(ndim):
            dd[:,j] *= 2*theta[j]

        derivative_dic = {"dx": dx, "dd": dd}
        hess_dic = {"dx": dx, "dd": dd}
        r, dr = self._basis_types[self.options["corr"]](theta, self.D, derivative_params=derivative_dic)
        #d2r = self._basis_types[self.options["corr"]](self.theta, self.D, hess_params=hess_dic)

        # get the covariance matrix
        ntot = r.shape[0]
        full_size = self.nt
        if(self.options["use_derivatives"]):
            full_size += ndim*self.nt

        A = np.zeros([full_size, self.nc+1])
        b = np.zeros(full_size)
        for k in range(ntot):
            A[self.ij[k][0],self.ij[k][1]] = r[k]

            if(self.options["use_derivatives"]):
                # now the derivatives 
                for j in range(ndim):
                    A[(j+1)*self.nt + self.ij[k][0],self.ij[k][1]] = dr[k,j]
            

            
        for i in range(self.nt):
            A[i,-1] = 1
            b[i] = self.y_norma[i]

            if(self.options["use_derivatives"]):
                # now the derivatives
                for j in range(ndim):
                    b[(j+1)*self.nt+i] = self.training_points[None][j+1][1][i]*self.X_scale[j]/self.y_std

        #linalg.lstsq(A, b)
        #print("cond = ", np.linalg.cond(A))
        # if(self.options["use_derivatives"]):
        #     import pdb; pdb.set_trace()
        sol = linalg.lstsq(A, b)

        cond = np.linalg.cond(A)
        if(self.options["use_derivatives"]):
            print("grad cond = ", cond)
        else:
            print("no g cond = ", cond)


        # import pdb; pdb.set_trace()
        if(opt == True):
            Ainv2 = linalg.inv(np.dot(A.T, A))
            eloo = np.dot(sol[0], sol[0])/(self.nt*np.diag(Ainv2))
            return np.sum(np.abs(eloo))
        else:
            return sol
