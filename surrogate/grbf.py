"""
Implementation of direct Gradient-Enhanced RBF with squar_exp in the SMT package
"""
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
from scipy.stats import multivariate_normal as m_norm
from smt.sampling_methods import LHS

class GRBF(SurrogateModel):

    _regression_types = {"constant": constant, "linear": linear, "quadratic": quadratic}

    _basis_types = {
        "squar_exp": squar_exp,
        "matern32": matern32
    }

    name = "GRBF"

    def _initialize(self):
        super(GRBF, self)._initialize()
        declare = self.options.declare
        declare(
            "t0",
            1.0,
            types=(int, float, list, np.ndarray),
            desc="basis function scaling parameter in exp(-d^2 * t0), t0 = 1/d0**2",
        )
        declare(
            "compute_theta",
            False,
            types=(bool),
            desc="choose to compute adaptive theta depending on average distances"
        )
        declare(
            "poly",
            "constant",
            values=("constant", "linear", "quadratic"),
            desc="Regression function type",
            types=(str),
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


        self.num = num
        self.par = {}




    def _new_train(self):
        num = self.num

        xt = self.training_points[None][0][0]
        yt = self.training_points[None][0][1]
        self.nt = xt.shape[0]

        # Center and scale X and y
        (
            self.X_norma,
            self.y_norma,
            self.X_offset,
            self.y_mean,
            self.X_scale,
            self.y_std,
        ) = standardization(xt, yt)

        # Calculate matrix of distances D between samples
        D, self.ij = cross_distances(self.X_norma)
        self.D = self._componentwise_distance(D)

        if np.min(np.sum(np.abs(self.D), axis=1)) == 0.0:
            print(
                "Warning: multiple x input features have the same value (at least same row twice)."
            )
        ####

        # Regression matrix and parameters
        self.F = self._regression_types[self.options["poly"]](self.X_norma)
        n_samples_F = self.F.shape[0]
        if self.F.ndim > 1:
            p = self.F.shape[1]
        else:
            p = 1
        self._check_F(n_samples_F, p)

        # Determine theta, either by fixed values, or adaptively depending on average distance
        if self.options["compute_theta"]:
            l_adapt = np.mean(self.D, axis=0)
            t_adapt = 1./(l_adapt**2)
            self.theta = t_adapt
            import pdb; pdb.set_trace()
        else:
            self.theta = self.options["t0"]

        # jac = np.empty(num["radial"] * num["dof"])
        # #self.rbfc.compute_jac(num["radial"], xt.flatten(), jac)
        # jac = jac.reshape((num["radial"], num["dof"]))

        # Compute the GRBF covariance matrix
        dx = differences(self.X_norma, self.X_norma)
        dxx, ij = cross_distances(self.X_norma)
        dd = self._componentwise_distance(
            dxx, theta=self.theta, return_derivative=True
        )
        dx = self.D
        dd = D.copy()
        for j in range(dd.shape[1]):
            dd[:,j] *= 2*self.theta[j]

        derivative_dic = {"dx": dx, "dd": dd}
        hess_dic = {"dx": dx, "dd": dd}
        r, dr = self._basis_types[self.options["corr"]](self.theta, self.D, derivative_params=derivative_dic)
        d2r = self._basis_types[self.options["corr"]](self.theta, self.D, hess_params=hess_dic)

        # get the covariance matrix and its inverse
        P, Pg, S = getDirectCovariance(r, dr, d2r, self.theta, self.nt, self.ij)
        
        # invert R (Lockwood and Anitescu 2012)
        #defs
        Pinv = linalg.inv(P)
        PgPinv = np.dot(Pg, Pinv)
        PPginv = PgPinv.T
        M = S - np.dot(PgPinv, Pg.T)
        Minv = linalg.inv(M)

        full_size = self.nt + self.nt*dd.shape[1]
        Rinv = np.zeros([full_size, full_size])
        Rinv[0:self.nt, 0:self.nt] = Pinv + np.dot(np.dot(PPginv, Minv), PgPinv)
        Rinv[0:self.nt, self.nt:] = -np.dot(PPginv, Minv)
        Rinv[self.nt:, 0:self.nt] = -np.dot(Minv, PgPinv)
        Rinv[self.nt:, self.nt:] = Minv
        R = np.zeros([full_size, full_size])
        R[0:self.nt, 0:self.nt] = P
        R[0:self.nt, self.nt:] = Pg.T
        R[self.nt:, 0:self.nt] = Pg
        R[self.nt:, self.nt:] = S
        # Right hand side  
        # augmented y vector w/ gradients
        Ya = self.y_norma.copy()
        for i in range(self.nt):
            for j in range(dd.shape[1]):
                Ya = np.append(Ya, -self.training_points[None][j+1][1][i]*(self.X_scale[j]/self.y_std))      
        
        # augmented regression matrix
        Fa = self.F.copy()
        for i in range(self.nt):
            Fa = np.append(Fa, np.zeros([dd.shape[1], Fa.shape[1]]), axis=0)
        
        # solve the least squares problem
        beta = linalg.lstsq(Fa, Ya)[0]
        self.par["beta"] = beta

        rhs = Ya - np.dot(Fa, beta)
        #import pdb; pdb.set_trace()

        self.par["gamma"] = np.dot(Rinv, rhs)

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
        full_size = n_eval + n_eval*n_features_x

        X_cont = (x - self.X_offset) / self.X_scale
        # Get pairwise componentwise L1-distances to the input training set
        dx = differences(X_cont, Y=self.X_norma.copy())
        d = self._componentwise_distance(dx)
        dd = dx.copy()
        for j in range(dd.shape[1]):
            dd[:,j] *= 2*self.theta[j]
        derivative_dic = {"dx": dx, "dd": dd}

        # Compute the correlation function
        r = self._basis_types[self.options["corr"]](self.theta, d
        ).reshape(n_eval, self.nt)
        dum, dr = self._basis_types[self.options["corr"]](self.theta, d, derivative_params=derivative_dic)
        dr = dr.reshape(n_eval, self.nt*n_features_x)
        ra = np.zeros([n_eval, self.nt + self.nt*n_features_x])
        #import pdb; pdb.set_trace()
        for i in range(n_eval):
            ra[i] = np.append(r[i], dr[i,:])
        #import pdb; pdb.set_trace()

        y = np.zeros(full_size)
        ya = self.y_norma.copy()
        for i in range(self.nt):
            for j in range(n_features_x):
                ya = np.append(ya, self.training_points[None][j+1][1][i]*(self.X_scale[j]/self.y_std))
        
        # Compute the regression function
        f = self._regression_types[self.options["poly"]](X_cont)
        # fa = np.zeros([n_eval, self.nt + self.nt*n_features_x])
        # import pdb; pdb.set_trace()
        # for i in range(n_eval):
        #     fa[i] = np.append(f[i], np.zeros([self.nt*n_features_x]))
        # Scaled predictor

        #import pdb; pdb.set_trace()
        y_ = np.dot(f, self.par["beta"]) + np.dot(ra, self.par["gamma"])
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

    def _check_F(self, n_samples_F, p):
        """
        This function check the F-parameters of the model.
        """

        if n_samples_F != self.nt:
            raise Exception(
                "Number of rows in F and X do not match. Most "
                "likely something is going wrong with the "
                "regression model."
            )
        if p > n_samples_F:
            raise Exception(
                (
                    "Ordinary least squares problem is undetermined "
                    "n_samples=%d must be greater than the "
                    "regression model size p=%d."
                )
                % (self.nt, p)
            )
