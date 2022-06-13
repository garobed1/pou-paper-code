"""
Implementation of direct Gradient-Enhanced Kriging in the SMT package
"""
import numpy as np
from smt.surrogate_models.krg_based import KrgBased
from smt.utils.kriging_utils import ge_compute_pls

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
from scipy.stats import multivariate_normal as m_norm
from smt.sampling_methods import LHS

class DGEK(KrgBased):
    name = "DGEK"

    def _initialize(self):
        super(DGEK, self)._initialize()
        declare = self.options.declare
        # DGEK used only with "abs_exp" and "squar_exp" correlations
        declare(
            "corr",
            "squar_exp",
            values=("abs_exp", "squar_exp"),
            desc="Correlation function type",
            types=(str),
        )
        declare(
            "xlimits",
            types=np.ndarray,
            desc="Lower/upper bounds in each dimension - ndarray [nx, 2]",
        )
        self.supports["training_derivatives"] = True

    def _componentwise_distance(self, dx, opt=0, theta=None, return_derivative=False):
        d = componentwise_distance(
            dx,
            self.options["corr"],
            self.nx,
            theta=theta,
            return_derivative=return_derivative,
        )
        return d

    # No PLS used here
    def _compute_pls(self, X, y):

        # npts = X.shape[0]
        # ndim = X.shape[1]

        # # append gradients to the y vector
        # for i in range(ndim):
        #     y = np.append(y, self.training_points[None][i+1][1], axis=0)
        #     X = np.append(X, X, axis=0) #necessary?
        

        return X, y

    def _reduced_likelihood_function(self, theta):
        """
        This function determines the BLUP parameters and evaluates the reduced
        likelihood function for the given autocorrelation parameters theta.
        Maximizing this function wrt the autocorrelation parameters theta is
        equivalent to maximizing the likelihood of the assumed joint Gaussian
        distribution of the observations y evaluated onto the design of
        experiments X.

        Parameters
        ----------
        theta: list(n_comp), optional
            - An array containing the autocorrelation parameters at which the
              Gaussian Process model parameters should be determined.

        Returns
        -------
        reduced_likelihood_function_value: real
            - The value of the reduced likelihood function associated to the
              given autocorrelation parameters theta.
        par: dict()
            - A dictionary containing the requested Gaussian Process model
              parameters:
            sigma2
            Gaussian Process variance.
            beta
            Generalized least-squares regression weights for
            Universal Kriging or for Ordinary Kriging.
            gamma
            Gaussian Process weights.
            C
            Cholesky decomposition of the correlation matrix [R].
            Ft
            Solution of the linear equation system : [R] x Ft = F
            Q, G
            QR decomposition of the matrix Ft.
        """
        # Initialize output

        reduced_likelihood_function_value = -np.inf
        par = {}
        # Set up R
        nugget = self.options["nugget"]

        # No noise evaluation
        if self.options["eval_noise"]:
            nugget = 0

        noise = self.noise0
        tmp_var = theta

        dx = 0
        dd = self.D.copy()
        for j in range(dd.shape[1]):
            dd[:,j] *= 2*theta[j]

        derivative_dic = {"dx": dx, "dd": dd}
        hess_dic = {"dx": dx, "dd": dd}
        r = self._correlation_types[self.options["corr"]](theta, self.D).reshape(
            -1, 1
        )
        r, dr = self._correlation_types[self.options["corr"]](theta, self.D, derivative_params=derivative_dic)
        d2r = self._correlation_types[self.options["corr"]](theta, self.D, hess_params=hess_dic)

        n_elem = dd.shape[0]
        n_comp = dd.shape[1]
        full_size = self.nt + self.nt*n_comp

        R = np.zeros([full_size, full_size])
        R[0:self.nt, 0:self.nt] = np.eye(self.nt) * (1.0 + nugget + noise)
        R[self.ij[:, 0], self.ij[:, 1]] = r[:, 0]
        R[self.ij[:, 1], self.ij[:, 0]] = r[:, 0]
        
        # hessian
        for k in range(n_elem):
            R[(self.nt + self.ij[k,0]*n_comp):(self.nt + self.ij[k,0]*n_comp+n_comp), (self.nt + self.ij[k,1]*n_comp):(self.nt + self.ij[k,1]*n_comp+n_comp)] = d2r[k]
            R[(self.nt + self.ij[k,1]*n_comp):(self.nt + self.ij[k,1]*n_comp+n_comp), (self.nt + self.ij[k,0]*n_comp):(self.nt + self.ij[k,0]*n_comp+n_comp)] = d2r[k]

        # upper and lower grad
        for k in range(n_elem):
            R[self.ij[k,0], (self.nt + self.ij[k,1]*n_comp):(self.nt + self.ij[k,1]*n_comp+n_comp)] = dr[k]
            R[self.ij[k,1], (self.nt + self.ij[k,0]*n_comp):(self.nt + self.ij[k,0]*n_comp+n_comp)] = -dr[k]
            R[(self.nt + self.ij[k,1]*n_comp):(self.nt + self.ij[k,1]*n_comp+n_comp), self.ij[k,0]] = -dr[k].T
            R[(self.nt + self.ij[k,0]*n_comp):(self.nt + self.ij[k,0]*n_comp+n_comp), self.ij[k,1]] = dr[k].T
        import pdb; pdb.set_trace()

        # Cholesky decomposition of R

        try:
            C = linalg.cholesky(R, lower=True)
        except (linalg.LinAlgError, ValueError) as e:
            print("exception : ", e)
            return reduced_likelihood_function_value, par

        # Get generalized least squared solution
        Ft = linalg.solve_triangular(C, self.F, lower=True)
        Q, G = linalg.qr(Ft, mode="economic")
        sv = linalg.svd(G, compute_uv=False)
        rcondG = sv[-1] / sv[0]
        if rcondG < 1e-10:
            # Check F
            sv = linalg.svd(self.F, compute_uv=False)
            condF = sv[0] / sv[-1]
            if condF > 1e15:
                raise Exception(
                    "F is too ill conditioned. Poor combination "
                    "of regression model and observations."
                )

            else:
                # Ft is too ill conditioned, get out (try different theta)
                return reduced_likelihood_function_value, par

        Yt = linalg.solve_triangular(C, self.y_norma, lower=True)
        beta = linalg.solve_triangular(G, np.dot(Q.T, Yt))
        rho = Yt - np.dot(Ft, beta)

        # The determinant of R is equal to the squared product of the diagonal
        # elements of its Cholesky decomposition C
        detR = (np.diag(C) ** (2.0 / self.nt)).prod()

        # Compute/Organize output
        p = 0
        q = 0
        if self.name in ["MFK", "MFKPLS", "MFKPLSK"]:
            p = self.p
            q = self.q
        sigma2 = (rho ** 2.0).sum(axis=0) / (self.nt - p - q)
        reduced_likelihood_function_value = -(self.nt - p - q) * np.log10(
            sigma2.sum()
        ) - self.nt * np.log10(detR)
        par["sigma2"] = sigma2 * self.y_std ** 2.0
        par["beta"] = beta
        par["gamma"] = linalg.solve_triangular(C.T, rho)
        par["C"] = C
        par["Ft"] = Ft
        par["G"] = G
        par["Q"] = Q

        if self.name in ["MGP"]:
            reduced_likelihood_function_value += self._reduced_log_prior(theta)

        # A particular case when f_min_cobyla fail
        if (self.best_iteration_fail is not None) and (
            not np.isinf(reduced_likelihood_function_value)
        ):

            if reduced_likelihood_function_value > self.best_iteration_fail:
                self.best_iteration_fail = reduced_likelihood_function_value
                self._thetaMemory = np.array(tmp_var)

        elif (self.best_iteration_fail is None) and (
            not np.isinf(reduced_likelihood_function_value)
        ):
            self.best_iteration_fail = reduced_likelihood_function_value
            self._thetaMemory = np.array(tmp_var)
        if reduced_likelihood_function_value > 1e15:
            reduced_likelihood_function_value = 1e15
        return reduced_likelihood_function_value, par
