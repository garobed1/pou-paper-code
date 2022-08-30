"""
Implementation of direct Gradient-Enhanced Kriging in the SMT package
"""
import numpy as np
from scipy import linalg
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

        declare(
            "corr",
            "squar_exp",
            values=("abs_exp", "squar_exp", "matern32"),
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


        # Compute correlation kernels
        dx = differences(self.X_norma, self.X_norma)
        dxx, ij = cross_distances(self.X_norma)
        dd = self._componentwise_distance(
            dxx, theta=theta, return_derivative=True
        )
        # dx = self.D
        # dx = 0
        # dd, ij = cross_distances(self.X_norma)
        # for j in range(dd.shape[1]):
        #     dd[:,j] *= 2*theta[j]
        # import pdb; pdb.set_trace()
        derivative_dic = {"dx": dxx, "dd": dd}
        hess_dic = {"dx": dxx, "dd": dd}
        r = self._correlation_types[self.options["corr"]](theta, self.D).reshape(
            -1, 1
        )
        r, dr = self._correlation_types[self.options["corr"]](theta, self.D, derivative_params=derivative_dic)
        d2r = self._correlation_types[self.options["corr"]](theta, self.D, derivative_params=derivative_dic, hess_params=hess_dic)


        #check derivs
        # h = 1e-6
        # # self.D[0] = 0.
        # self.D[0][0] += h
        # rf = self._correlation_types[self.options["corr"]](theta, self.D)
        # self.D[0][0] -= 2*h
        # rb = self._correlation_types[self.options["corr"]](theta, self.D)
        # d2rfd = (rf[0] - 2*r[0] + rb[0])/(h*h)
        # print(d2rfd)
        # print(d2r[0])
        # import pdb; pdb.set_trace()

        # Assemble augmented correlation matrix
        n_elem = dd.shape[0]
        n_comp = dd.shape[1]
        full_size = self.nt + self.nt*n_comp

        R = np.zeros([full_size, full_size])
        P = np.eye(self.nt)
        Pg = np.zeros([self.nt*n_comp, self.nt])
        R[0:self.nt, 0:self.nt] = np.eye(self.nt) #* (1.0 + nugget + noise)


        if(self.options["corr"] == "squar_exp"):
            S = np.eye(self.nt*n_comp)
            for j in range(n_comp):
                S[j::n_comp] *= (2*theta[j])
                R[self.nt + j::n_comp, self.nt + j::n_comp] = (2*theta[j])
        elif(self.options["corr"] == "matern32"):
            S = np.eye(self.nt*n_comp)
            for j in range(n_comp):
                S[j::n_comp] *= (3*(theta[j]**2))
                R[self.nt + j::n_comp, self.nt + j::n_comp] = (3*(theta[j]**2))
        else:
            raise ValueError("Not available for this correlation kernel")

        # R[self.nt:(self.nt+self.nt*n_comp), self.nt:(self.nt+self.nt*n_comp)] = np.eye(self.nt)*(3*theta*theta)
        R[self.ij[:, 0], self.ij[:, 1]] = r[:, 0]
        R[self.ij[:, 1], self.ij[:, 0]] = r[:, 0]
        P[self.ij[:, 0], self.ij[:, 1]] = r[:, 0]
        P[self.ij[:, 1], self.ij[:, 0]] = r[:, 0]
        # print(theta)
        # print(np.linalg.cond(P))
        # import pdb; pdb.set_trace()
        # R[self.nt:, self.nt:] = S.copy()

        # hessian
        for k in range(n_elem):
            R[(self.nt + self.ij[k,0]*n_comp):(self.nt + self.ij[k,0]*n_comp+n_comp), (self.nt + self.ij[k,1]*n_comp):(self.nt + self.ij[k,1]*n_comp+n_comp)] = -d2r[k]
            R[(self.nt + self.ij[k,1]*n_comp):(self.nt + self.ij[k,1]*n_comp+n_comp), (self.nt + self.ij[k,0]*n_comp):(self.nt + self.ij[k,0]*n_comp+n_comp)] = -d2r[k].T

            S[(self.ij[k,0]*n_comp):(self.ij[k,0]*n_comp+n_comp), (self.ij[k,1]*n_comp):(self.ij[k,1]*n_comp+n_comp)] = -d2r[k]
            S[(self.ij[k,1]*n_comp):(self.ij[k,1]*n_comp+n_comp), (self.ij[k,0]*n_comp):(self.ij[k,0]*n_comp+n_comp)] = -d2r[k].T
        
        # upper and lower grad
            R[self.ij[k,0], (self.nt + self.ij[k,1]*n_comp):(self.nt + self.ij[k,1]*n_comp+n_comp)] = -dr[k]
            R[self.ij[k,1], (self.nt + self.ij[k,0]*n_comp):(self.nt + self.ij[k,0]*n_comp+n_comp)] = dr[k]
            R[(self.nt + self.ij[k,1]*n_comp):(self.nt + self.ij[k,1]*n_comp+n_comp), self.ij[k,0]] = -dr[k].T
            R[(self.nt + self.ij[k,0]*n_comp):(self.nt + self.ij[k,0]*n_comp+n_comp), self.ij[k,1]] = dr[k].T

            Pg[(self.ij[k,1]*n_comp):(self.ij[k,1]*n_comp+n_comp), self.ij[k,0]] = -dr[k].T
            Pg[(self.ij[k,0]*n_comp):(self.ij[k,0]*n_comp+n_comp), self.ij[k,1]] = dr[k].T



        # Assemble the right hand side and the regresssion
        # augmented y vector w/ gradients
        Ya = self.y_norma.copy()
        for i in range(self.nt):
            for j in range(n_comp):
                Ya = np.append(Ya, self.training_points[None][j+1][1][i]*(self.X_scale[j]/self.y_std))

        # Rinv = linalg.inv(R)
        Oa = np.ones(full_size)
        Oa[self.nt:] = 0

        
        # print(theta)
        # print(np.linalg.cond(R))
        # import pdb; pdb.set_trace()

        # Invert the augmented covariance matrix R (Lockwood and Anitescu 2012)
        #defs
        # PC = linalg.cholesky(P)
        Pinv = linalg.inv(P)
        PgPinv = np.dot(Pg, Pinv)
        PPginv = PgPinv.T
        M = S - np.dot(PgPinv, Pg.T)
        Minv = linalg.inv(M)
        # MC = linalg.cholesky(M)

        detP = linalg.det(P)
        detM = linalg.det(M)
        detR = detP*detM
        Rinv = np.zeros([full_size, full_size])
        Rinv[0:self.nt, 0:self.nt] = Pinv + np.dot(np.dot(PPginv, Minv), PgPinv)
        Rinv[0:self.nt, self.nt:] = -np.dot(PPginv, Minv)
        Rinv[self.nt:, 0:self.nt] = -np.dot(Minv, PgPinv)
        Rinv[self.nt:, self.nt:] = Minv
        # C = linalg.cholesky(R, lower=True)
        # detR = (np.diag(C) ** (2.0 / full_size)).prod()

        # Rinv2 = linalg.inv(R)

        # Get generalized least squared solution
        # augmented regression matrix
        Fa = self.F.copy()
        for i in range(self.nt):
            Fa = np.append(Fa, np.zeros([n_comp, Fa.shape[1]]), axis=0)
        # Ft = linalg.solve_triangular(L, Fa, lower=True)
        # Q, G = linalg.qr(Ft, mode="economic")
        # sv = linalg.svd(G, compute_uv=False)
        # rcondG = sv[-1] / sv[0]
        # if rcondG < 1e-10:
        #     # Check F
        #     sv = linalg.svd(self.F, compute_uv=False)
        #     condF = sv[0] / sv[-1]
        #     if condF > 1e15:
        #         raise Exception(
        #             "F is too ill conditioned. Poor combination "
        #             "of regression model and observations."
        #         )

        #     else:
        #         # Ft is too ill conditioned, get out (try different theta)
        #         return reduced_likelihood_function_value, par
        
        # regression coeffs
        beta = linalg.lstsq(Fa, Ya)[0]
        rho = Ya - np.dot(Fa, beta)
        rhos = self.y_norma.reshape(self.nt) - np.dot(self.F, beta)
        # rhot = linalg.solve_triangular(C, rho, lower=True)

        # import pdb; pdb.set_trace()


        # Compute/Organize output
        sigma2 = np.dot(np.dot(rho, Rinv), rho) /self.nt
        # sigma2 = (rho ** 2.0).sum(axis=0) / (self.nt)
        work1 = -self.nt*np.log10(sigma2) 
        work2 = -np.log10(detR)#**(1/full_size)
        reduced_likelihood_function_value = work1 + work2
        # work = detR**(1/full_size)
        # reduced_likelihood_function_value = -sigma2*(work)
        #if(sigma2 < 0):
        # import pdb; pdb.set_trace()
        par["sigma2"] = sigma2 * self.y_std ** 2.0
        par["beta"] = beta
        # par["gamma"] = linalg.solve_triangular(C.T, rhot)#np.dot(Rinv, rho)
        par["gamma"] = np.dot(Rinv, rho)#
        par["C"] = 0
        par["Ft"] = 0#Ft
        par["G"] = 0#G
        par["Q"] = 0#Q
        par["cond"] = np.linalg.cond(R)
        #import pdb; pdb.set_trace()
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
        # if reduced_likelihood_function_value > 1e15:
        #     reduced_likelihood_function_value = 1e15
        # print(sigma2)
        # print(np.linalg.cond(R))
        # print(theta)
        # print(sigma2)
        # print(detR)
        # print(work1)
        # print(work2)
        # print(np.linalg.cond(P))
        return reduced_likelihood_function_value, par





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
        dd = self._componentwise_distance(
            dx, theta=self.optimal_theta, return_derivative=True
        )
        derivative_dic = {"dx": dx, "dd": dd}
        # Compute the correlation function
        r = self._correlation_types[self.options["corr"]](self.optimal_theta, d
        ).reshape(n_eval, self.nt)
        dum, dr = self._correlation_types[self.options["corr"]](self.optimal_theta, d, derivative_params=derivative_dic)
        dr = dr.reshape(n_eval, self.nt*n_features_x)
        ra = np.zeros([n_eval, self.nt + self.nt*n_features_x])
        #import pdb; pdb.set_trace()
        for i in range(n_eval):
            ra[i] = np.append(r[i], -dr[i,:])

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
        y_ = np.dot(f, self.optimal_par["beta"]) + np.dot(ra, self.optimal_par["gamma"])
        #y_ = np.dot(np.dot(ra, self.optimal_par["gamma"]), (ya - np.dot(f, self.optimal_par["beta"])))
        # Predictor
        # import pdb; pdb.set_trace()
        y = (self.y_mean + self.y_std * y_).ravel()
        return y[0:n_eval]

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
        full_size = n_eval + n_eval*n_features_x
        full_size_t = self.nt*(1+n_features_x)

        X_cont = (x - self.X_offset) / self.X_scale
        
        # Get pairwise componentwise L1-distances to the input training set
        dx = differences(X_cont, Y=self.X_norma.copy())
        d = self._componentwise_distance(dx)
        dd = dx.copy()
        for j in range(dd.shape[1]):
            dd[:,j] *= 2*self.optimal_theta[j]
        derivative_dic = {"dx": dx, "dd": dd}
        hess_dic = {"dx": dx, "dd": dd}

        # Compute the correlation function
        r = self._correlation_types[self.options["corr"]](
            self.optimal_theta, d
        ).reshape(n_eval, self.nt)
        dum, dr = self._correlation_types[self.options["corr"]](self.optimal_theta, d, derivative_params=derivative_dic)
        # dr = dr.reshape(n_eval, self.nt*n_features_x)
        d2r = self._correlation_types[self.options["corr"]](self.optimal_theta, d, derivative_params=derivative_dic, hess_params=hess_dic)

        dra = np.zeros([n_eval, full_size_t])

        #dr/dx_k
        for i in range(n_eval):
            dra[i][0:self.nt] = dr[:, kx]
        # import pdb; pdb.set_trace()
        #d2r/dx_k dx
        for i in range(n_eval):
            for j in range(self.nt):
                dra[i][(self.nt + n_features_x*j):(self.nt + n_features_x*(j+1))] = -d2r[j][:,kx]
        # dra[:, self.nt:] = d2r[]


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
        # d_dx = X_cont[:, kx].reshape((n_eval, 1)) - self.X_norma[:, kx].reshape((1, self.nt))
        theta = self.optimal_theta
        y = (
            (df_dx[kx] + np.dot(dra, gamma))
            * self.y_std
            / self.X_scale[kx]
        )
        return y