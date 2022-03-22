"""
Implements the heaviside function
"""
from cmath import cos, sin
import numpy as np

from smt.problems.problem import Problem

class Heaviside(Problem):
    def _initialize(self):
        self.options.declare("ndim", 1, values=[2], types=int)
        self.options.declare("name", "Heaviside", types=str)

    def _setup(self):
        self.xlimits[:, 0] = 0.0
        self.xlimits[:, 1] = 1.0

    def _evaluate(self, x, kx):
        """
        Arguments
        ---------
        x : ndarray[ne, nx]
            Evaluation points.
        kx : int or None
            Index of derivative (0-based) to return values with respect to.
            None means return function value rather than derivative.

        Returns
        -------
        ndarray[ne, 1]
            Functions values if kx=None or derivative values if kx is an int.
        """
        ne, nx = x.shape
        y = np.zeros((ne, 1), complex)

        for i in range(ne):
            if kx is None:
                if(x[i,0] < 0.5):
                    y[i,0] = 0
                else:
                    y[i,0] = 1
            else:
                y[:,0] = 0

        return y



class Quad2D(Problem):
    def _initialize(self):
        self.options.declare("ndim", 2, values=[2], types=int)
        self.options.declare("theta", 0., types=float)
        self.options.declare("name", "Quad2D", types=str)

    def _setup(self):
        self.xlimits[:, 0] = 0.0
        self.xlimits[:, 1] = 1.0

    def _evaluate(self, x, kx):
        """
        Arguments
        ---------
        x : ndarray[ne, nx]
            Evaluation points.
        kx : int or None
            Index of derivative (0-based) to return values with respect to.
            None means return function value rather than derivative.

        Returns
        -------
        ndarray[ne, 1]
            Functions values if kx=None or derivative values if kx is an int.
        """
        ne, nx = x.shape
        th = self.options["theta"]
        y = np.zeros((ne, 1), complex)

        A = np.array([[1, 0],[0, 0]])
        R = np.array([[cos(th), -sin(th)],[sin(th), cos(th)]])
        B = np.matmul(A,R)
        B = np.matmul(R.T,B)
        # y = x^T R^T A R x

        for i in range(ne):
            if kx is None:
                #y[i,0] = x[i,0]*x[i,0]
                y[i,0] = np.matmul(x[i].T, np.matmul(B,x[i]))
            # elif(kx == 0):
            #     y[i,0] = 2*x[i,0]
            else:
                yt = np.matmul(x[i].T, (B + B.T))
                y[i,0] = yt[kx]

        return y


# From Fuhg (2021), Problem 8
class FuhgP8(Problem):
    def _initialize(self):
        self.options.declare("ndim", 2, values=[2], types=int)
        self.options.declare("name", "FuhgP8", types=str)

    def _setup(self):
        self.xlimits[:, 0] = -2.5
        self.xlimits[:, 1] = 2.5

    def _evaluate(self, x, kx):
        """
        Arguments
        ---------
        x : ndarray[ne, nx]
            Evaluation points.
        kx : int or None
            Index of derivative (0-based) to return values with respect to.
            None means return function value rather than derivative.

        Returns
        -------
        ndarray[ne, 1]
            Functions values if kx=None or derivative values if kx is an int.
        """
        ne, nx = x.shape
        y = np.zeros((ne, 1), complex)

        for i in range(ne):
            X = x[i,:]
            if kx is None:
                y[i,0] = 100*(X[1] - X[0]*X[0])**2 + (X[0] - 1)**2
                if(X[1] > 1.5):
                    y[i,0] += (X[0] - 1)**2 + 700*X[0]*X[1]
                else: 
                    y[i,0] += (X[0] - 1)**2

            elif(kx == 0):
                y[i,0] = 100*2*(X[1] - X[0]*X[0])*-2*X[0] + 2*(X[0] - 1)
                if(X[1] > 1.5):
                    y[i,0] += 2*(X[0] - 1) + 700*X[1]
                else: 
                    y[i,0] += 2*(X[0] - 1)

            elif(kx == 1):
                y[i,0] = 100*2*(X[1] - X[0]*X[0])
                if(X[1] > 1.5):
                    y[i,0] += 700*X[0]
                else: 
                    y[i,0] += 0

        return y

# From Panda (2019)
class QuadHadamard(Problem):
    def _initialize(self):
        self.options.declare("ndim", 2, values=[2], types=int)
        self.options.declare("eigenrate", 1., types=float)
        self.options.declare("name", "QuadHadamard", types=str)

        self.eigen_vals = None
        self.eigen_vectors = None
        self.eigen_decayrate = None
        self.dim = None

    def _setup(self):
        self.xlimits[:, 0] = -10.
        self.xlimits[:, 1] = 10.

        self.dim = self.options["ndim"]
        self.eigen_vals = np.zeros(self.dim)
        self.eigen_vectors = np.zeros((self.dim, self.dim))
        self.eigen_decayrate = self.options["eigenrate"]
        self.getSyntheticEigenValues()
        self.getSyntheticEigenVectors()

    def _evaluate(self, x, kx):
        """
        Arguments
        ---------
        x : ndarray[ne, nx]
            Evaluation points.
        kx : int or None
            Index of derivative (0-based) to return values with respect to.
            None means return function value rather than derivative.

        Returns
        -------
        ndarray[ne, 1]
            Functions values if kx=None or derivative values if kx is an int.
        """
        ne, nx = x.shape
        y = np.zeros((ne, 1), complex)
        # y = x^T R^T A R x

        for i in range(ne):
            if kx is None:
                xi_hat = np.zeros(self.dim)
                self.applyHadamard(x[i,:], xi_hat)
                y[i,0] = np.dot(xi_hat, self.eigen_vals*xi_hat)

            else:
                xi_hat = np.zeros(self.dim)
                dfdrv = np.zeros(self.dim)
                self.applyHadamard(x[i,:], xi_hat)
                xi_hat = 2*self.eigen_vals*xi_hat
                self.applyHadamard(xi_hat, dfdrv)
                y[i,0] = dfdrv[kx]

        return y

    def getHessian(self):
        return 2*np.dot(self.eigen_vectors,(self.eigen_vals*self.eigen_vectors.T).T)

    def applyHadamard(self, x, y):
        """
        Multiplies `x` by a scaled orthonormal Hadamard matrix and returns `y`.
        This method uses Sylvester's construction and thus results in a
        symmetric Hadamrad matrix with trace zero.
        """

        n = np.size(x,0)
        assert n % 2  == 0, "size of x must be a power of 2"

        # Convert to 2D because numpy complains
        if x.ndim == 1:
            x_2D = x[:, np.newaxis]
            y_2D = y[:, np.newaxis]
        else:
            x_2D = x
            y_2D = y

        fac = 1.0/np.sqrt(2)
        if n == 2:
            y_2D[0,:] = fac*(x_2D[0,:] + x_2D[1,:])
            y_2D[1,:] = fac*(x_2D[0,:] - x_2D[1,:])
        else:
            n2 = n // 2
            Hx1 = np.zeros((n2, np.size(x_2D,1)))
            Hx2 = np.zeros((n2, np.size(x_2D,1)))
            self.applyHadamard(x_2D[0:n2,:], Hx1)
            self.applyHadamard(x_2D[n2:n,:], Hx2)
            y_2D[0:n2,:] = fac*(Hx1 + Hx2)
            y_2D[n2:n,:] = fac*(Hx1 - Hx2)

    def getSyntheticEigenValues(self):
        for i in range (0, self.dim):
            self.eigen_vals[i] = 1/(i+1)**self.eigen_decayrate

    def getSyntheticEigenVectors(self):
        iden = np.eye(self.dim)
        self.applyHadamard(iden, self.eigen_vectors)


# f(x) = arctan(alpha*(x dot t))
class MultiDimJump(Problem):
    def _initialize(self):
        self.options.declare("ndim", 2, values=[2], types=int)
        self.options.declare("alpha", 5., types=float)
        self.options.declare("name", "MultiDimJump", types=str)

        self.t = None

    def _setup(self):
        self.xlimits[:, 0] = -2.5
        self.xlimits[:, 1] = 2.5

        self.t = np.ones(self.options["ndim"])# np.random.normal(size=self.options["ndim"])
        
        self.t = self.t/np.linalg.norm(self.t)
        self.alpha = self.options["alpha"]

    def _evaluate(self, x, kx):
        """
        Arguments
        ---------
        x : ndarray[ne, nx]
            Evaluation points.
        kx : int or None
            Index of derivative (0-based) to return values with respect to.
            None means return function value rather than derivative.

        Returns
        -------
        ndarray[ne, 1]
            Functions values if kx=None or derivative values if kx is an int.
        """
        ne, nx = x.shape
        y = np.zeros((ne, 1), complex)

        for i in range(ne):
            work = np.dot(x[i,:], self.t)
            if kx is None:
                y[i,0] = np.arctan(self.alpha*work)
            else:
                work2 = (1./(1.+work*work*self.alpha*self.alpha))*self.alpha
                y[i,0] = work2*self.t[kx]

        return y

