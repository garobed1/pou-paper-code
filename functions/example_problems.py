"""
Implements the heaviside function
"""
from cmath import cos, sin
import numpy as np

from smt.problems.problem import Problem


class Sine1D(Problem):
    def _initialize(self):
        self.options.declare("ndim", 1, values=[2], types=int)
        self.options.declare("name", "Sine1D", types=str)

    def _setup(self):
        self.xlimits[:, 0] = 0.0
        self.xlimits[:, 1] = 2.*np.pi

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

        if kx is None:
            y[:,0] = np.sin(x[:,0])
        else:
            y[:,0] = np.cos(x[:,0])

        return y

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



# From Fuhg (2021), Problem 3
class FuhgP3(Problem):
    def _initialize(self):
        self.options.declare("ndim", 1, values=[2], types=int)
        self.options.declare("name", "FuhgP3", types=str)

    def _setup(self):
        self.xlimits[:, 0] = -12.
        self.xlimits[:, 1] = 12.

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
                if(X[0] < 0.):
                    y[i,0] = -np.sinh(0.3*X[0])
                elif(X[0] >= 0 and X[0] < 5.): 
                    y[i,0] = -5.*X[0]
                elif(X[0] >= 5. and X[0] < 8.): 
                    y[i,0] = (25./3.)*X[0] - (200./3.)
                else:
                    y[i,0] = np.sinh(0.9*(X[0] - 8.))

            elif(kx == 0):
                if(X[0] < 0.):
                    y[i,0] = -0.3*np.cosh(0.3*X[0])
                elif(X[0] >= 0 and X[0] < 5.): 
                    y[i,0] = -5.
                elif(X[0] >= 5. and X[0] < 8.): 
                    y[i,0] = (25./3.)
                else:
                    y[i,0] = 0.9*np.cosh(0.9*(X[0] - 8.))


        return y



# From Fuhg (2021)
class FuhgSingleHump(Problem):
    def _initialize(self):
        self.options.declare("ndim", 1, values=[2], types=int)
        self.options.declare("name", "FuhgSingleHump", types=str)

    def _setup(self):
        self.xlimits[:, 0] = -1.5
        self.xlimits[:, 1] = 5.

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
                y[i,0] = 3*X[0]
                y[i,0] -= 0.05/((X[0] - 4.75)**2 + 0.04)
                y[i,0] -= 0.07/((X[0] - 4.45)**2 + 0.005) - 6.

            elif(kx == 0):
                y[i,0] = 3.
                y[i,0] += 2*0.05*(X[0] - 4.75)/((X[0] - 4.75)**2 + 0.04)**2
                y[i,0] += 2*0.07*(X[0] - 4.45)/((X[0] - 4.45)**2 + 0.005)**2

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

        if(self.dim == 16):
            self.mean = 112.68464900969741
            self.stdev = 50.64728616481403

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

        self.dim = self.options["ndim"]
        self.t = np.ones(self.options["ndim"])# np.random.normal(size=self.options["ndim"])
        
        if(self.dim == 1):
            self.mean = 0.
            self.stdev = 1.3950154957734922

        if(self.dim == 2):
            self.mean = 0.
            self.stdev = 1.3643327395075313

        if(self.dim == 6):
            self.mean = 0.
            self.stdev = 1.362917285776268

        if(self.dim == 12):
            self.mean = 0.
            self.stdev = 1.3613630211906838

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



# f(x) = arctan(alpha*(x dot t)) times e^(||x||^2)
class MultiDimJumpTaper(Problem):
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
                y[i,0] = np.arctan(self.alpha*work)*np.exp(-np.linalg.norm(x[i,:])**2)
            else:
                work2 = (1./(1.+work*work*self.alpha*self.alpha))*self.alpha
                y[i,0] = work2*self.t[kx]*np.exp(-np.linalg.norm(x[i,:])**2) + \
                        np.arctan(self.alpha*work)*np.exp(-np.linalg.norm(x[i,:])**2)*-2*x[i,kx]

        return y


# f(x) = arctan(alpha*(x dot t))
class MultiDimJumpTwist(Problem):
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
            work1 = x[i, :] - np.array([np.sin(x[i,0]), np.sin(x[i,1])])
            work = np.dot(work1, self.t)
            if kx is None:
                y[i,0] = np.arctan(self.alpha*work)
            else:
                dwork1 = 1-np.cos(x[i,kx])
                work2 = (1./(1.+work*work*self.alpha*self.alpha))*self.alpha
                y[i,0] = work2*self.t[kx]*dwork1

        return y



# From Fuhg (2021), Problem 9
class FuhgP9(Problem):
    def _initialize(self):
        self.options.declare("ndim", 2, values=[2], types=int)
        self.options.declare("name", "FuhgP9", types=str)

    def _setup(self):
        self.xlimits[0, 0] = -2.0
        self.xlimits[0, 1] = 2.0
        self.xlimits[1, 0] = -1.0
        self.xlimits[1, 1] = 1.0

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
                y[i,0] = (4. - 2.1*X[0]*X[0] + X[0]*X[0]*X[0]*X[0]/3.)*X[0]*X[0]
                y[i,0] += X[0]*X[1] + (-4. + 4.*X[1]*X[1])*X[1]*X[1]


            elif(kx == 0):
                y[i,0] = (8.*X[0] - 4*2.1*X[0]*X[0]*X[0] + 6*X[0]*X[0]*X[0]*X[0]*X[0]/3.)
                y[i,0] += X[1] 

            elif(kx == 1):
                y[i,0] = X[0] + (-8.*X[1] + 16.*X[1]*X[1]*X[1])

        return y


# From Fuhg (2021), Problem 10
class FuhgP10(Problem):
    def _initialize(self):
        self.options.declare("ndim", 2, values=[2], types=int)
        self.options.declare("name", "FuhgP10", types=str)

    def _setup(self):
        self.xlimits[:, 0] = -6.0
        self.xlimits[:, 1] = 2.0

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
                if(X[0] < -2.5 and X[1] < -2.5):
                    for j in range(nx):
                        y[i,0] += 0.2*X[j]*X[j]*X[j] - 10.*np.cos(2.*np.pi*X[j])
                else:
                    for j in range(nx):
                        y[i,0] += 0.2*X[j]*X[j]*X[j] + 3.*np.abs(X[j]) - 30.*np.sin(np.pi*np.abs(X[j]))

            else:
                j = kx
                if(X[0] < -2.5 and X[1] < -2.5):
                    y[i,0] += 0.6*X[j]*X[j] + 20.*np.pi*np.sin(2.*np.pi*X[j])
                else:
                    y[i,0] += 0.6*X[j]*X[j] + np.sign(X[j])*3. - np.sign(X[j])*30.*np.pi*np.cos(np.pi*np.abs(X[j]))


        return y


# Peaks function from Matlab
class Peaks2D(Problem):
    def _initialize(self):
        self.options.declare("ndim", 2, values=[2], types=int)
        self.options.declare("name", "Peaks2D", types=str)

    def _setup(self):
        self.xlimits[0, 0] = -3.0
        self.xlimits[0, 1] = 3.0
        self.xlimits[1, 0] = -3.0
        self.xlimits[1, 1] = 3.0

        self.mean = 0.362770232179922
        self.stdev = 1.9065492039328813

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
            C1 = 3*((1-X[0])**2)
            C2 = -10*(X[0]/5. - X[0]**3 - X[1]**5)
            C3 = -(1./3.)
            if kx is None:
                y[i,0] =  C1*np.exp(-X[0]**2 - (X[1]+1)**2)
                y[i,0] += C2*np.exp(-X[0]**2 - X[1]**2)
                y[i,0] += C3*np.exp(-(X[0]+1)**2 - X[1]**2)

            elif(kx == 0):
                dC1 = -6*(1-X[0])
                dC2 = (-2. + 30.*X[0]**2)#-10*(x[0]/5. - x[0]**3 - x[1]**5)
                dC3 = 0.

                y[i,0] =  (dC1 + C1*(-2.*X[0]))*np.exp(-X[0]**2 - (X[1]+1)**2)
                y[i,0] += (dC2 + C2*(-2.*X[0]))*np.exp(-X[0]**2 - X[1]**2)
                y[i,0] += (dC3 + C3*(-2.*(X[0]+1)))*np.exp(-(X[0]+1)**2 - X[1]**2)

            elif(kx == 1):
                dC1 = 0.
                dC2 = 50.*X[1]**4 #-10*(x[0]/5. - x[0]**3 - x[1]**5)
                dC3 = 0.

                y[i,0] =  (dC1 + C1*(-2.*(X[1]+1)))*np.exp(-X[0]**2 - (X[1]+1)**2)
                y[i,0] += (dC2 + C2*(-2.*X[1]))*np.exp(-X[0]**2 - X[1]**2)
                y[i,0] += (dC3 + C3*(-2.*X[1]))*np.exp(-(X[0]+1)**2 - X[1]**2)

        return y


# Emulated shock problem function
class FakeShock(Problem):
    def _initialize(self):
        self.options.declare("ndim", 2, values=[2], types=int)
        self.options.declare("name", "FakeShock", types=str)

    def _setup(self):
        self.xlimits[0,:] = [23., 27.]
        self.xlimits[1,:] = [0.36, 0.51]

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
        sc = 0.0004
        bs = 0.0001
        root = 24. + 83./500.

        for i in range(ne):
            X = x[i,:]
            work = X[0] - 25.5
            work2 = X[1] - self.xlimits[1,0]
            if kx is None:
                if(X[0] > root):
                    y[i,0] = 0.2*(work**4) + work2
                else:
                    y[i,0] = -0.1*work + 0.5 + work2

            elif(kx == 0):
                if(X[0] > root):
                    y[i,0] = 0.8*(work**3)
                else:
                    y[i,0] = -0.1
            elif(kx == 1):
                y[i,0] = 1.0
        
        y *= sc

        if kx is None:
            y += bs

        return y



# Ishigami function 
class Ishigami(Problem):
    def _initialize(self):
        self.options.declare("ndim", 3, values=[2], types=int)
        self.options.declare("name", "Ishigami", types=str)

    def _setup(self):
        self.xlimits[:, 0] = -np.pi
        self.xlimits[:, 1] = np.pi

        self.a = 7.
        self.b = 0.1

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
                y[i,0] = (1. + self.b*(X[2]**4))*np.sin(X[0]) + self.a*(np.sin(X[1])**2)

            elif(kx == 0):
                y[i,0] = (1. + self.b*(X[2]**4))*np.cos(X[0])
            elif(kx == 1):
                y[i,0] = self.a*(np.sin(2*X[1]))
            elif(kx == 2):
                y[i,0] = 4*self.b*(X[2]**3)*np.sin(X[0])
                
        return y



# Clark 2020, choose to use nonlinear design variable or not
class ToyLinearScale(Problem):
    def _initialize(self):
        self.options.declare("ndim", 1, values=[2], types=int)
        self.options.declare("name", "Sine1D", types=str)
        self.options.declare("use_design", False, types=bool)

    def _setup(self):
        self.dim_u = self.options["ndim"]
        if(self.options["use_design"]):
            self.dim_u -= 1
            self.xlimits[self.dim_u, 0] = 5.0
            self.xlimits[self.dim_u, 1] = 15.0

        sigtotal = 0.01
        fac = 0
        for i in range(self.dim_u):
            fac += 1./(i+1.)
        A = sigtotal/fac
        
        for i in range(self.dim_u):
            sigs = np.sqrt(A/(i+1.))
            self.xlimits[i, 0] = -6.*sigs
            self.xlimits[i, 1] = 6.*sigs


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

        d1 = 10.
        if(self.options["use_design"]):
            d1 = x[:,-1]

        d12 = np.multiply(d1, d1)

        if kx is None:
            y[:,0] = 1. - 4.02/d1 - 25.25/d12 - (d1/15.)*np.sum(x[:,0:self.dim_u], axis=1)
        else:
            if(kx == self.dim_u and self.options["use_design"]):
                y[:,0] = 4.02/d12 + 50.5/np.multiply(d12, d1) - (1./15.)*np.sum(x[:,0:self.dim_u], axis=1)
            else:
                y[:,0] = -(d1/15.)

        return y


class Ellipse(Problem):
    def _initialize(self):
        self.options.declare("ndim", 2, values=[2], types=int)
        self.options.declare("foci", default=[2., 1.], types=list)
        self.options.declare("name", "Ellipse", types=str)

    def _setup(self):
        self.xlimits[:, 0] = -10.0
        self.xlimits[:, 1] = 10.0

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
        f = self.options["foci"]

        y = np.zeros((ne, 1), complex)
        if kx is None:
            y[:,0] = (x[:,0] - f[0])**2 + (x[:,1] - f[1])**2
            y[:,0] += (x[:,0] + f[0])**2 + (x[:,1] + f[1])**2
        else:
            y[:,0] = 2*(x[:, kx] - f[kx])#/np.sqrt((x[:,0] - f[0])**2 + (x[:,1] - f[1])**2)
            y[:,0] += 2*(x[:, kx] + f[kx])#/np.sqrt((x[:,0] + f[0])**2 + (x[:,1] + f[1])**2)

        return y

"""
2D (1D design x_d, 1D uncertain x_u) robust mean optimization benchmark given a beta 
distribution of x_u with alpha = 3, beta = 1

f(x_u,x_d) = (1/3)(-(3/8)D(x_d) + (1/20)x_u^3 + (1/64)D(x_d)x_u^5)

x_u \in [0,1]
x_d \in [0,10]

The mean with respect to x_u reduces to D(x_d) = x_d\sin(x_d) + 0.5333, which we want to optimize.

"""
class BetaRobust1D(Problem):
    def _initialize(self):
        self.options.declare("ndim", 2, values=[2], types=int)
        self.options.declare("name", "BetaRobust1D", types=str)

    def _setup(self):
        self.xlimits[0, 0] = 0.0
        self.xlimits[0, 1] = 1.0
        self.xlimits[1, 0] = 0.0
        self.xlimits[1, 1] = 10.0

        self.a = -3./8.
        self.b = 1./20.
        self.c = 1./64.

        self.alpha = 3.
        self.beta = 1.
        self.sh = self.alpha/self.beta # beta shape param ratio

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
        D = x[:,1]*np.sin(x[:,1])
        if kx is None:
            y[:,0] = self.a*D + self.b*x[:,0]*x[:,0]*x[:,0]
            y[:,0] += self.c*D*x[:,0]*x[:,0]*x[:,0]*x[:,0]*x[:,0]
            y[:,0] *= self.sh
        elif kx == 0:
            y[:,0] = 3*self.b*x[:,0]*x[:,0]
            y[:,0] += 5*self.c*D*x[:,0]*x[:,0]*x[:,0]*x[:,0]
            y[:,0] *= self.sh
        elif kx == 1:
            dD = np.sin(x[:,1]) + x[:,1]*np.cos(x[:,1])
            y[:,0] = self.a*dD
            y[:,0] += self.c*dD*x[:,0]*x[:,0]*x[:,0]*x[:,0]*x[:,0]
            y[:,0] *= self.sh

        return y

# from matplotlib import pyplot as plt

# prob = Ellipse(foci = [1, 3])
# xlimits = prob.xlimits

# ndir = 100

# x = np.linspace(xlimits[0][0], xlimits[0][1], ndir)
# y = np.linspace(xlimits[1][0], xlimits[1][1], ndir)

# X, Y = np.meshgrid(x, y)
# Z = np.zeros([ndir, ndir])

# for i in range(ndir):
#     for j in range(ndir):
#         xi = np.zeros([1,2])
#         xi[0,0] = x[i]
#         xi[0,1] = y[j]
#         Z[i,j] = prob(xi)

# #grad 
# h = 1e-5
# xg = np.zeros([1,2])
# xgs = np.zeros([1,2])
# xg[0] = [5, 5]

# fg = prob(xg)
# fgs = np.zeros([1,2])
# ga = np.zeros([1,2])
# for i in range(2):
#     ga[0,i] = prob(xg, i)
#     xgs[0] = xg[0]
#     xgs[0,i] += h
#     fgs[0,i] = prob(xgs)

# gd = (1./h)*(fgs-fg)

# plt.contour(X, Y, Z)
# plt.savefig("ellipse.png")