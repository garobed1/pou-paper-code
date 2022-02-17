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