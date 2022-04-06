"""
Implements the impinging shock problem as an smt Problem
"""
from cmath import cos, sin
import numpy as np

from smt.problems.problem import Problem
from 

class ImpingingShock(Problem):
    def _initialize(self):
        self.options.declare("ndim", 1, values=[2], types=int)
        self.options.declare("name", "Heaviside", types=str)
        
        
        self.options.declare("inputs", ["mach", "rsak"], types=list)


    def _setup(self):
        self.options["ndim"] = len(self.options["inputs"])
        
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

