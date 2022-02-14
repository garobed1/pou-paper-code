"""
Implements the n-dimensional ellipse function in the SMT context
"""
import numpy as np

from smt.problems.problem import Problem

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