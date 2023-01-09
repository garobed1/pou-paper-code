import unittest
import numpy as np
import sys

from utils.sutils import quadraticSolve, quadraticSolveHOnly, symMatfromVec, maxEigenEstimate, boxIntersect
from utils.error import stat_comp, meane
from smt.problems import RobotArm, Rosenbrock
from smt.surrogate_models import KRG
from smt.sampling_methods import FullFactorial

class UtilTest(unittest.TestCase):
    
    def test_boxIntersect(self):
        dim = 2
        trueFunc = RobotArm(ndim=dim)
        xlimits = trueFunc.xlimits

        xc = np.array([0.9, np.pi])
        xdir = np.array([1, 4])
        xdir = xdir/np.linalg.norm(xdir)

        p0, p1 = boxIntersect(xc, xdir, xlimits)

        self.assertTrue(p0 - -3.238279585853798 < 1.e-14)
        self.assertTrue(p1 - 0.41231056256176596 < 1.e-14)


# dim = 2
# trueFunc = Quad2D(ndim=dim, theta=np.pi/4)
# xlimits = trueFunc.xlimits
# sampling = LHS(xlimits=xlimits)

# nt0  = 3

# t0 = np.array([[0.25, 0.75],[0.8, 0.5],[0.75, 0.1]])# sampling(nt0)[0.5, 0.5],
# f0 = trueFunc(t0)
# g0 = np.zeros([nt0,dim])
# for i in range(dim):
#     g0[:,i:i+1] = trueFunc(t0,i)

# quadraticSolveHOnly(t0[0,:], t0[1:3,:], f0[0], f0[1:3], g0[0,:], g0[1:3,:])





# x = np.array([1, 2, 3, 4])
# xn = np.zeros([6, 4])
# for i in range(6):
#     xn[i,:] = 0.5*i

# f = 10
# fn = np.zeros(6)
# for i in range(6):
#     fn[i] = i

# g = x
# gn = xn
# for i in range(6):
#     gn[i,:] = i

# quadraticSolve(x, xn, f, fn, g, gn)

if __name__ == '__main__':
    unittest.main()