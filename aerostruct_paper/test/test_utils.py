import unittest
import numpy as np
import sys
sys.path.insert(1,"../")

from surrogate.utils import quadraticSolve, quadraticSolveHOnly, symMatfromVec, maxEigenEstimate

class ProblemDiffTest(unittest.TestCase):
    
    None
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