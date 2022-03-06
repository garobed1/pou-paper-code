import unittest
import numpy as np
import sys
sys.path.insert(1,"../")

from surrogate.example_problems import FuhgP8, Heaviside, Quad2D, QuadHadamard, MultiDimJump

class ProblemDiffTest(unittest.TestCase):
    
    def test_MultiDimJumpGradient(self):
        h = 1e-5
        dim = 5
        alpha = 9.129
        trueFunc = MultiDimJump(ndim=dim, alpha=alpha)
        xi = np.random.rand(dim)
        spread = trueFunc.xlimits[:,1] - trueFunc.xlimits[:,0]
        offset = -trueFunc.xlimits[:,0]
        xg = np.random.rand(dim)*spread - offset

        fgm = trueFunc(np.array([xg-h*xi]))
        fgp = trueFunc(np.array([xg+h*xi]))
        ga = np.zeros([1,dim])
        for i in range(dim):
            ga[0,i] = trueFunc(np.array([xg]), i)

        finitediff = (1./(2*h))*(fgp-fgm)
        analytic = np.dot(ga, xi)
        err = abs(analytic - finitediff)
        self.assertTrue(err < 1.e-8)


    def test_FuhgP8Gradient(self):
        h = 1e-5
        dim = 2
        trueFunc = FuhgP8(ndim=dim)
        xi = np.random.rand(dim)
        spread = trueFunc.xlimits[:,1] - trueFunc.xlimits[:,0]
        offset = -trueFunc.xlimits[:,0]
        xg = np.random.rand(dim)*spread - offset

        fgm = trueFunc(np.array([xg-h*xi]))
        fgp = trueFunc(np.array([xg+h*xi]))
        ga = np.zeros([1,dim])
        for i in range(dim):
            ga[0,i] = trueFunc(np.array([xg]), i)

        finitediff = (1./(2*h))*(fgp-fgm)
        analytic = np.dot(ga, xi)
        err = abs(analytic - finitediff)
        self.assertTrue(err < 1.e-8)


    def test_QuadHadamardGradient(self):
        h = 1e-5
        dim = 8
        rate = 1.4
        trueFunc = QuadHadamard(ndim=dim, eigenrate=rate)
        xi = np.random.rand(dim)
        spread = trueFunc.xlimits[:,1] - trueFunc.xlimits[:,0]
        offset = -trueFunc.xlimits[:,0]
        xg = np.random.rand(dim)*spread - offset

        fgm = trueFunc(np.array([xg-h*xi]))
        fgp = trueFunc(np.array([xg+h*xi]))
        ga = np.zeros([1,dim])
        for i in range(dim):
            ga[0,i] = trueFunc(np.array([xg]), i)

        finitediff = (1./(2*h))*(fgp-fgm)
        analytic = np.dot(ga, xi)
        err = abs(analytic - finitediff)

        self.assertTrue(err < 1.e-8)

if __name__ == '__main__':
    unittest.main()