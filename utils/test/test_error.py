import unittest
import numpy as np
import sys

from utils.sutils import quadraticSolve, quadraticSolveHOnly, symMatfromVec, maxEigenEstimate, boxIntersect
from utils.error import stat_comp, meane
from smt.problems import RobotArm, Rosenbrock
from functions.example_problems import BetaRobust1D
from smt.surrogate_models import KRG
from smt.sampling_methods import FullFactorial

class StatsTest(unittest.TestCase):

    def test_stat_comp(self):
        dim = 2
        trueFunc = Rosenbrock(ndim=dim)
        xlimits = trueFunc.xlimits

        # first check that meane still works as expected with a surrogate
        sampling = FullFactorial(xlimits=xlimits)
        xtrainK = sampling(16*16)
        ftrainK = trueFunc(xtrainK)

        surr = KRG()
        surr.set_training_values(xtrainK, ftrainK)
        surr.train()

        # different N?
        merr, serr = meane(surr, trueFunc, N=5000, xdata=None, fdata=None, return_values=False)

        self.assertTrue(merr < 1.e-2)
        self.assertTrue(serr < 1.e-2)

    # test if mean w.r.t. x_u comes to x_d*sin(x_d) + 0.53333333 NOTE: Change this as I play with the actual function
    def test_BetaRobust1D(self):
        
        ndir = 100
        trueFunc = BetaRobust1D(ndim=2)
        xlimits = trueFunc.xlimits

        x = np.linspace(xlimits[1,0], xlimits[1,1], ndir)
        y = np.zeros(ndir)

        for i in range(ndir):
            pdfs = [x[i], ['beta', 3., 1.]] # fix x_d at 0, x_u w/ beta dist
            out1, out2 = stat_comp(model=None, prob=trueFunc, N=5000, pdfs=pdfs)
            y[i] = out1
        
        import matplotlib.pyplot as plt
        plt.plot(x, y)
        plt.savefig('betarobtest.png')
        plt.clf()
        import pdb; pdb.set_trace()

        #TODO: need to finish this part up

        serr = 1
        self.assertTrue(serr < 1.e-2)

if __name__ == '__main__':
    unittest.main()