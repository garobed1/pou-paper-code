import unittest
import numpy as np
import sys

from utils.sutils import quadraticSolve, quadraticSolveHOnly, symMatfromVec, maxEigenEstimate, boxIntersect
from utils.error import stat_comp, meane
from smt.problems import RobotArm, Rosenbrock
from smt.surrogate_models import KRG
from smt.sampling_methods import FullFactorial

class ErrorTest(unittest.TestCase):

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

    # test if mean w.r.t. x_u comes to x_d*sin(x_d) + 0.53333333
    def test_BetaRobust1D(self):
        serr = 1
        self.assertTrue(serr < 1.e-2)