"""
Implements the impinging shock problem as an smt Problem
"""
from cmath import cos, sin
import numpy as np
from mpi4py import MPI
import sys
sys.path.insert(1,"../mphys/")
sys.path.insert(1,"../surrogate")

import openmdao.api as om
from smt.problems.problem import Problem
from impinge_analysis import Top
import impinge_setup
from utils import divide_cases

# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# size = comm.Get_size()

class ImpingingShock(Problem):
    def _initialize(self):
        self.options.declare("ndim", 2, values=[2], types=int)
        self.options.declare("name", "ImpingingShock", types=str)
        
        
        self.options.declare("inputs", ["M0", "rsak"], types=list)
        self.options.declare("input_bounds", np.zeros([2,2]), types=np.ndarray)
        self.options.declare("output", "test.aero_post.cd_def", types=str)

        self.options.declare("comm", MPI.COMM_WORLD, types=MPI.Comm)



    def _setup(self):
        
        self.comm = self.options["comm"]
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        assert(self.options["ndim"] == len(self.options["inputs"]))

        # list of inputs we would need to finite difference
        self.fdlist = ['M0', 'P0', 'T0']

        # list of inputs we can get exact derivatives for
        self.adlist = self.options["inputs"]
        remove = []
        for i in range(len(self.adlist)):
            key = self.adlist[i]
            if((key == self.fdlist).any()):
                remove.append(i)
        for key in remove:
            self.adlist.pop(key)

        # ensure we can get SA derivatives
        saconsts = ['rsak','rsacb1','rsacb2','rsacb3','rsacv1','rsacw2','rsacw3','rsact1','rsact2','rsact3','rsact4','rsacrot']
        salist = []
        for key in self.options["inputs"]:
            if(key in saconsts):
                salist = salist + key
        impinge_setup.aeroOptions["SAGrads"] = salist

        self.xlimits[:, 0] = self.options["input_bounds"][:,0]
        self.xlimits[:, 1] = self.options["input_bounds"][:,1]

        self.prob = om.Problem()
        self.prob.model = Top()

        # set up model inputs
        dim = self.options["ndim"]
        for i in range(dim):
            self.prob.model.add_design_var(self.options["inputs"][i], \
                                    lower=self.xlimits[i,0], upper=self.xlimits[i,1])

        self.prob.model.add_objective("output")

        self.prob.setup(mode='rev')

        # keep the current input state to avoid recomputing the gradient on subsequent calls
        self.xcur = None
        self.fcur = None
        self.gcur = None

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


        if(x != self.xcur): # Don't recompute if we already have the answer for the given inputs
            self.xcur = x
            self.fcur = np.zeros((ne, 1), complex)
            self.gcur = np.zeros((ne, nx), complex)
            cases = divide_cases(ne, self.size)
            #for i in range(ne):
            for i in cases:
                for j in range(nx):
                    self.prob.set_val(self.options["inputs"][j], x[i, j])
                self.prob.run_model()
                self.fcur[i] = self.prob.get_val(self.options["output"])
                self.gcur[i] = self.prob.compute_totals(return_format="array")

            self.comm.allreduce(self.fcur)
            self.comm.allreduce(self.gcur)

        for i in range(ne):
            if kx is None:
                y[i,0] = self.fcur[i]
            else:
                y[i,0] = self.gcur[i,kx]

        return y

