"""
Implements the impinging shock problem as an smt Problem
"""
from cmath import cos, sin
import numpy as np
from mpi4py import MPI
import sys

import openmdao.api as om
from smt.problems.problem import Problem
from mphys_comp.impinge_analysis import Top
from utils.sutils import divide_cases

import mphys_comp.impinge_setup as default_impinge_setup

# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# size = comm.Get_size()

class ImpingingShock(Problem):
    def _initialize(self):
        self.options.declare("ndim", 2, values=[2], types=int)
        self.options.declare("name", "ImpingingShock", types=str)
        
        
        self.options.declare("problem_settings", default=default_impinge_setup)
        self.options.declare("inputs", ["shock_angle", "rsak"], types=list)
        self.options.declare("input_bounds", np.zeros([2,2]), types=np.ndarray)
        self.options.declare("output", ["test.aero_post.cd_def"], types=list) #surrogate only returns the first element but we'll store the others

        self.options.declare("comm", MPI.COMM_WORLD, types=MPI.Comm)



    def _setup(self):
        
        self.comm = self.options["comm"]
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        assert(self.options["ndim"] == len(self.options["inputs"]))

        # list of inputs we would need to finite difference
        self.fdlist = ['M0', 'P0', 'T0', 'shock_angle']
        # self.fdlist = [] # try this? and lower tolerance on solver

        # list of inputs we can get exact derivatives for
        #self.adlist = self.options["inputs"]
        self.adind = []
        self.fdind = []
        for i in range(len(self.options["inputs"])):
            key = self.options["inputs"][i]
            if(key in self.fdlist):
                self.fdind.append(i)
            else:
                self.adind.append(i)


        # ensure we can get SA derivatives
        actual_settings = self.options["problem_settings"]

        saconsts = ['rsak','rsacb1','rsacb2','rsacb3','rsacv1','rsacw2','rsacw3','rsact1','rsact2','rsact3','rsact4','rsacrot']
        salist = []
        for key in self.options["inputs"]:
            if(key in saconsts):
                salist = salist + [key]
        actual_settings.aeroOptions["SAGrads"] = salist

        self.xlimits[:, 0] = self.options["input_bounds"][:,0]
        self.xlimits[:, 1] = self.options["input_bounds"][:,1]

        self.prob = om.Problem(comm=MPI.COMM_SELF)
        self.prob.model = Top(problem_settings=actual_settings)

        # set up model inputs
        dim = self.options["ndim"]
        for i in range(dim):
            self.prob.model.add_design_var(self.options["inputs"][i], \
                                    lower=self.xlimits[i,0], upper=self.xlimits[i,1])

        for i in range(len(self.options["output"])):
            self.prob.model.add_objective(self.options["output"][i])

        self.prob.setup(mode='rev')

        # keep the current input state to avoid recomputing the gradient on subsequent calls
        self.xcur = np.zeros([1])
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
        h = 1e-6
        #import pdb; pdb.set_trace()
        if(not np.array_equal(x, self.xcur)): # Don't recompute if we already have the answer for the given inputs
            self.xcur = x
            self.fcur = np.zeros((ne, 1), complex)
            self.gcur = np.zeros((ne, nx), complex)
            cases = divide_cases(ne, self.size)
            #for i in range(ne):
            for i in cases[self.rank]:
                for j in range(nx):
                    self.prob.set_val(self.options["inputs"][j], x[i, j])
                self.prob.run_model()
                if not self.prob.driver.fail:
                    self.fcur[i] = self.prob.get_val(self.options["output"][0])

                    #analytic
                    work = [self.options["inputs"][k] for k in self.adind]
                    adgrads = self.prob.compute_totals(of=self.options["output"][0], wrt=work, return_format="array")
                    self.gcur[i][self.adind] = adgrads#[:,0]
                else:
                    self.fcur[i] = np.nan
                    self.gcur[i][self.adind] = np.nan

                
                
                #finite diff
                for key in self.fdind:
                    self.prob.set_val(self.options["inputs"][key], x[i, key] + h)
                    self.prob.run_model()
                    self.gcur[i][key] = (self.prob.get_val(self.options["output"][0]) - self.fcur[i])/h
                    self.prob.set_val(self.options["inputs"][key], x[i, key])


            self.fcur = self.comm.allreduce(self.fcur)
            self.gcur = self.comm.allreduce(self.gcur)

        for i in range(ne):
            if kx is None:
                y[i,0] = self.fcur[i]
            else:
                y[i,0] = self.gcur[i,kx]

        return y

