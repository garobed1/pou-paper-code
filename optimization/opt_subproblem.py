import numpy as np
import copy
import collections

from optimization.optimizers import optimize
from optimization.robust_objective import RobustSampler

from smt.utils.options_dictionary import OptionsDictionary

import openmdao.api as om
from openmdao.utils.mpi import MPI


class OptSubproblem():
    """
    Base class that defines a generic optimization subproblem, i.e. 

    Quadratic Trust Region model
    Surrogate Trust model
    Low order UQ approximation

    These may be solved in a subregion of the domain (trust radius), or the whole
    thing. Either way, the results of solving this should be compared to some 
    "truth" model, and various parameters should be updated (such as trust radius,
    surrogate sampling, etc.)



    Needs:

    Optimizer with settings
    Subdomain definition
    Appropriate constraints to handle subdomain if used
    Access to SMT surrogate methods
    Access to adaptive refinement criteria, sampling

    """
    def __init__(self, prob_model=None, prob_truth=None, **kwargs):
        """
        Initialize attributes
        """
        self.name = 'subopt_object'
        self.pathname = None
        self.comm = None

        # OM problems to be treated as submodel and truth. Could be same
        # regardless, they need the same design variable inputs, outputs, etc.
        self.prob_model = prob_model
        self.prob_truth = prob_truth

        # list of names of the ins, outs, and cons of the actual problem. 
        # the methods here may add more constraints or modify the objective, for example
        self.prob_ins = None
        self.prob_outs = None
        self.prob_cons = None

        self.options = OptionsDictionary()

        self.outer_iter = 0
        self.truth_iters = 0
        self.model_iters = 0

        self.setup_completed = False

        self.result_cur = None

        self._declare_options()
        self.options.update(kwargs)

    def _declare_options(self):
        declare = self.options.declare
        

        declare(
            "max_iter", 
            default=50, 
            types=int,
            desc="Maximum number of outer iterations"
        )

        declare(
            "print", 
            default=1, 
            types=int,
            desc="Print level for outer iterations"
        )


    def set_model(self, model):
        self.prob_model = model
        self.setup_completed = False


    def set_truth(self, truth):
        self.prob_truth = truth
        self.setup_completed = False


    def setup_optimization(self):
        
        # assume that each problem has been set up appropriately
        if self.prob_model is None or self.prob_truth is None:
            print(f"{self.name}: Both the model and truth systems need to be assigned before setting up!")
            return


        model_ins = list(self.prob_model.model.get_design_vars().keys())
        truth_ins = list(self.prob_truth.model.get_design_vars().keys())

        model_outs = list(self.prob_model.model.get_objectives().keys())
        truth_outs = list(self.prob_truth.model.get_objectives().keys())

        model_cons = list(self.prob_model.model.get_constraints().keys())
        truth_cons = list(self.prob_truth.model.get_constraints().keys())

        assert(collections.Counter(model_ins) == collections.Counter(truth_ins))
        assert(collections.Counter(model_outs) == collections.Counter(truth_outs))
        assert(collections.Counter(model_cons) == collections.Counter(truth_cons))
        
        self.prob_ins = model_ins
        self.prob_outs = model_outs
        self.prob_cons = model_cons

        self.setup_completed = True
        
    def solve_full(self):
        """
        Solve the overall optimization problem by solving successive subproblems

        The manner in which this is done is determined by derived classes
        """

        pass

    def _solve_subproblem(self, zk):
        """
        Find s_k that may update z_k to z_{k+1} = z_k + s_k by solving the subproblem

        """

        prob = self.prob_model

        # zk is initial condicition
        i = 0
        if MPI:
            prob.comm.Bcast(zk, root=0)
        
        for name, meta in prob.driver._designvars.items():
            size = meta['size']
            # prob.set_val(name, zk[i:i + size])
            val = zk[name]
            if size == 1:
                val = zk[name][0]
            prob.set_val(name, val)
            i += size

        self.prob_model.run_driver()

        self.model_iters += self.prob_model.model.stat.get_fidelity()*self.prob_model.driver.iter_count

        #TODO: Record number of evaluations (should come from the eventual component)
        # 1 unless overriden by uncertain component

    def _eval_truth(self, zk):
        """
        At certain points of the optimization, evaluate the "truth" model for comparison
        or approval, and use the result to take action
        """

        prob = self.prob_truth

        i = 0
        if MPI:
            prob.comm.Bcast(zk, root=0)
        
        for name, meta in prob.driver._designvars.items():
            size = meta['size']
            # prob.set_val(name, zk[i:i + size])
            val = zk[name]
            if size == 1:
                val = zk[name][0]
            prob.set_val(name, val)
            i += size


        self.prob_truth.run_model()

        #TODO: Record number of evaluations (should come from the eventual component)
        # 1 unless overriden by uncertain component
        self.truth_iters += self.prob_truth.model.stat.get_fidelity()

    # def _array_to_model_set

"""
Fully-solve an optimization at a low fidelity, validate, refine, and fully solve
again. Ad hoc approach suggested by Jason Hicken

"""
class SequentialFullSolve(OptSubproblem):
    def __init__(self, **kwargs):


        super().__init__(**kwargs)


    def _declare_options(self):
        
        super()._declare_options()
        
        declare = self.options.declare
        

        # both must be satisfied to converge
        declare(
            "ftol", 
            default=1e-6, 
            types=float,
            desc="Maximum allowable difference between truth and model values at sub-optimizations"
        )

        declare(
            "gtol", 
            default=1e-6, 
            types=float,
            desc="Maximum allowable TRUTH gradient L2 norm at sub-optimization solutions"
        )

        declare(
            "flat_refinement", 
            default=5, 
            types=int,
            desc="Flat refinement amount to apply at each outer iter"
        )

    def solve_full(self):

        ftol = self.options['ftol']
        gtol = self.options['gtol']
        miter = self.options['max_iter']

        ferr = 1e6
        gerr = 1e6

        zk = self.prob_model.driver.get_design_var_values()
        # DICT TO ARRAY (OR NOT)

        fail = 0
        k = 0

        fetext = '-'
        getext = '-'

        # we assume that fidelity belongs to the top level system
        # calling it stat for now
        reflevel = self.prob_model.model.stat.get_fidelity()
        refjump = self.options["flat_refinement"]
        #TODO: Need constraint conditions as well

        fail = 1
        while (ferr > ftol or gerr > gtol) and (k < miter):

            if self.options["print"]:
                print("\n")
                print(f"Outer Iteration {k} ")
                print(f"-------------------")
                print(f"    OBJ ERR: {fetext}")
                # Add constraint loop as well
                print(f"    -")
                print(f"    GRD ERR: {getext}")
                print(f"    Fidelity: {reflevel}")
                
                self.prob_model.list_problem_vars()
            
                print(f"    Solving subproblem...")
            #Complete a full optimization
            self._solve_subproblem(zk)
            
            fmod = copy.deepcopy(self.prob_model.get_val(self.prob_outs[0]))



            #Eval Truth
            if self.options["print"]:
                print(f"    Validating with truth model...")

            zk = self.prob_model.driver.get_design_var_values()
            self._eval_truth(zk)

            ftru = copy.deepcopy(self.prob_truth.get_val(self.prob_outs[0]))

            # this needs to be the lagrangian gradient with constraints
            # gmod = self.prob_model.compute_totals(return_format='array')
            gtru = self.prob_truth.compute_totals(return_format='array')

            ferr = abs(fmod-ftru)

            # perhaps instead we try the condition from Kouri (2013)?
            # not really, it uses the model gradient, which is known to be
            # close to the true gradient as a result of the algorithm assumptions

            gerr = np.linalg.norm(gtru)

            fetext = str(ferr)
            getext = str(gerr)
            # ferr = 
            # gerr = c

            #If f or g metrics are not met, 
            if gerr < gtol:
                fail = 0
                break

            if self.options["print"]:
                print(f"    Refining model by: {refjump}")
            self.prob_model.model.stat.refine_model(refjump)
            reflevel += refjump

            k += 1            
            self.outer_iter = k

        if fail:
            succ = f'unsuccessfully, true gradient norm: {getext}'
        else:
            succ = 'successfully!'

        zk = self.prob_model.driver.get_design_var_values()
        self.result_cur = zk

        print("\n")
        print(f"Optimization terminated {succ}")
        print(f"-------------------")
        print(f"    Outer Iterations: {self.outer_iter}")
        # Add constraint loop as well
        print(f"    -")
        print(f"    Final design vars: {zk}")
        print(f"    Final objective: {ftru}")
        print(f"    Final gradient norm: {getext}")
        print(f"    Final model error: {fetext}")
        print(f"    Final model level: {reflevel}")

        print(f"    Total model samples: {self.model_iters}")
        print(f"    Total truth samples: {self.truth_iters}")
        print(f"    Total samples: {self.model_iters + self.truth_iters}")
            