import numpy as np

from optimization.optimizers import optimize
from optimization.robust_objective import RobustQuantity

from smt.utils.options_dictionary import OptionsDictionary


try:
    import pyoptsparse
    Optimization = pyoptsparse.Optimization
except ImportError:
    Optimization = None
    pyoptsparse = None

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
    def __init__(self, **kwargs):
        """
        Initialize attributes
        """
        self.name = ''
        self.pathname = None
        self.comm = None

        self.optimizer = None

        self.obj = None
        self.dobj = None
        self.cons = [] #empty list
        self.dcons = []

        self.options = OptionsDictionary(parent_name=type(self).__name__)


    def _declare_options(self):
        self.options.declare('optimizer', default=optimize,
                             desc='function handle ')    


    def set_objective(self, func, grad=None):
        """
        Set objective function/function handle

        Parameters
        ----------
        func : function handle OR RobustQuantity
            function handle or robust quantity object that gives handles
        
        grad : function handle for gradient
        """

        if isinstance(func, RobustQuantity):
            self.obj = func.func
            self.dobj = func.grad
        else:
            self.obj = func
            self.dobj = grad
        
    def add_constraint(self, func, lower=None, upper=None, equals=None, grad=None):

        """
        Set objective function/function handle

        Parameters
        ----------
        func : function handle OR RobustQuantity
            function handle or robust quantity object that gives handles
        
        grad : function handle 
            function handle for gradient
        lower : float
            lower bound for inequality 
        upper : float
            upper bound for inequality 
        equals : float
            equality value        
        """

        # From OM
        # A constraint cannot be an equality and inequality constraint
        if equals is not None and (lower is not None or upper is not None):
            msg = "{}: Constraint '{}' cannot be both equality and inequality."
            raise ValueError(msg.format(self.msginfo, name))
    
    def set_optimizer(self, opt=optimize):
        """
        Set the optimization object that solves the subproblem.
        """

        self.optimizer = opt
        
        # scipy optimizer
        if callable(opt):
            self.options["optimizer"] = 'scipy'
        # pyOptSparse optimizer
        elif isinstance(opt, Optimization):
            self.options["optimizer"] = 'pyoptsparse'
            if opt is None:
                raise RuntimeError("pyOptSparse not installed!")

        else:
            raise RuntimeError("Invalid optimizer")

        # actual call that prepares things
        self._setup_optimization()
    
    def solve_subproblem(self):
        """
        Find s_k that may update z_k to z_k + s_k by solving the subproblem

        Must be implemented in derived classes
        """
        pass

    def eval_truth(self, zk)
        """
        At the end of optimization, evaluate the "truth" model for comparison
        or approval, and use the result to take action
        """

        pass


    def _setup_optimization(self):
        """
        Set up arguments, settings, objectives, constraints, etc. for either 
        the scipy optimization call or the pyOptSparse object, similar to how 
        the OpenMDAO drivers do it
        
        """

        # for scipy, set up a dictionary of arguments
        if self.options["optimizer"] == 'scipy':
            pass

        # for pyoptsparse, call setup functions
        if self.options["optimizer"] == 'pyoptsparse':
            pass