import numpy as np
import copy
from collections import defaultdict
from utils.error import stat_comp, _gen_var_lists
from smt.sampling_methods import LHS

from smt.utils.options_dictionary import OptionsDictionary

"""
Provides an interface to a consistent set of sampling points for a function with
design inputs and uncertain inputs, 
"""
class RobustSampler():
    def __init__(self, x_d_init, N, **kwargs):
        
        self.has_points = False #are uncertain samples generated at the current design?

        self.x_d_cur = x_d_init # current design point, same length as x_d_ind
        self.x_d_ind = None # indices of the design variables in the function call
        self.x_u_ind = None 

        self.x_d_dim = 0
        self.x_u_dim = 0


        # self.x_samples = None
        # self.f_samples = None
        # self.g_samples = None
        self.current_samples = defaultdict(dict)
        self.history = {}
        self.design_history = []
        self.nested_ref_ind = None #list of indices of current iteration that existed in previous
        self.func_computed = False #do we have function data at the samples?
        self.grad_computed = False #do we have gradient data at the samples?

        self.sampling = None #SMT sampler object

        """
        Max index to keep track of previous iterations. Increments either when 
        design is changed, or UQ is refined/changed
        """
        self.iter_max = 0

        self.comp_track = 0 #track total number of evaluations since instantiation
        self.grad_track = 0

        self.options = OptionsDictionary()
        self.options.declare(
            "xlimits",
            types=np.ndarray,
            desc="The interval of the domain in each dimension with shape dim x 2 (required)",
        )
        
        self.options.declare(
            "probability_functions",
            types=list,
            desc="gives pdfs of uncertain variables, implicit list of design variables",
        )

        self.options.declare(
            "retain_uncertain_points",
            types=bool,
            default=True,
            desc="keep the same points in the uncertain space as we traverse the design space",
        )

        self.options.declare(
            "name",
            types=str,
            default='sampler',
            desc="keep the same points in the uncertain space as we traverse the design space",
        )

        self.options.update(kwargs)

        self.N = N

        self.initialize()

    def initialize(self):
        # run sampler for the first time on creation
        pdfs = self.options["probability_functions"]
        xlimits = self.options["xlimits"]
        

        pdf_list, uncert_list, static_list, scales = _gen_var_lists(pdfs, xlimits)
        
        self.x_u_dim = len(uncert_list)
        self.x_d_dim = len(static_list)
        self.x_u_ind = uncert_list
        self.x_d_ind = static_list
        self.pdf_list = pdf_list
        self.dim = self.x_u_dim + self.x_d_dim

        self._initialize()

        self.generate_uncertain_points(self.N)    


    """
    Start of methods to override
    """
    def _initialize(self):
        # run sampler for the first time on creation
        xlimits = self.options["xlimits"]

        u_xlimits = xlimits[self.x_u_ind]
        self.sampling = LHS(xlimits=u_xlimits, criterion='maximin')

    def _declare_options(self):
        pass

    def _new_sample(self, N):
        #TODO: options for this
        u_tx = self.sampling(N)
        tx = np.zeros([N, self.dim])
        tx[:, self.x_u_ind] = u_tx
        tx[:, self.x_d_ind] = self.x_d_cur#[self.x_d_cur[i] for i in self.x_d_ind]
        return tx

    def _refine_sample(self, N):
        tx = self.current_samples['x']

        # track matching points #TODO: standardize this
        self.nested_ref_ind = range(tx.shape[0]).tolist()
        return tx

    """
    End of methods to override
    """

    def set_design(self, x_d_new):
        
        x_d_buf = x_d_new
        if np.allclose(x_d_buf, self.x_d_cur, rtol = 1e-15, atol = 1e-15):
            print("No change in design, returning")
            return
        else: 
            if self.options["retain_uncertain_points"]:
                self._internal_save_state()

            self.x_d_cur = x_d_new
            self._attribute_reset()

            # no need to generate uncertain points if this is the case
            if self.options["retain_uncertain_points"]:
                self.current_samples['x'][:, self.x_d_ind] = self.x_d_cur
                # self.x_samples[:, self.x_d_ind] = self.x_d_cur#[self.x_d_cur[i] for i in self.x_d_ind]
                self.has_points = True

    def generate_uncertain_points(self, N):
        """
        First of two functions that will increment the sampling iteration

        Generates 

        Parameters
        ----------
        N: int or float or list
            generic reference to sampling level, by default just number of points to sample

        """
        # check if we already have them NOTE: EVEN IF A DIFFERENT NUMBER IS REQUESTED FOR NOW
        if self.has_points:
            print(f"Iter {self.iter_max}: No design change requested, no points generated")
            return

        tx = self._new_sample(N)

        # archive previous dataset
        self._internal_save_state(refine=False)

        self.current_samples['x'] = tx
        self.has_points = True

    def refine_uncertain_points(self, N):
        """
        Second of two functions that will increment the sampling iteration
        Add more UQ points to the current design. Usually this is nested

        N: int or float or list
            generic reference to sampling level, by default just number of points to add to sample
        """
        # check if we already have them NOTE: EVEN IF A DIFFERENT NUMBER IS REQUESTED FOR NOW
        if self.has_points and N is None:
            print(f"Iter {self.iter_max}: No refinement requested, no points generated")
            return

        tx = self._refine_sample(N)
        
        # archive previous dataset
        self._internal_save_state(refine=True)
        
        self.current_samples['x'] = tx
        self.has_points = True

    def set_evaluated_func(self, f):

        self.current_samples['f'] = f
        self.comp_track += f.shape[0]
        self.func_computed = True

    def set_evaluated_grad(self, g):

        self.current_samples['g'] = g
        self.grad_track += g.shape[0]
        self.grad_computed = True

    # resets attributes as space is traversed
    def _attribute_reset(self):

        self.current_samples['x'] = None
        self.current_samples['f'] = None
        self.current_samples['g'] = None
        self.func_computed = False
        self.grad_computed = False
        self.has_points = False


    # saving to a dict of dicts
    def _internal_save_state(self, refine=False):
        """
        Internal version, increments sample counter since it's called just
        before updating the state

        Parameters:
        -----------
        refine: bool
            True if refining in place, false if traversing
        
        """

        #TODO: name should depend on what kind of increment 
        affix = '_trav'
        if refine:
            affix = '_ref'

        name = self.options['name'] + '_' + str(self.iter_max) + affix
        self.save_state(name)
        
        # update design history, simple list
        self.design_history.append(self.x_d_cur)

        # increment iteration counter
        self._attribute_reset()
        self.iter_max += 1


    def save_state(self, name):
        """
        Save current state to a dict of dicts

        Parameters:
        -----------

        name: None or str
            custom name for this entry
        
        """

        #TODO: option to write to file, probably

        #This default naming scheme allows built in methods to access iterations
        if not name:
            name = self.options['name'] + '_' + str(self.iter_max) + '_saved'

        self.history[name] = copy.deepcopy(self.current_samples)
        self.history[name]['func_computed'] = copy.deepcopy(self.func_computed)
        self.history[name]['grad_computed'] = copy.deepcopy(self.grad_computed)


        # This tracks full points between refinement, so we know what we don't have to recompute
        self.history[name]['nested_ref_ind'] = copy.deepcopy(self.nested_ref_ind)

        # also keep track of if func/grad computed, other stuff?

    def load_state(self, name, filename=None):
        """
        Load state from name, set attributes as current

        Parameters:
        -----------

        name: None or str
            name of entry to load, if 

        """

        pass


# class RobustQuantity():
#     """
#     Base class that defines a generic robust quantity of interest. Provides
#     the function handles that get minimized/evaluated for constraints, along 
#     with handles for their gradients
    
#     Flag as an objective or a constraint

#     Types:
#         "musigma" (default): Mean + \eta \sqrt{Variance}
#         "failprob": 

#     Types are implemented as derived classes

#     """

#     def __init__(self, **kwargs):
#         """
#         Initialize attributes
#         """
#         self.name = ''
#         self.pathname = None
#         self.comm = None


#         self.options = OptionsDictionary(parent_name=type(self).__name__)

#         self._declare_options()
#         self.initialize()

#         self.options.update(kwargs)


#     def _declare_options(self):
#         self.options.declare('opt_type', default="obj", values=["obj", "con"],
#                              desc='objective or constraint')    
#         self.options.declare('sampler', default=None, 
#                              desc='sampler object, determines the nature of the samples used to compute the quantity')    

#     def initialize(self):
#         pass

#     def _func(self, xd):
#         pass

#     def _grad(self, xd):
#         pass


# class MeanAndVar(RobustQuantity):

#     """
#     Quantity which is (1-\eta)\mu + \eta\sigma

#     """

#     def _declare_options(self):
        
#         super()._declare_options()

#         self.options.declare('eta_val', types=float, default=0.5,
#                              desc='balance parameter between mean and sigma, likely to be swept for pareto front')