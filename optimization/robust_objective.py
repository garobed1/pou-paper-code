import numpy as np
import copy
from collections import defaultdict
from utils.error import stat_comp, _gen_var_lists
from scipy.special import legendre, hermite, jacobi, roots_legendre, roots_hermite, roots_jacobi
from smt.sampling_methods import LHS

from smt.utils.options_dictionary import OptionsDictionary


# this one better suited for surrogate model


_poly_root_types = {
    "uniform": roots_legendre,
    "norm": roots_hermite,
    "beta": roots_jacobi
}

"""
Provides an interface to a consistent set of sampling points for a function with
design inputs and uncertain inputs, using LHS
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
        self.stop_generating = True

        self._attribute_reset()

        self.sampling = None #SMT sampler object

        """
        Max index to keep track of previous iterations. Increments either when 
        design is changed, or UQ is refined/changed
        """
        self.iter_max = -1

        self.comp_track = 0 #track total number of evaluations since instantiation
        self.grad_track = 0

        self.options = OptionsDictionary()
        self.options.declare(
            "xlimits",
            types=np.ndarray,
            desc="The interval of the domain in each dimension with shape dim x 2 (required)",
        )

        self.options.declare(
            "name",
            default='',
            types=str,
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
        

        pdf_list, uncert_list, static_list, scales, pdf_name_list = _gen_var_lists(pdfs, xlimits)
        
        self.x_u_dim = len(uncert_list)
        self.x_d_dim = len(static_list)
        self.x_u_ind = uncert_list
        self.x_d_ind = static_list
        self.pdf_list = pdf_list
        self.pdf_name = pdf_name_list
        self.dim = self.x_u_dim + self.x_d_dim
        self.scales = scales

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
        # add exclusive options
        self.options.declare(
            "external_only",
            types=bool,
            default=False,
            desc="only use with surrogate. when design is updated, don't add new points at all",
        )   #TODO: This will likely need tweaking, and allow for both kinds of training data updates

        self.options.declare(
            "design_noise",
            types=float,
            default=0.0,
            desc="only use with surrogate. when sampling the uncertain space, add random noise to the dvs, scaled by this option",
        )

    def _new_sample(self, N):
        #TODO: options for this
        u_tx = self.sampling(N)
        tx = np.zeros([N, self.dim])
        tx[:, self.x_u_ind] = u_tx
        tx[:, self.x_d_ind] = self.x_d_cur#[self.x_d_cur[i] for i in self.x_d_ind]


        return tx

    def _refine_sample(self, N):
        tx = self.current_samples['x']
        noise = self.options["design_noise"]

        # just produce new LHS
        newsize = N + tx.shape[0]
        u_tx = self.sampling(newsize)
        tx = np.zeros([newsize, self.dim])
        tx[:, self.x_u_ind] = u_tx
        tx[:, self.x_d_ind] = self.x_d_cur
        # track matching points #TODO: standardize this
        # self.nested_ref_ind = range(tx.shape[0]).tolist()
        
        """
        function this out
        
        if noise > 1e-12:
            perturb = np.random.rand(self.x_d_dim)
            xlimits = self.options["xlimits"]
            d_xlimits = xlimits[self.x_d_ind]
            bound = np.zeros([self.x_d_dim, 2])
            # for i in range(self.x_d_dim):
            
            
            

            scale = d_xlimits[:,1] - d_xlimits[:,0]
            nlower = self.x_d_cur[:] - noise*scale
            bound[:,0] = np.maximum(d_xlimits[:][0], nlower)
            nupper = self.x_d_cur[:] + noise*scale
            bound[:,1] = np.minimum(d_xlimits[:][1], nupper)
            bscale = bound[:,1] - bound[:,0]
            bperturb = perturb*bscale - bound[:,0]

            tx[:, self.x_d_ind] += bperturb 
        """
            



        return tx

    """
    End of methods to override
    """

    def set_design(self, x_d_new):
        
        x_d_buf = x_d_new
        ret = 0
        print(f"{self.options['name']} Iter {self.iter_max}: Design {x_d_buf}")
        if np.allclose(x_d_buf, self.x_d_cur, rtol = 1e-15, atol = 1e-15):
            print(f"{self.options['name']} Iter {self.iter_max}: No change in design, returning")
            return ret # indicates that we have not moved, useful for gradient evals, avoiding retraining

        if not self.options["external_only"]:
            self.has_points = False
            if self.options["retain_uncertain_points"]:
                tx = copy.deepcopy(self.current_samples['x'])
                self._internal_save_state()
                self.current_samples['x'] = tx
                self.current_samples['x'][:, self.x_d_ind] = x_d_buf
                # self.x_samples[:, self.x_d_ind] = self.x_d_cur#[self.x_d_cur[i] for i in self.x_d_ind]
                self.has_points = True
            ret = 1
        else:
            print(f"{self.options['name']} Iter {self.iter_max}: Only design is changed, no data added")
            ret = 0
        self.x_d_cur = x_d_buf

        return ret # indicates that we have not moved, useful for gradient evals, avoiding retraining


    #NOTE: both generate_ and refine_ need an option to introduce noise to the sample
    def generate_uncertain_points(self, N):
        """
        First of two functions that will increment the sampling iteration

        Generates 

        Parameters
        ----------
        N: int or float or list
            generic reference to sampling level, by default just number of points to sample

        """
        # check if we already have them
        if self.has_points:
            print(f"{self.options['name']} Iter {self.iter_max}: Already have points, no points generated")
            return 0

        tx = self._new_sample(N)

        # archive previous dataset
        self._internal_save_state(refine=False)

        self.current_samples['x'] = tx
        self.has_points = True

        return 1

    def refine_uncertain_points(self, N):
        """
        Second of two functions that will increment the sampling iteration
        Add more UQ points to the current design. Usually this is nested

        N: int or float or list
            generic reference to sampling level, by default just number of points to add to sample
        """
        # check if we already have them NOTE: EVEN IF A DIFFERENT NUMBER IS REQUESTED FOR NOW
        if self.has_points and N is None:
            print(f"{self.options['name']} Iter {self.iter_max}: No refinement requested, no points generated")
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
    def _internal_save_state(self, refine=False, insert=False):
        """
        Internal version, increments sample counter since it's called just
        before updating the state

        Parameters:
        -----------
        refine: bool
            True if refining in place, false if traversing

        insert: bool
            True if points are added external to sampler generation
        """
        if self.iter_max < 0:
            self.iter_max += 1
            return

        affix = '_mov'
        if refine:
            affix = '_ref'
        if insert:
            affix = '_ins'

        name = self.options['name'] + '_' + str(self.iter_max) + affix
        self.save_state(name)

        # update design history, simple list
        self.design_history.append(self.x_d_cur)

        # increment iteration counter
        self._attribute_reset()
        self.iter_max += 1

    # add samples to this object from outside, in the format of current_samples
    #
    def add_data(self, new_samples, replace_current=False):

        # check that new_samples is good
        assert 'x' in new_samples

        if replace_current:

            # archive previous dataset
            self._internal_save_state(insert=True)

            self.current_samples['x'] = copy.deepcopy(new_samples['x'])
            self.has_points = True
            if 'f' in new_samples:
                self.set_evaluated_func(new_samples['f'])
            if 'g' in new_samples:
                self.set_evaluated_grad(new_samples['g'])

            self.stop_generating = False
        else:
            print(f"{self.options['name']} Iter {self.iter_max}: Adding points without replacing not implemented!")



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


"""
Sampler object for stochastic collocation points. N represents total polynomial order, though it may be adapted
on a per-direction basis
"""
class CollocationSampler(RobustSampler):
    


    def _initialize(self):
        # add exclusive options
        # self.options.declare(
        #     "external_only",
        #     types=bool,
        #     default=False,
        #     desc="only use with surrogate. when design is updated, don't add new points at all",
        # )   #TODO: This will likely need tweaking, and allow for both kinds of training data updates
        
        # run sampler for the first time on creation
        self.xlimits = self.options["xlimits"]

        

        # given pdfs, generate list of appropriate polynomials
        #TODO: Only works for uniform/legendre and normal/hermite, beta needs to pass args
        poly_list = []
        for i in range(self.x_u_dim):
            j = self.x_u_ind[i]
            pname = self.pdf_name[i]
            if pname == "beta":
                poly_list.append(lambda n: _poly_root_types["beta"](n, 
                                                                    alpha=self.pdf_list[j][1],
                                                                    beta=self.pdf_list[j][2]))
            else: # don't require additional args
                poly_list.append(_poly_root_types[pname])

        self.poly_list = poly_list
        self.weights = None
        
        self.absc_nsc = None
        self.weig_ind = None

        self.N_act = None
        self.jumps = None
        # N represents the order of the polynomial basis functions, not the samples directly (N_act)
        # If int, that is the order for all directions. If list (of length x_u_dim), apply to each 



        # u_xlimits = xlimits[self.x_u_ind]
        # self.sampling = LHS(xlimits=u_xlimits, criterion='maximin')

    def _declare_options(self):
        pass

    def _new_sample(self, N):

        self.N = N
        xlimits = self.options["xlimits"]
        if isinstance(N, int):
            self.N = self.x_u_dim*[N]

        # use recursion to form full tensor products
        N_act = self._recurse_total_points(0, self.N, 1)
        self.N_act = N_act

        # use recursion to get jumps for each dimension
        jumps = np.zeros(self.x_u_dim, dtype=int)

        # gather list of all abscissae in each direction
        absc = []
        absc_nsc = [] #no scale
        weig = []
        #TODO: ONLY WORKS FOR DISTS RANGING -1 to 1, NORMAL DIST IS UNCLEAR
        for i in range(self.x_u_dim):
            x_nsc, w = self.poly_list[i](self.N[i])
            # x = x*(self.scales[i]/2) + (0.5*self.scales[i] + xlimits[i,0])
            absc_nsc.append(x_nsc)
            x = 0.5*(x_nsc + 1.)*self.scales[i] + xlimits[i,0]
            absc.append(x)
            weig.append(w)
            jumps[i] = self._recurse_total_points(i, self.N, 1)/self.N[i]

        self.u_tx = np.zeros([N_act, self.x_u_dim])
        self.weights = np.ones(N_act)
        
        si = np.zeros(self.x_u_dim, dtype=int)
        self._recurse_sc_formation(0, si, jumps, absc, weig, self.N)

        # N_act = len(u_tx)
        # u_tx = np.array(u_tx)
        # save abscissae as well
        self.jumps = jumps
        self.absc_nsc = absc_nsc
        self.weig_ind = weig
        self.weights /= np.power(2, self.x_u_dim)
        tx = np.zeros([N_act, self.dim])
        tx[:, self.x_u_ind] = self.u_tx
        tx[:, self.x_d_ind] = self.x_d_cur#[self.x_d_cur[i] for i in self.x_d_ind]

        # import pdb; pdb.set_trace()
        return tx

    def _refine_sample(self, N):
        N_old = self.N

        if isinstance(N, int):
            N_add = self.x_u_dim*[N]
            
        N_new = np.array(N_old) + np.array(N_add)
        N_new = N_new.tolist()

        tx = self._new_sample(N)

        return tx

    def _recurse_sc_formation(self, di, si, jumps, absc, weig, N):
        
        N_cur = N[di]
        
        for i in range(N_cur):
            super_ind = np.dot(si, jumps)
            # recurse if more dimensions
            if di < self.x_u_dim - 1:
                self._recurse_sc_formation(di+1, si, jumps, absc, weig, N)
                si[di+1:] = 0
            
            self.weights[super_ind] = 1.0
            for j in range(self.x_u_dim):
                try:
                    self.u_tx[super_ind][j] = absc[j][si[j]]
                    self.weights[super_ind] *= weig[j][si[j]]
                except:
                    print("SC Recursion Failure!")
                    import pdb; pdb.set_trace()

            si[di] += 1
            

    def _recurse_total_points(self, di, N, tot):
        
        tot *= N[di]
        # recurse if more dimensions
        if di < self.x_u_dim - 1:
            tot = self._recurse_total_points(di+1, N, tot)

        return tot



if __name__ == '__main__':

    x_init = 0.
    # N = [5, 3, 2]
    # xlimits = np.array([[-1., 1.],
    #            [-1., 1.],
    #            [-1., 1.],
    #            [-1., 1.]])
    # pdfs =  [x_init, 'uniform', 'uniform', 'uniform']
    # samp1 = CollocationSampler(np.array([x_init]), N=N,
    #                             xlimits=xlimits, 
    #                             probability_functions=pdfs, 
    #                             retain_uncertain_points=True)
    
    
    from smt.problems import Rosenbrock
    from utils.sutils import convert_to_smt_grads
    N = [5, 3]
    pdfs = ['uniform', 'uniform']
    func = Rosenbrock(ndim=2)
    xlimits = func.xlimits
    samp = CollocationSampler(np.array([x_init]), N=N,
                                xlimits=xlimits, 
                                probability_functions=pdfs, 
                                retain_uncertain_points=True)
    


    xt = samp.current_samples['x']
    ft = func(xt)
    gt = convert_to_smt_grads(func, xt)
    samp.set_evaluated_func(ft)
    samp.set_evaluated_grad(gt)

    from utils.stat_comps import _mu_sigma_comp, _mu_sigma_grad
    stats, vals = _mu_sigma_comp(func, xt.shape[0], xt, xlimits, samp.scales, pdfs, tf = ft, weights=samp.weights)
    # gstats, grads = _mu_sigma_grad(func, xt.shape[0], xt, xlimits, samp.scales, pdfs, tf = ft, tg=gt, weights=samp.weights)
    import pdb; pdb.set_trace()
    #TODO: Make a test for this

