import numpy as np
import time

from smt.surrogate_models.surrogate_model import SurrogateModel
from smt.utils.options_dictionary import OptionsDictionary
from optimization.robust_objective import CollocationSampler
from collections import defaultdict
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.special import legendre, hermite, jacobi
from scipy.interpolate import lagrange
from scipy.stats import qmc
from utils.sutils import estimate_pou_volume, innerMatrixProduct, quadraticSolveHOnly, symMatfromVec
from utils.sutils import standardization2


_poly_types = {
    "uniform": legendre,
    "norm": hermite,
    "beta": jacobi
}

#from pou_cython_ext import POUEval

"""
Polynomial Chaos Expansion-based surrogate model, using strictly data at
collocation points. CollocationSampler object REQUIRED
"""
class PCEStrictSurrogate(SurrogateModel): 
    name = "PCEStrict"
    """
    Create the surrogate object

    Parameters
    ----------

    """
    _poly_types = {
        "legendre": legendre
    }
    
    
    def _initialize(self):#, xcenter, func, grad, rho, delta=1e-10):
        # initialize data and parameters
        super(PCEStrictSurrogate, self)._initialize()
        declare = self.options.declare

        declare(
            "bounds",
            None,
            types=(list, np.ndarray),
            desc="Domain boundaries"
        )

        declare(
            "sampler",
            None,
            desc="sampler object for collocation points. REQUIRED"
        )

        self.sampler = None
        self.xt = None
        self.ft = None
        self.we = None
        self.po = None
        self.ld = None
        # self.la = None # list of lists of lagrange polynomials in each direction

        # self.set_collocation_sampler(self.options["sampler"])
        #NOTE: No options for sparsity yet

        self.supports["training_derivatives"] = False



    
    def _train(self) -> None:
        self.set_collocation_sampler(self.options["sampler"])



        return 0
    
    """
    Assign collocation data here, sampler needs evaluated func set
    """
    def set_collocation_sampler(self, sampler):

        if sampler is None:
            print("Need to set a sampler object in options!")
            return
        
        xlimits = self.options['bounds']
        self.dim = sampler.x_u_dim

        self.xt = sampler.current_samples["x"][sampler.x_u_ind]
        self.ft = sampler.current_samples["f"]
        # self.we = sampler.weights
        # self.po = sampler.

        # scaled basis centers
        self.xt_scale = np.zeros_like(self.xt)
        absc = sampler.absc_nsc

        # denominators of lagrange polynomials
        self.ld = []#np.ones(self.dim)
        for i in range(self.dim):
            self.xt_scale[:,i] = 2.*(self.xt[:,i] - xlimits[i,0])/sampler.scales[i] - 1.

            ncur = sampler.N[i]
            self.ld.append(np.ones(ncur)) #N[i] denominators per direction
            for j in range(ncur):
                for k in range(ncur):
                    if j != k:
                        self.ld[i][j] *= absc[i][j] - absc[i][k]

        # poly_list = []
        # for i in range(sampler.x_u_dim):
        #     j = sampler.x_u_ind[i]
        #     pname = sampler.pdf_name[i]
        #     if pname == "beta":
        #         poly_list.append(lambda n: _poly_types["beta"](n, 
        #                                                             alpha=self.pdf_list[j][1],
        #                                                             beta=self.pdf_list[j][2]))
        #     else: # don't require additional args
        #         poly_list.append(_poly_types[pname])

        # self.po = poly_list



        self.sampler = sampler

    def _predict_values(self, x: np.ndarray) -> np.ndarray:
        xlimits = self.options['bounds']
        sampler = self.sampler
        dim = self.dim

        # scale input x (may need to scale y as well)
        X_cont = np.zeros_like(x)
        nq = x.shape[0]
        absc = sampler.absc_nsc
        N = sampler.N
        N_act = sampler.N_act

        # lagrange numerators
        ln = []
        for i in range(dim):
            # scale each column
            X_cont[:,i] = 2.*(x[:,i] - xlimits[i,0])/sampler.scales[i] - 1.

            # store every numerator
            ncur = sampler.N[i]
            ln.append(np.zeros([nq, ncur]))
            for j in range(ncur):
                ln[i][:,j] = X_cont[:,i] - absc[i][j] #vec - scalar
        
        # now multiply 1D numerators
        ld = self.ld
        acc = []
        ft = self.ft
        for i in range(dim):
            ncur = N[i]
            ind = np.arange(0, ncur)
            acc.append(np.ones([nq, N[i]]))
            for j in range(ncur):
                # import pdb; pdb.set_trace()
                acc[i][:,j] = np.prod(ln[i][:,ind != j], axis=1)/ld[i][j]

        # now recurse to get every polynomial term for each query
        self.Ls = np.ones([nq, N_act])
        si = np.zeros(dim, dtype=int)
        # self.Ls = np.outer(acc[0], acc[1]).flatten()
        self._recurse_sc_interp(0, si, sampler.jumps, acc, N)
        
        #finally

        y = np.matmul(self.Ls, ft)
        import pdb; pdb.set_trace()
        return y

    def _recurse_sc_interp(self, di, si, jumps, acc, N):
        
        N_cur = N[di]
        
        for i in range(N_cur):
            super_ind = np.dot(si, jumps)
            # recurse if more dimensions
            if di < self.dim - 1:
                self._recurse_sc_interp(di+1, si, jumps, acc, N)
                si[di+1:] = 0

            else:            
                for j in range(self.dim):
                    try:
                        self.Ls[:,super_ind] *= acc[j][:,si[j]]
                        # import pdb; pdb.set_trace()
                    except:
                        print("SC Recursion Failure!")
                        import pdb; pdb.set_trace()

            si[di] += 1

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

    surr = PCEStrictSurrogate(bounds=xlimits, sampler=samp)
    surr.set_training_values(samp.current_samples['x'], samp.current_samples['f'])
    surr.train()

    x = np.array([[-0.4,-0.5],[1.0, 1.2],[1.5, -0.5]])
    # x = np.array([[2.0,2.0]])

    y = surr.predict_values(x)

       # gstats, grads = _mu_sigma_grad(func, xt.shape[0], xt, xlimits, samp.scales, pdfs, tf = ft, tg=gt, weights=samp.weights)
    import pdb; pdb.set_trace()
    #TODO: Make a test for this
    #TODO: SCALE, PLANE WORK