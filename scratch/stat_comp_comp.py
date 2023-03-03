import numpy as np
import openmdao.api as om
from utils.error import stat_comp
from optimization.robust_objective import RobustSampler
"""
Compute some statistical measure of a model at a given design
"""
class StatCompComponent(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('sampler', desc="object that tracks samples of the func")
        self.options.declare('pdfs', desc="prob dists of inputs")
        self.options.declare('func', desc="SMT Func handle", recordable=False)
        self.options.declare('eta', desc="mean to stdev ratio")
        self.options.declare('stat_type', desc="Type of robust function to compute")

        self.sampler = None
        self.func = None
        self.eta = None
        self.stat_type = None
        self.pdfs = None

    def setup(self):
        
        self.sampler = self.options["sampler"]
        self.func = self.options["func"]
        self.eta = self.options["eta"]
        self.stat_type = self.options["stat_type"]
        self.pdfs = self.options["pdfs"]

        # inputs
        self.add_input('x_d', shape=1,
                              desc='Current design point')
        
        self.add_output('musigma', shape=1,
                                   desc='mean + stdev at design point')

        self.declare_partials('*','*')

    def compute(self, inputs, outputs):

        x = inputs['x_d']
        eta = self.eta

        self.pdfs[1] = x
        self.sampler.set_design(np.array([x]))
        self.sampler.generate_uncertain_points(self.sampler.N)
        res = stat_comp(None, self.func, 
                                stat_type=self.stat_type, 
                                pdfs=self.pdfs, 
                                xdata=self.sampler)
        fm = res[0]
        fs = res[1]

        outputs['musigma'] = eta*fm + (1-eta)*fs

    def compute_partials(self, inputs, partials):

        x = inputs['x_d']
        eta = self.eta

        self.pdfs[1] = x
        self.sampler.set_design(np.array([x]))
        self.sampler.generate_uncertain_points(self.sampler.N)
        gres = stat_comp(None, self.func, 
                                get_grad=True, 
                                stat_type=self.stat_type, 
                                pdfs=self.pdfs, 
                                xdata=self.sampler)
        gm = gres[0]
        gs = gres[1]

        partials['musigma','x_d'] = eta*gm + (1-eta)*gs

    def get_fidelity(self):
        # return current number of samples
        return self.sampler.current_samples['x'].shape[0]
    
    def refine_model(self, N):

        self.sampler.N += N
        self.sampler.refine_uncertain_points(N)