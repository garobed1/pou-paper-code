import numpy
import os, sys
import openmdao.api as om
import plate_ffd as pf
import math
from mpi4py import MPI
from idwarp import USMesh
from baseclasses import *
from adflow import ADFLOW
import sa_lhs
import numpy as np
from pygeo import DVGeometry, DVConstraints
from mphys import MultipointParallel

"""
This component manages all the aerostructural cycles corresponding to each uncertain sample

May need to implement multifidelity methods here
"""

# Base class for sample-based UQ implementation
class SampleUQGroup(MultipointParallel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.dim = len(kwargs)

        # Get keywords to determine random independent inputs and their distributions (e.g. kwargs = {'kappa': [0.36814, 0.50683, 'u']}) 

        # Stamp out all the points you need
        #TODO: allow for a variable number of SA sample points, eventually incorporate other uncertainties
        self.sample = self.generate_samples()
        self.add('sample_points', om.IndepVarComp('xi', val=np.zeros((len(self.sample), self.dim))),
                                        promotes=['xi'])

        #TODO: develop an aggregate container for outputs of each sample group (e.g. drag, etc.)
        self.add('aggregate', np.zeros(len(self.sample)))

        cycleg = self.add('multi_point', om.ParallelGroup())

        
        for i in range(np):
            s_name = 's%d'%i
            cycleg.add(s_name, om.Point(i))
            self.connect('x', 'multi_point.%s.x'%s_name, src_indices=[i])
            self.connect('w', 'multi_point.%s.w'%s_name, src_indices=[i])
            self.connect('multi_point.%s.Cd'%s_name,'aggregate.Cd_%d'%i)

    # Generate samples and weights
    def generate_samples():
        return 0 # This method must be overloaded

    def compute_statistics():
        return 0

# Monte Carlo Points
class MonteCarloGroup(SampleUQGroup):

    def __init__(self, ns, **kwargs):
        self.ns = ns

        super().__init__(**kwargs)

    def generate_samples():
        x = zeros
        return x, w
