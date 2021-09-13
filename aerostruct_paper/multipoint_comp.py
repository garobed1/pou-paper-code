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
from pygeo import DVGeometry, DVConstraints

"""
This component manages all the aerostructural cycles corresponding to each uncertain sample

May need to implement multifidelity methods here
"""

# Monte Carlo implementation
class MonteCarloGroup(om.Group):

    def __init__(self, np, lhs = False):
        super(om.ParallelMultiPoint, self).__init__()


        self.add('sample_points', om.IndepVarComp('xi', val=np.zeros((np, 4))),
                                        promotes=['xi'])

        #TODO: develop an aggregate container for outputs of each sample group (e.g. drag, etc.)
        self.add('aggregate', np.zeros(np))


        cycleg = self.add('multi_point', om.ParallelGroup())

        #Stamp out all the points you need
        #TODO: allow for a variable number of SA sample points, eventually incorporate other uncertainties
        self.sample = sa_lhs.genLHS(s=np, mcs = lhs)
        for i in range(np):
            s_name = 's%d'%i
            cycleg.add(s_name, om.Point(i))
            self.connect('x', 'multi_point.%s.x'%s_name, src_indices=[i])
            self.connect('multi_point.%s.Cd'%s_name,'aggregate.Cd_%d'%i)

            