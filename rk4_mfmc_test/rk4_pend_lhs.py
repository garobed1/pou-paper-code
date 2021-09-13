import numpy
import math
import sys, os
import openmdao.api as om
import plate_ffd as pf
import math
import plate_sa_lhs
import subprocess
import time
from mpi4py import MPI
from rk4_pend import rk4_pend_solver
from rk4_pend_opts import uqOptions

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

class RK4PendCompLHS(om.ExplicitComponent):
    """Stochastic Damped Pendulum Problem, with LHS Monte Carlo Samples"""
    def initialize(self):
        # Need to modify this dictionary when we change the SA constants
        #import pdb; pdb.set_trace()
        self.uoptions = uqOptions
        self.NS0 = self.uoptions['NS0']
        self.NS = self.uoptions['NS']
        self.P = self.uoptions['P']

        self.L = self.uoptions['L']
        self.m = self.uoptions['m']
        self.g = self.uoptions['g']
        self.crange = self.uoptions['crange']

        self.Lmax = self.uoptions['levels']

        self.solver = rk4_pend_solver()

        if self.uoptions['MCTimeBudget']:
            # Generate FFD and DVs
            if rank == 0:
                csamp = self.crange[0] + (self.crange[1] - self.crange[0])*numpy.random.rand(self.NS0)
            else:
                csamp = None
            self.csample = comm.bcast(csamp, root=0)

            # Scatter samples, multi-point parallelism
            self.cases = divide_cases(self.NS0, size)
            self.nsp = len(self.cases[rank])#int(ns/size) # samples per processor
            self.csamplep = self.csample[self.cases[rank]]

            # Compute number of samples based on budget
            sumtp = 0.
            sumt = 0.
            tspan = numpy.zeros(2)
            u0 = numpy.zeros(2)
            tspan[0] = 0.
            tspan[1] = self.uoptions['Tf']
            u0[0] = self.uoptions['u0'][0]
            u0[1] = self.uoptions['u0'][1]
            num_steps = 2**(5+self.Lmax-1)
            for i in range(self.nsp):
                num_steps = 2**(5+self.Lmax-1)
                c = self.csamplep[i]
                pc0 = time.process_time()
                SOL = self.solver.solve_for_uf(self.m, self.g, c, self.L, tspan, u0, num_steps)
                pc1 = time.process_time()
                sumtp += (pc1-pc0)
            sumt = comm.allreduce(sumtp)
            #import pdb; pdb.set_trace()
            Et = sumt/self.NS0
            self.NS = math.ceil(self.P/Et)
        else:
            self.NS = self.uoptions['NS']

        # Generate new samples and scatter
        if rank == 0:
            csamp = self.crange[0] + (self.crange[1] - self.crange[0])*numpy.random.rand(self.NS)
        else:
            csamp = None
        self.csample = comm.bcast(csamp, root=0)
        #import pdb; pdb.set_trace()
        self.cases = divide_cases(self.NS, size)
        self.nsp = len(self.cases[rank])#int(ns/size) # samples per processor

        self.csamplep = self.csample[self.cases[rank]]

        # Actually create solvers (and aeroproblems?) (and mesh?) now
        

    def setup(self):
        self.add_output('uf_m', 0.0, desc="Mean Final Position")
        self.add_output('uf_v', 0.0, desc="Final Position Variance")
        self.add_output('N1', self.NS, desc="Number of Samples Used")
    
    def compute(self, inputs, outputs):
        # run the bump shape model
        #import pdb; pdb.set_trace()
        # evaluate each sample point
        #print("hello")
        sump = 0.
        musp = numpy.zeros(self.nsp)
        tspan = numpy.zeros(2)
        u0 = numpy.zeros(2)
        tspan[0] = 0.
        tspan[1] = self.uoptions['Tf']
        u0[0] = self.uoptions['u0'][0]
        u0[1] = self.uoptions['u0'][1]
        num_steps = 2**(5+self.Lmax-1)
        #self.aps[0].setDesignVars(dvdict)
        for i in range(self.nsp):
            c = self.csamplep[i]
            SOL = self.solver.solve_for_uf(self.m, self.g, c, self.L, tspan, u0, num_steps)
            musp[i] = SOL
            sump += SOL
        
        # compute mean and variance
        sum = comm.allreduce(sump)
        mus = comm.allgather(musp)
        #import pdb; pdb.set_trace()
        E = sum/self.NS
        sum2 = 0.
        for i in range(len(mus)): #range(size):
            for j in range(len(mus[i])): #range(self.nsp):
                sum2 += (mus[i][j]-E)**2
        V = sum2/self.NS

        #import pdb; pdb.set_trace()
        outputs['uf_m'] = E
        outputs['uf_v'] = math.sqrt(V)
        outputs['N1'] = self.NS

def divide_cases(ncases, nprocs):
    """
    From parallel OpenMDAO beam problem example

    Divide up load cases among available procs.

    Parameters
    ----------
    ncases : int
        Number of load cases.
    nprocs : int
        Number of processors.

    Returns
    -------
    list of list of int
        Integer case numbers for each proc.
    """
    data = []
    for j in range(nprocs):
        data.append([])

    wrap = 0
    for j in range(ncases):
        idx = j - wrap
        if idx >= nprocs:
            idx = 0
            wrap = j

        data[idx].append(j)

    return data
