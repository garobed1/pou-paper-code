import numpy
import os, sys
import openmdao.api as om
import plate_ffd as pf
import math
import plate_sa_lhs
import time
from mpi4py import MPI
from idwarp import USMesh
from baseclasses import *
from adflow import ADFLOW
from pygeo import DVGeometry, DVConstraints
from plate_comp_opts import aeroOptions, warpOptions, optOptions, uqOptions

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

class WMHComponentRobust(om.ExplicitComponent):
    """Bump-In-Channel Validation Problem"""
    def initialize(self):
        # Need to modify this dictionary when we change the SA constants
        #sys.stdout = open(os.devnull, "w")
        self.aoptions = aeroOptions
        self.woptions = warpOptions
        self.ooptions = optOptions
        self.uoptions = uqOptions


        # starting flat mesh
        meshname = self.aoptions['gridFile']
        gridFile = meshname
        
        # flow characteristics
        alpha = 0.0
        mach = self.ooptions['mach']#0.95
        Re = self.ooptions['Re']#50000
        Re_L = 1.0
        temp = 537
        arearef = 1.0
        chordref = 1.0

        # Spalart Allmaras model constants, to be changed in UQ
        saconstsm = [0.41, 0.1355, 0.622, 0.66666666667]
        self.saconstsb = [7.1, 0.3, 2.0, 1.0, 2.0, 1.2, 0.5, 2.0]
        self.saconsts = saconstsm + self.saconstsb
        #self.aoptions['SAConsts'] = self.saconsts
        #self.gridSol = f'{meshname}_{saconstsm}_sol'
        solname = self.ooptions['prob_name']
        self.gridSol = f'{solname}_sol'

        self.NS = self.uoptions['NS']
        if rank == 0:
            rank0sam = plate_sa_lhs.genLHS(s=self.NS, mcs = self.uoptions['MCPure'])
        else:
            rank0sam = None
        self.sample = comm.bcast(rank0sam, root=0)

        self.cases = divide_cases(self.NS, size)
        self.nsp = len(self.cases[rank])#int(ns/size) # samples per processor
        self.samplep = self.sample[self.cases[rank]]#self.sample[(rank*self.nsp):(rank*self.nsp+(self.nsp))] #shouldn't really need to "scatter" per se
        #assert len(self.samplep) == self.nsp

        # Actually create solvers (and aeroproblems?) (and mesh?) now
        self.aps = []
        self.solvers = []
        self.meshes = []

        # create aeroproblems 
        for i in range(self.nsp):
            namestr = self.gridSol + "_" + str(self.cases[rank][i])
            self.aps.append(AeroProblem(name=namestr, alpha=alpha, mach=mach, reynolds=Re, reynoldsLength=Re_L, T=temp, areaRef=arearef, chordRef=chordref, evalFuncs=['cdp']))
            self.aps[i].setBCVar("PressureStagnation", 1.007*self.aps[i].__dict__["P"], 'inflow')
            self.aps[i].setBCVar("TemperatureStagnation", 1.002*self.aps[i].__dict__["T"], 'inflow')
            self.aps[i].setBCVar("VelocityUnitVectorX", 1.0, 'inflow')
            self.aps[i].setBCVar("VelocityUnitVectorY", 0.0, 'inflow')
            self.aps[i].setBCVar("VelocityUnitVectorY", 0.0, 'inflow')
            self.aps[i].setBCVar("Pressure", 0.99962*self.aps[i].__dict__["P"], 'outflow')
        self.meshes = USMesh(options=self.woptions, comm=MPI.COMM_SELF)

        # Create solver
        self.solvers = ADFLOW(options=self.aoptions, comm=MPI.COMM_SELF)
        time.sleep(0.1)
        self.solvers.setOption('SAConsts', self.saconsts)
        self.solvers.setMesh(self.meshes)

        
        #import pdb; pdb.set_trace()

        sys.stdout = sys.__stdout__

    def setup(self):
        self.add_output('Cd_max', 0.0, desc="Drag Coefficient Maximal")
        self.add_output('Cd_min', 0.0, desc="Drag Coefficient Minimal")

        saconstsm = [0.41, 0.1355, 0.622, 0.66666666667]
        self.add_output('sac_max', saconstsm, desc="SA Coeffs Maximal")
        self.add_output('sac_min', saconstsm, desc="SA Coeffs Minimal")
        self.add_output('ind_max', 0, desc="Maximal Case")
        self.add_output('ind_min', 0, desc="Minimal Case")
    
    def compute(self, inputs, outputs):
        # run the model

        funcs = {}
        musp = numpy.zeros(self.nsp)
        #self.aps[0].setDesignVars(dvdict)
        for i in range(self.nsp):
            saconstsm = self.samplep[i].tolist()
            self.saconsts = saconstsm + self.saconstsb
            self.solvers.setOption('SAConsts', self.saconsts)
            self.solvers(self.aps[i])
            self.solvers.evalFunctions(self.aps[i], funcs)
            astr = self.gridSol + "_" + str(self.cases[rank][i]) +"_cdp"
            musp[i] = funcs[astr]

        mus = comm.allgather(musp)
        minc = min(map(min,mus))
        maxc = max(map(max,mus))
        index_min_1 = numpy.argmin(list(map(min,mus)))
        index_max_1 = numpy.argmax(list(map(max,mus)))
        index_min_2 = numpy.argmin(mus[index_min_1])
        index_max_2 = numpy.argmax(mus[index_max_1])
        outputs['Cd_min'] = minc
        outputs['Cd_max'] = maxc
        outputs['ind_min'] = self.cases[index_min_1][index_min_2]
        outputs['ind_max'] = self.cases[index_max_1][index_max_2]
        outputs['sac_min'] = self.sample[self.cases[index_min_1][index_min_2]]
        outputs['sac_max'] = self.sample[self.cases[index_max_1][index_max_2]]




def divide_cases(ncases, nprocs):
    """
    From parallel OpenMDAO beam problem example

    Divide up load cases among available procs.
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
