import numpy
import os, sys
import openmdao.api as om
import plate_ffd as pf
import math
from mpi4py import MPI
from idwarp import USMesh
from baseclasses import *
from adflow import ADFLOW
from pygeo import DVGeometry, DVConstraints
from plate_comp_opts import aeroOptions, warpOptions, optOptions

class BICComponent(om.ExplicitComponent):
    """Bump-In-Channel Validation Problem"""
    def initialize(self):
        # Need to modify this dictionary when we change the SA constants
        #sys.stdout = open(os.devnull, "w")
        self.aoptions = aeroOptions
        self.woptions = warpOptions
        self.ooptions = optOptions


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
        self.saconsts = saconstsm + [7.1, 0.3, 2.0, 1.0, 2.0, 1.2, 0.5, 2.0]
        #self.aoptions['SAConsts'] = self.saconsts
        #self.gridSol = f'{meshname}_{saconstsm}_sol'
        solname = self.ooptions['prob_name']
        self.gridSol = f'{solname}_sol'

        # Aerodynamic problem description
        self.ap = AeroProblem(name=self.gridSol, alpha=alpha, mach=mach, reynolds=Re, reynoldsLength=Re_L, T=temp, areaRef=arearef, chordRef=chordref, 
            evalFuncs=['cd'])

        # Create solver
        self.CFDSolver = ADFLOW(options=self.aoptions, comm=MPI.COMM_WORLD)
        #self.CFDSolver.setOption('SAConsts', self.aoptions['SAConsts'])
        # Set up mesh warping
        self.mesh = USMesh(options=self.woptions, comm=MPI.COMM_WORLD)
        self.CFDSolver.setMesh(self.mesh)

        self.ap.setBCVar("PressureStagnation", 1.007*self.ap.__dict__["P"], 'inflow')
        self.ap.setBCVar("TemperatureStagnation", 1.002*self.ap.__dict__["T"], 'inflow')
        self.ap.setBCVar("VelocityUnitVectorX", 1.0, 'inflow')
        self.ap.setBCVar("VelocityUnitVectorY", 0.0, 'inflow')
        self.ap.setBCVar("VelocityUnitVectorY", 0.0, 'inflow')
        self.ap.setBCVar("Pressure", 0.99962*self.ap.__dict__["P"], 'outflow')
        #import pdb; pdb.set_trace()
        # Try setting the DVGeo coordinates here
        coords = self.CFDSolver.getSurfaceCoordinates(groupName=self.CFDSolver.allWallsGroup)

        sys.stdout = sys.__stdout__

    def setup(self):
        self.add_output('Cd', 0.0, desc="Drag Coefficient")
    
    def compute(self, inputs, outputs):
        # run the model

        funcs = {}
        self.CFDSolver(self.ap)
        self.CFDSolver.evalFunctions(self.ap, funcs)

        str = self.gridSol + '_cd'
        outputs['Cd'] = funcs[str]



