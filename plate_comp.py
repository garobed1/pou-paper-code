import numpy
import openmdao.api as om
import plate_ffd as pf
from mpi4py import MPI
from idwarp import USMesh
from baseclasses import *
from adflow import ADFLOW
from pygeo import DVGeometry, DVConstraints
from plate_comp_opts import aeroOptions, warpOptions, optOptions

class PlateComponent(om.ExplicitComponent):
    """Deterministic Bump Flow Problem"""
    def initialize(self):
        # Need to modify this dictionary when we change the SA constants
        self.aoptions = aeroOptions
        self.woptions = warpOptions
        self.ooptions = optOptions

        # Generate FFD and DVs
        self.DVGeo = pf.createFFD()

        # starting flat mesh
        meshname = 'bump_69x49_0.1_0.05_0.02'
        gridFile = f'./{meshname}.cgns'
        self.aoptions['gridFile'] = gridFile
        
        # flow characteristics
        alpha = 0.0
        mach = 0.8
        Re = 50000
        Re_L = 1.0
        temp = 540
        arearef = 2.0
        chordref = 1.0

        # Spalart Allmaras model constants, to be changed in UQ
        saconstsm = [0.41, 0.1355, 0.622, 0.66666666667, 7.1, 0.3, 2.0]
        self.saconsts = saconstsm + [1.0, 2.0, 1.2, 0.5, 2.0]
        self.aoptions['SAConsts'] = self.saconsts
        self.gridSol = f'{meshname}_{saconstsm}_sol'

        # Aerodynamic problem description
        self.ap = AeroProblem(name=self.gridSol, alpha=alpha, mach=mach, reynolds=Re, reynoldsLength=Re_L, T=temp, areaRef=arearef, chordRef=chordref, 
        evalFuncs=['cd'])

        # Create solver
        self.CFDSolver = ADFLOW(options=aeroOptions, comm=MPI.COMM_WORLD)
        self.CFDSolver.setDVGeo(self.DVGeo)

        # Set up mesh warping
        self.mesh = USMesh(options=self.woptions, comm=MPI.COMM_WORLD)
        self.CFDSolver.setMesh(self.mesh)

        # Set constraints
        self.DVCon = DVConstraints()
        self.DVCon.setDVGeo(self.DVGeo)

        self.DVCon.setSurface(self.CFDSolver.getTriangulatedMeshSurface())

        # DV should be same along spanwise
        lIndex = self.DVGeo.getLocalIndex(0)
        indSetA = []
        indSetB = []
        for i in range(lIndex.shape[0]):
            indSetA.append(lIndex[i, 0, 0])
            indSetB.append(lIndex[i, 0, 1])
        for i in range(lIndex.shape[0]):
            indSetA.append(lIndex[i, 1, 0])
            indSetB.append(lIndex[i, 1, 1])
        self.DVCon.addLinearConstraintsShape(indSetA, indSetB, factorA=1.0, factorB=-1.0, lower=0, upper=0)

    def setup(self):
        #initialize shape and set deformation points as inputs
        a_init = self.DVGeo.getValues()
        a_init['pnts'][:] = self.ooptions['DVUpperBound']
        self.add_input('a', a_init['pnts'], desc="Bump Shape Control Points")

        self.add_output('Cd', 0.0, desc="Drag Coefficient")
    
    def setup_partials(self):
        self.declare_partials('Cd','a')
    
    def compute(self, inputs, outputs):
        # run the bump shape model

        # Solve and evaluate functions
        funcs = {}
        self.CFDSolver(self.ap)
        self.CFDSolver.evalFunctions(self.ap, funcs)

        str = self.gridSol + '_cd'
        outputs['Cd'] = funcs[str]

    def compute_partials(self, inputs, J):

        funcSens = {}
        self.CFDSolver.evalFunctionsSens(self.ap, funcSens, ['cd'])

        str = self.gridSol + '_cd'
        J['Cd','a'] = funcSens[str]


