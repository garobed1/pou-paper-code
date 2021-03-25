import numpy
import openmdao.api as om
import plate_ffd as pf
import math
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
        meshname = self.aoptions['gridFile']
        gridFile = meshname
        
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
        #self.gridSol = f'{meshname}_{saconstsm}_sol'
        solname = self.ooptions['prob_name']
        self.gridSol = f'{solname}_sol'

        # Aerodynamic problem description
        self.ap = AeroProblem(name=self.gridSol, alpha=alpha, mach=mach, reynolds=Re, reynoldsLength=Re_L, T=temp, areaRef=arearef, chordRef=chordref, 
        evalFuncs=['cd'])

        # Create solver
        self.CFDSolver = ADFLOW(options=self.aoptions, comm=MPI.COMM_WORLD)
        self.CFDSolver.setDVGeo(self.DVGeo)

        # Set up mesh warping
        self.mesh = USMesh(options=self.woptions, comm=MPI.COMM_WORLD)
        self.CFDSolver.setMesh(self.mesh)

        self.CFDSolver.DVGeo.getFlattenedChildren()[1].writePlot3d("ffdp_opt_def.xyz")


        # Set constraints
        self.DVCon = DVConstraints()
        self.DVCon.setDVGeo(self.CFDSolver.DVGeo.getFlattenedChildren()[1])

        self.DVCon.setSurface(self.CFDSolver.getTriangulatedMeshSurface(groupName='allSurfaces'))

        # DV should be same into page (not doing anything right now)
        #import pdb; pdb.set_trace()
        lIndex = self.CFDSolver.DVGeo.getFlattenedChildren()[1].getLocalIndex(0)
        indSetA = []
        indSetB = []
        nXc = optOptions['NX']
        self.NC = math.trunc(((1.0 - self.ooptions['DVFraction'])*self.ooptions['NX']))
        ind = [int(nXc/2) - int(self.NC/2), int(nXc/2) + int(self.NC/2)]
        for i in range(ind[0], ind[1]):
            indSetA.append(lIndex[i, 0, 1])
            indSetB.append(lIndex[i, 1, 1])
        # for i in range(lIndex.shape[0]):
        #     indSetA.append(lIndex[i, 0, 1])
        #     indSetB.append(lIndex[i, 1, 1])
        self.DVCon.addLinearConstraintsShape(indSetA, indSetB, factorA=1.0, factorB=-1.0, lower=0, upper=0, name='eqs')

        # Thickness constraints (one for each active DV)
        #import pdb; pdb.set_trace()

        # Maximum thickness of the domain, translates to minimum thickness of bump
        ub = 1.0 - self.ooptions['DCMinThick']
        tcf = self.ooptions['DCThickFrac']
        ra = optOptions['bumpBounds']
        span = numpy.linspace(0, 1, nXc)
        xc = span * (ra[1] - ra[0]) + ra[0]
        #ind = range(int(nXc/2) - int(self.NC/2), int(nXc/2) + int(self.NC/2)))
        ind = [int(nXc/2) - int(tcf*self.NC/2), int(nXc/2) + int(tcf*self.NC/2)]
        ptList = numpy.zeros([2, 3])
        ptList[:,0] = xc[ind]
        ptList[:,1] = 0.5
        ptList[:,2] = 0.5

        self.DVCon.addThicknessConstraints1D(ptList, self.NC, [0,0,1], lower=0.5, upper=ub, name='tcs')

    def setup(self):
        #initialize shape and set deformation points as inputs
        a_init = self.CFDSolver.DVGeo.getValues()
        a_init['pnts'][:] = self.ooptions['DVInit']
        mult = numpy.linspace(1.0,1.5,num=int(0.5*len(a_init['pnts'])))
        mult = numpy.concatenate((mult, mult))
        a_init['pnts'] = numpy.multiply(mult, a_init['pnts'])
        self.add_input('a', a_init['pnts'], desc="Bump Shape Control Points")
        #self.add_input('a', 0.2, desc="Bump Shape Control Points")

        self.add_output('TC', numpy.zeros(self.NC), desc='Thickness Constraints')
        self.add_output('Cd', 0.0, desc="Drag Coefficient")
        self.add_output('EQ', numpy.zeros(int(len(a_init['pnts'])/2)), desc="Control Point Symmetry")


    
    def setup_partials(self):
        self.declare_partials('Cd','a', method='exact')
        self.declare_partials('TC','a', method='exact')
        self.declare_partials('EQ','a', method='exact')
    
    def compute(self, inputs, outputs):
        # run the bump shape model

        # move the mesh
        #import pdb; pdb.set_trace()
        dvdict = {'pnts':inputs['a']}
        self.CFDSolver.DVGeo.setDesignVars(dvdict)
        self.ap.setDesignVars(dvdict)

        #self.CFDSolver.DVGeo.update("coords")

        # Solve and evaluate functions
        funcs = {}
        self.CFDSolver(self.ap)
        self.DVCon.evalFunctions(funcs, includeLinear=True)
        self.CFDSolver.evalFunctions(self.ap, funcs)

        str = self.gridSol + '_cd'
        outputs['Cd'] = funcs[str]
        outputs['TC'] = funcs['tcs']
        outputs['EQ'] = funcs['eqs']

        #outputs['Cd'] = inputs['a']*inputs['a']

    def compute_partials(self, inputs, J):

        # move the mesh
        #import pdb; pdb.set_trace()
        dvdict = {'pnts':inputs['a']}
        self.CFDSolver.DVGeo.setDesignVars(dvdict)
        self.ap.setDesignVars(dvdict)
        #self.CFDSolver.DVGeo.update("coords")
 
 
        funcSens = {}
        self.CFDSolver(self.ap)
        self.DVCon.evalFunctionsSens(funcSens, includeLinear=True)
        self.CFDSolver.evalFunctionsSens(self.ap, funcSens, ['cd'])
 
        str = self.gridSol + '_cd'
        J['Cd','a'] = funcSens[str]['pnts']
        J['TC','a'] = funcSens['tcs']['pnts']
        J['EQ','a'] = funcSens['eqs']['pnts']

       #J['Cd','a'][0] = 2*inputs['a']


