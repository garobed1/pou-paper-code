
import numpy
from mpi4py import MPI
from baseclasses import *
from adflow import ADFLOW

aeroGridFile = './imp_test_shock_121_61_1.cgns'
alpha = 0.0
mach = 3#5.0
T = 217#410
P = 2919#2120000
areaRef = 1.0
chordRef = 1.0
reRef = 0.254
reynolds = 40000000*reRef
name = 'fc'

aeroOptions = { #ADflow aero solver options
    # Common Parameters
    'gridFile':aeroGridFile,
    'outputDirectory':'./results/',
    'writeTecplotSurfaceSolution':False,
    'writeSurfaceSolution':True,
    'writeVolumeSolution':True,
    
    # Physics Parameters
    'equationType':'Euler',
    'turbulenceModel':'SA',
    'turbulenceProduction':'vorticity',
    'useft2SA':True,
    'eddyVisInfRatio':3.0,
    # [kappa, cb1,    cb2,   sigma,         cv1, cw2, cw3, ct1, ct2, ct3, ct4, rot]
    # [0.41,  0.1355, 0.622, 0.66666666667, 7.1, 0.3, 2.0, 1.0, 2.0, 1.2, 0.5, 2.0]
    #'SAConsts':[0.41,  0.1355, 0.622, 0.66666666667, 7.1, 0.3, 2.0, 1.0, 2.0, 1.2, 0.5, 2.0],
    
    # Common Parameters
    "CFL": 1.5,
    "CFLCoarse": 1.25,
    'MGCycle':'sg',
    'nCycles':100,
    'monitorvariables':["resrho", "resturb","resrhoe"],
    'useNKSolver':True,
    'NKSwitchTol':1e-3,
    'NKSubspaceSize':200,
    'NKPCILUFill':3,
    'NKLS':'none',
    'useANKSolver':True,
    'ANKCoupledSwitchTol':1e-2,
    'ANKConstCFLStep':0.4,
    'ANKCFLLimit':1000000000.0,
    "L2Convergence": 1e-8,
    # "L2ConvergenceCoarse": 1e-2,
    # "L2ConvergenceRel": 1e-4,
    "forcesAsTractions": False,
    
    # Design options
    #'meshSurfaceFamily':'allSurfaces',
    #'designSurfaceFamily':'allSurfaces',

    # Adjoint options
    'adjointL2Convergence': 1e-06,

    # Output
    'volumeVariables':['eddyratio','mach','cp'],
    'surfaceVariables':['yplus','cf','cp','cfx','cfy','cfz'],
    'printIterations':True,
    'printTiming':False,
    'printWarnings':False,
    'setMonitor':False
    }


# Aerodynamic problem description
ap = AeroProblem(name=name, alpha=alpha, mach=mach, P=P, #reynolds=reynolds,
areaRef=areaRef, chordRef=chordRef, reynoldsLength = reRef, T = T,
evalFuncs=['cl','cd'])

# Create solver
CFDSolver = ADFLOW(options=aeroOptions)

# Solve and evaluate functions
funcs = {}
CFDSolver(ap)
CFDSolver.evalFunctions(ap, funcs)

# Print the evaluatd functions
if MPI.COMM_WORLD.rank == 0:
    print(funcs)