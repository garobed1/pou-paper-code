import numpy
from mpi4py import MPI
from baseclasses import *
from adflow import ADFLOW#, ADFLOW_C
import numpy as np

alpha = 0. #
beta = 10.0
mach = 2.6381157549933598
areaRef = 1.0
chordRef = 1.0
T = 254.02071103827234 
P = 4987.6905797938707
probName = 'impinge_mphys'
aeroGridFile = f'./imp_mphys_73_73_25.cgns'#f'../meshes/wing_vol.cgns'#

aeroOptions = { #ADflow aero solver options
    # Common Parameters
    'gridFile':aeroGridFile,
    'outputDirectory':'../results/',
    'writeTecplotSurfaceSolution':False,
    'writeSurfaceSolution':False,
    'writeVolumeSolution':False,
    
    # Physics Parameters
    'equationType':'Euler',
    'turbulenceModel':'SA',
    'turbulenceProduction':'vorticity',
    'useft2SA':True,
    'eddyVisInfRatio':3.0,

    # Common Parameters
    "CFL": 1.5,
    "CFLCoarse": 1.25,
    'MGCycle':'sg',
    'nCycles':100000,
    'monitorvariables':["resrho", "resturb"],
    'useNKSolver':True,
    'NKSwitchTol':1e-4,#e-1,
    'NKSubspaceSize':50,
    'NKPCILUFill':3,
    'NKLS':'none',
    'useANKSolver':True,
    'ANKCoupledSwitchTol':1e-3,
    'ANKConstCFLStep':0.4,
    'ANKCFLLimit':1000000000.0,
    "L2Convergence": 1e-14,
    "forcesAsTractions": False,
    
    # Adjoint options
    'adjointL2Convergence': 1e-6,
    # Output
    'volumeVariables':['eddyratio','mach','cp','temp'],
    'surfaceVariables':['yplus','cf','cp','cfx','cfy','cfz'],
    'printIterations':True,
    'printTiming':False,
    'printWarnings':True,
    'setMonitor':False
    }

# Create solver
CFDSolver = ADFLOW(options=aeroOptions)
#CFDSolverC = ADFLOW_C(options=aeroOptions)

# Aerodynamic problem description
ap = AeroProblem(
    name=probName,
    mach=mach,
    alpha =alpha,
    beta =beta,
    areaRef = 1.0,
    chordRef = 1.0,
    T = T, 
    P = P, 
    evalFuncs=["cdv"],
)
# apc = AeroProblem(
#     name=probName,
#     mach=mach,
#     alpha =alpha,
#     beta =beta,
#     areaRef = 1.0,
#     chordRef = 1.0,
#     T = T, 
#     P = P+0j, 
#     evalFuncs=["cdv"],
# )
ap.addDV("P")    
# apc.addDV("P")

# Solve and evaluate functions
funcs = {}
funcsens = {}
CFDSolver(ap)
CFDSolver.evalFunctions(ap, funcs)
CFDSolver.evalFunctionsSens(ap, funcsens)


# central difference check, directional derivative
# step size
h = 1e-4

# just do full fd/cs
fd = 0.


funcs2 = {'impinge_mphys_cdv':0+0j}
ap.P += h
CFDSolver(ap)
CFDSolver.evalFunctions(ap, funcs2)

funcs3 = {}
ap.P -= 2*h*1j
CFDSolver(ap)
CFDSolver.evalFunctions(ap, funcs3)

fd = (funcs2['impinge_mphys_cdv'] - funcs3['impinge_mphys_cdv'])/(2*h)
#fd = np.imag(funcs2['impinge_mphys_cdv'])/h

if MPI.COMM_WORLD.rank == 0:
    print(fd)
    print(funcsens['impinge_mphys_cdv'])