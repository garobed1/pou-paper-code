# ======================================================================
#         Import modules
# ======================================================================
import numpy
from mpi4py import MPI
from baseclasses import *
from adflow import ADFLOW

# ======================================================================
#         Input Information -- Modify accordingly!
# ======================================================================
outputDirectory = './'
name = 'ffd_warped'
gridFile = f'./{name}.cgns'
alpha = 0.0
mach = 0.8
Re = 50000
Re_L = 1.0
temp = 540
arearef = 2.0
chordref = 1.0
saconstsm = [0.41, 0.1355, 0.622, 0.66666666667, 7.1, 0.3, 2.0]
saconsts = saconstsm + [1.0, 2.0, 1.2, 0.5, 2.0]
#saconsts = [0.41, 0.1355, 0.622, 0.66666666667, 7.1, 0.3, 2.0, 1.0, 2.0, 1.2, 0.5, 2.0]
gridSol = f'{name}_{saconstsm}_sol'

aeroOptions = {
# Common Parameters
'gridFile':gridFile,
'outputDirectory':outputDirectory,
'writeTecplotSurfaceSolution':True,

# Physics Parameters
'equationType':'RANS',
'turbulenceModel':'SA',
'turbulenceProduction':'vorticity',
'useft2SA':True,
'eddyVisInfRatio':3.0,
'SAConsts':saconsts,

# Common Parameters
'MGCycle':'sg',
'nCycles':100000,
'monitorvariables':['resrho','resmom','cl','cd','resturb'],
'useNKSolver':True,
'NKSwitchTol':1e-16,
'NKSubspaceSize':200,
'NKLS':'none',
'useANKSolver':True,
'ANKCoupledSwitchTol':1e-5,
'ANKConstCFLStep':0.4,
'ANKCFLLimit':1000000000.0,
'L2Convergence':1e-13,

# Output
'volumeVariables':['eddy','eddyratio','dist']
}

# Aerodynamic problem description
ap = AeroProblem(name=gridSol, alpha=alpha, mach=mach, reynolds=Re, reynoldsLength=Re_L, T=temp, areaRef=arearef, chordRef=chordref, 
evalFuncs=['cd'])

# Create solver
CFDSolver = ADFLOW(options=aeroOptions)

# Solve and evaluate functions
funcs = {}
CFDSolver(ap)
CFDSolver.evalFunctions(ap, funcs)

# Print the evaluated functions
if MPI.COMM_WORLD.rank == 0:
    print(funcs)