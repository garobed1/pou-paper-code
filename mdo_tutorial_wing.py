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
gridFile = './grid_struct_35x25_vol_mod.cgns'
alpha = 0.0
mach = 0.2
Re = 5000000
Re_L = 1.0
temp = 540
arearef = 1.0
chordref = 1.0
altitude = 10000
name = 'fc'

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

# Common Parameters
'MGCycle':'sg',
'nCycles':10000,
'monitorvariables':['resrho','resmom','cl','cd','resturb'],
'useNKSolver':True,
'NKSwitchTol':1e-05,
'NKSubspaceSize':200,
'useANKSolver':True,
'ANKCoupledSwitchTol':1.0,

# Output
'volumeVariables':['eddy','eddyratio','dist']
}

# Aerodynamic problem description
ap = AeroProblem(name=name, alpha=alpha, mach=mach, reynolds=Re, reynoldsLength=Re_L, T=temp, areaRef=arearef, chordRef=chordref, 
evalFuncs=['cl','cd'])

# Create solver
CFDSolver = ADFLOW(options=aeroOptions)

# Solve and evaluate functions
funcs = {}
CFDSolver(ap)
CFDSolver.evalFunctions(ap, funcs)

# Print the evaluated functions
if MPI.COMM_WORLD.rank == 0:
    print(funcs)