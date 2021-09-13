from mpi4py import MPI
from baseclasses import *
from adflow import ADFLOW

aeroOptions = { #ADflow aero solver options
    # Common Parameters
    'gridFile':f'hump2newtop_noplenumZ103x28.cgns',
    'outputDirectory':'./results/',
    'writeTecplotSurfaceSolution':False,
    'writeSurfaceSolution':True,
    'writeVolumeSolution':True,
    
    # Physics Parameters
    'equationType':'RANS',
    'turbulenceModel':'SA',
    'turbulenceProduction':'vorticity',
    'useft2SA':True,
    'eddyVisInfRatio':3.0,
    
    # Common Parameters
    'MGCycle':'sg',
    'nCycles':1000000,
    'monitorvariables':['resrho','resmom','cd','resturb'],
    'useNKSolver':True,
    'NKSwitchTol':1e-3,
    'NKSubspaceSize':200,
    'NKLS':'none',
    'useANKSolver':True,
    'ANKCoupledSwitchTol':1e-2,
    'ANKConstCFLStep':0.4,
    'ANKCFLLimit':1000000000.0,
    'L2Convergence':1e-12,
    
    # Design options
    'meshSurfaceFamily':'allSurfaces',
    'designSurfaceFamily':'allSurfaces',

    # Adjoint options
    'adjointL2Convergence': 1e-06,

    # Output
    'volumeVariables':['eddyratio','mach','cp'],
    'surfaceVariables':['yplus','cf','cp'],
    'printIterations':True,
    'printTiming':False,
    'printWarnings':False,
    'setMonitor':False
    }

probName = 'hump_valid_3'
        
# flow characteristics
alpha = 0.0
mach = 0.1
Re = 936000
Re_L = 1.0
temp = 537
arearef = 1.0
chordref = 1.0

gridSol = f'{probName}_sol'

# Aerodynamic problem description
ap = AeroProblem(name=gridSol, alpha=alpha, mach=mach, reynolds=Re, reynoldsLength=Re_L, T=temp, areaRef=arearef, chordRef=chordref, 
    evalFuncs=['cd'])

# Create solver
CFDSolver = ADFLOW(options=aeroOptions, comm=MPI.COMM_WORLD)
ap.setBCVar("PressureStagnation", 1.007*ap.__dict__["P"], 'inflow')
ap.setBCVar("TemperatureStagnation", 1.002*ap.__dict__["T"], 'inflow')
ap.setBCVar("VelocityUnitVectorX", 1.0, 'inflow')
ap.setBCVar("VelocityUnitVectorY", 0.0, 'inflow')
ap.setBCVar("VelocityUnitVectorY", 0.0, 'inflow')
ap.setBCVar("Pressure", 0.99962*ap.__dict__["P"], 'outflow')
#import pdb; pdb.set_trace()

funcs = {}
CFDSolver(ap)
CFDSolver.evalFunctions(ap, funcs)

str = gridSol + '_cd'
Cd = funcs[str]

print('Cd = %.15g' % Cd)
