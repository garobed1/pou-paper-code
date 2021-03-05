# ======================================================================
#         Import modules
# ======================================================================
import numpy
from mpi4py import MPI
from baseclasses import *
from adflow import ADFLOW
#saconsts = [0.41, 0.1355, 0.622, 0.66666666667, 7.1, 0.3, 2.0, 1.0, 2.0, 1.2, 0.5, 2.0]
# ======================================================================

N = 5 #number of factors
M = 3 #number of points per factor

#bounds of each factor
scmlb = [0.31, 0.1155, 0.422, 0.5, 6.1]
scmub = [0.51, 0.1555, 0.822, 0.833333333333, 8.1]
diff = numpy.subtract(scmub, scmlb)
index = range(0, M-1)

#common settings
outputDirectory = './'
name = 'bump_69x49_0.3_0.05_0.02'
gridFile = f'./{name}.cgns'
alpha = 0.0
mach = 0.8
Re = 50000
Re_L = 1.0
temp = 540
arearef = 2.0
chordref = 1.0

results = []

#loop for each realization
for i in index: 
    for j in index: 
        for k in index: 
            for l in index: 
                for m in index:

                    scale = [i/M, j/M, k/M, l/M, m/M]

                    saconstsm = scmlb + numpy.multiply(scale, diff)
                    saconsts = [*saconstsm, *[0.3, 2.0, 1.0, 2.0, 1.2, 0.5, 2.0]]
                    gridSol = f'bump_{saconstsm}_sol'

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
                    evalFuncs=['cl','cd'])

                    # Create solver
                    CFDSolver = ADFLOW(options=aeroOptions)

                    # Solve and evaluate functions
                    funcs = {}
                    CFDSolver(ap)
                    CFDSolver.evalFunctions(ap, funcs)

                    results.append([saconstsm, funcs])

                    # Print the evaluated functions
                    #if MPI.COMM_WORLD.rank == 0:
                    #    print(funcs)