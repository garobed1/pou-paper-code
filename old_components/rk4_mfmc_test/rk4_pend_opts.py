import math
probName = 'rk4_pend_MC_Pcomp_5'

uqOptions = { #general UQ parameters
    'prob_name':probName,
    'mode':'MC', # MC: Normal Monte Carlo with LHS points
                 # MLMC: Multi-Level Monte Carlo with LHS points
    'NS':1000, #number of sample points
    'NS0':1000, #start up sample number for multi-fidelity
    'use-predetermined-samples':False, #input N1 at each level instead of running MLMC
    'MCTimeBudget':True, #
    'predet-N1':[100,100,100], #user-determined N1
    'levels':3, #number of levels
    'vartol': 2e-5, #ML variance tolerance for convergence
    'L':2.0,
    'm':1.0,
    'g':10.0,
    'crange':[0.8, 1.0],
    'Tf':5.0,
    'u0':[math.pi/8, 0.],
    'P':5. #computational budget in seconds

}

