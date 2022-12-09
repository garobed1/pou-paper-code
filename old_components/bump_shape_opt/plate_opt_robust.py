import math
import os, sys
import time
import numpy as np
import openmdao.api as om
from mpi4py import MPI
import plate_comp as pc
import plate_comp_lhs as pcl
import plate_comp_mfmc as pcf
import plate_comp_sc as pcs

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Get options from the python file specified in a command line argument, e.g. (plate_opts.py)
# This file needs 'aeroOptions', 'warpOptions', 'optOptions', and 'uqOptions'
if len(sys.argv) <= 1:
    exit("Need to supply an options file argument")
plate_comp_opts = __import__(sys.argv[1].replace('.py', ''))
optOptions = plate_comp_opts.optOptions 
aeroOptions = plate_comp_opts.aeroOptions 
warpOptions = plate_comp_opts.warpOptions 
uqOptions = plate_comp_opts.uqOptions 

# Script to run plate optimization
ooptions = optOptions
uoptions = uqOptions

# Print options file
if rank == 0:
    fname = ooptions['prob_name']+'.txt'
    resfile = open(fname, 'w')
    log = open("./"+sys.argv[1], "r").read()
    print(log, file = resfile)

nRuns = ooptions['nRuns']
sl = uoptions['ParamSlice']
if uoptions['mode'] == 'SC': #run only once for SC
    nRuns = 1

meanm = []
meanv = []
means = []
meanr = []
meanwt = 0.
meanpt = 0.
for i in range(nRuns):
    #sys.stdout = open(os.devnull, "w")
    prob = om.Problem()
    if uqOptions['mode'] == 'MFMC':
        prob.model.add_subsystem('bump_plate', pcf.PlateComponentMFMC(plate_comp_opts), promotes_inputs=['a'])
    elif uqOptions['mode'] == 'SC':
        prob.model.add_subsystem('bump_plate', pcs.PlateComponentSC(plate_comp_opts), promotes_inputs=['a'])
    else:
        prob.model.add_subsystem('bump_plate', pcl.PlateComponentLHS(plate_comp_opts), promotes_inputs=['a'])

    # setup the optimization
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['debug_print'] = ['desvars','objs']
    prob.driver.options['tol'] = 1e-9
    prob.driver.options['maxiter'] = 200

    # design vars and objectives
    NV = 2*math.trunc(((1.0 - optOptions['DVFraction'])*optOptions['NX']))
    ub = optOptions['DVUpperBound']*np.ones(NV)
    lb = optOptions['DVLowerBound']*np.zeros(NV)
    prob.model.add_design_var('a', lower=lb, upper=ub)
    prob.model.add_objective('bump_plate.Cd_r', scaler=1)
    lbc = ooptions['DCMinThick']
    lba = ooptions['DCMinArea']
    if ooptions['constrain_opt']:
        if ooptions['use_area_con']:
            prob.model.add_constraint('bump_plate.SA', lower = lba, scaler=1)
        else:
            prob.model.add_constraint('bump_plate.TC', lower = lbc, scaler=1)

    prob.model.add_constraint('bump_plate.EQ', equals = 0.0, scaler=1)

    prob.setup()

    wc0 = time.perf_counter()
    pc0 = time.process_time()

    if ooptions['check_partials']:
        prob.check_partials(method = 'fd')
    elif ooptions['run_once']:
        prob.run_model()
    else:
        prob.run_driver()

    wc1 = time.perf_counter()
    pc1 = time.process_time()
    wct = wc1 - wc0
    pct = pc1 - pc0

    #sys.stdout = sys.__stdout__

    prob.model.list_inputs(values = False, hierarchical=False)
    prob.model.list_outputs(values = False, hierarchical=False)

    # minimum value
    if rank == 0:
        print('run = ', i, file = resfile)
        print('WC time = %.15g' % wct, file = resfile)
        print('PC time = %.15g' % pct, file = resfile)
        print('E = %.15g' % prob['bump_plate.Cd_m'], file = resfile)
        print('V = %.15g' % prob['bump_plate.Cd_v'], file = resfile)
        print('S = %.15g' % prob['bump_plate.Cd_s'], file = resfile)
        print('E + rhoS = %.15g' % prob['bump_plate.Cd_r'], file = resfile)
        if ooptions['constrain_opt']:
            if ooptions['use_area_con']:
                print('SA = ', prob['bump_plate.SA'], file = resfile)
            else:
                print('TC = ', prob['bump_plate.TC'], file = resfile)
        print('Sol = ',  ','.join(map(str, prob['a'])) , file = resfile)

        #if uoptions['mode'] == 'MLMC':
        print('N1 = ', prob['bump_plate.N1'], file = resfile)
        if uoptions['mode'] == 'MFMC':   
            print('a1 = ', prob['bump_plate.a1'], file = resfile)
        print('Pr = ', prob['bump_plate.Pr'], file = resfile)
        if sl is not None and uoptions['mode'] == 'SC':
            print('Param = ', sl, file = resfile)
            print('Order = ', prob['bump_plate.OR'], file = resfile)
            print('Slice = ', prob['bump_plate.SL'], file = resfile)

    meanm.append(prob['bump_plate.Cd_m'][0])
    meanv.append(prob['bump_plate.Cd_v'][0])
    means.append(prob['bump_plate.Cd_s'][0])
    meanr.append(prob['bump_plate.Cd_r'][0])
    meanwt += wct
    meanpt += pct

#print run result lists to the end of the file
meanwt /= nRuns
meanpt /= nRuns
if rank == 0:
    print('VALUES OF ALL RUNS', file = resfile)
    print('WC time = %.15g' % meanwt, file = resfile)
    print('PC time = %.15g' % meanpt, file = resfile)
    print('E = ', meanm, file = resfile)
    print('V = ', meanv, file = resfile)
    print('S = ', means, file = resfile)
    print('E + rhoS = ', meanr, file = resfile)
