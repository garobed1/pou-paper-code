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
from plate_comp_opts import aeroOptions, warpOptions, optOptions, uqOptions


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Script to run plate optimization
ooptions = optOptions
uoptions = uqOptions

fname = ooptions['prob_name']+'.txt'
resfile = open(fname, 'w')
nRuns = ooptions['nRuns']
sl = uoptions['ParamSlice']
if uoptions['mode'] == 'SC': #run only once for SC
    nRuns = 1

# Print options file
if rank == 0:
    log = open("./plate_comp_opts.py", "r").read()
    print(log, file = resfile)
    print('Number of code runs = ', nRuns, file = resfile)

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
        prob.model.add_subsystem('bump_plate', pcf.PlateComponentMFMC(), promotes_inputs=['a'])
    elif uqOptions['mode'] == 'SC':
        prob.model.add_subsystem('bump_plate', pcs.PlateComponentSC(), promotes_inputs=['a'])
    else:
        prob.model.add_subsystem('bump_plate', pcl.PlateComponentLHS(), promotes_inputs=['a'])

    # setup the optimization
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['debug_print'] = ['desvars','objs']
    prob.driver.options['tol'] = 1e-6
    prob.driver.options['maxiter'] = 100

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

    # recording?
    # if ooptions['run_once'] == False:
    #     prob.driver.recording_options['includes'] = ['*']
    #     prob.driver.recording_options['record_objectives'] = True
    #     prob.driver.recording_options['record_constraints'] = True
    #     prob.driver.recording_options['record_desvars'] = True
    #     prob.driver.recording_options['record_inputs'] = True
    #     prob.driver.recording_options['record_outputs'] = True
    #     prob.driver.recording_options['record_residuals'] = True
    #     recorder = om.SqliteRecorder(ooptions['prob_name']+'.sql')
    #     #recorder._parallel = True
    #     prob.driver.add_recorder(recorder)


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
