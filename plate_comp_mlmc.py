import numpy
import sys, os
import openmdao.api as om
import plate_ffd as pf
import math
import plate_sa_lhs
import subprocess
import time
from mpi4py import MPI
from idwarp import USMesh
from baseclasses import *
from adflow import ADFLOW
from pygeo import DVGeometry, DVConstraints
from plate_comp_opts import aeroOptions, warpOptions, optOptions, uqOptions

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

class PlateComponentMLMC(om.ExplicitComponent):
    """Robust Bump Flow Problem, with LHS Multi Level Monte Carlo Samples"""
    def initialize(self):
        # Need to modify this dictionary when we change the SA constants
        #import pdb; pdb.set_trace()
        sys.stdout = open(os.devnull, "w")
        self.aoptions = aeroOptions
        self.woptions = warpOptions
        self.ooptions = optOptions
        self.uoptions = uqOptions

        # Generate FFD and DVs
        if rank == 0:
            rank0dvg = pf.createFFD()
        else:
            rank0dvg = None
        self.DVGeo = comm.bcast(rank0dvg, root=0)

        # Get a full list of every mesh name we have, assume they are ordered properly by level
        self.meshnames = self.uoptions['gridFileLevels']
        self.Lmax = len(self.meshnames) #as many levels as we have meshes

        if self.Lmax < 3:
            raise ValueError('Not enough meshes available, ' + self.Lmax + ' < 3')


        # Spalart Allmaras model constants, to be changed in UQ (4 for now)
        saconstsm = [0.41, 0.1355, 0.622, 0.66666666667]
        self.saconstsb = [7.1, 0.3, 2.0, 1.0, 2.0, 1.2, 0.5, 2.0]
        self.saconsts = saconstsm + self.saconstsb
        self.aoptions['SAConsts'] = self.saconsts
        #self.gridSol = f'{meshname}_{saconstsm}_sol'
        solname = self.ooptions['prob_name']
        self.gridSol = f'{solname}_sol'

        self.cases = []
        self.nsp = []   #keep track per level
        self.samplep = []
        self.naddedtot = []
        for i in range(self.Lmax):
            self.naddedtot.append(0)

        self.aps = []
        self.solvers = []
        self.meshes = []

        self.NS0 = self.uoptions['NS0']
        self.current_samples = self.NS0*self.Lmax
        self.N1 = None
##########
        # call distribute samples here?
        if not self.uoptions['use-predetermined-samples']:
            self.MLMC()
        else:
            self.N1 = self.uoptions['predet-N1']

        self.dist_samples()

##########

        # Set constraints, should only need one of those solvers, the meshes are all the same
        if rank == 0:
            self.DVCon = DVConstraints()
            self.DVCon2 = DVConstraints() 
            self.DVCon.setDVGeo(self.solvers[self.Lmax-1][0].DVGeo.getFlattenedChildren()[1])
            self.DVCon2.setDVGeo(self.solvers[self.Lmax-1][0].DVGeo)

            self.DVCon.setSurface(self.solvers[self.Lmax-1][0].getTriangulatedMeshSurface(groupName='allSurfaces'))
            # set extra group for surface area condition
            self.DVCon2.setSurface(self.solvers[self.Lmax-1][0].getTriangulatedMeshSurface(), name='wall')

            # DV should be same into page (not doing anything right now)
            #import pdb; pdb.set_trace()
            lIndex = self.solvers[self.Lmax-1][0].DVGeo.getFlattenedChildren()[1].getLocalIndex(0)
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
            ra = self.ooptions['bumpBounds']
            lim = self.ooptions['DCMinArea']
            span = numpy.linspace(0, 1, nXc)
            xc = span * (ra[1] - ra[0]) + ra[0]
            #ind = range(int(nXc/2) - int(self.NC/2), int(nXc/2) + int(self.NC/2)))
            ind = [int(nXc/2) - int(tcf*self.NC/2), int(nXc/2) + int(tcf*self.NC/2)]
            ptList = numpy.zeros([2, 3])
            ptList[:,0] = xc[ind]
            ptList[:,1] = 0.5
            ptList[:,2] = 0.5

            if self.ooptions['use_area_con']:
                self.DVCon2.addSurfaceAreaConstraint(lower=lim, upper=10., name='sas', surfaceName='wall')
            else:
                self.DVCon2.addThicknessConstraints1D(ptList, self.NC, [0,0,1], lower=0.5, upper=ub, name='tcs')

        dummy = rank
        dsum = comm.allgather(dummy)

        sys.stdout = sys.__stdout__

    def setup(self):
        #initialize shape and set deformation points as inputs
        a_init = self.DVGeo.getValues()
        a_init['pnts'][:] = self.ooptions['DVInit']
        # mult = numpy.linspace(1.0,1.5,num=int(0.5*len(a_init['pnts'])))
        # mult = numpy.concatenate((mult, mult))
        #if self.ooptions['run_once']:
        #    a_init['pnts'] = self.ooptions['ro_shape']
        #a_init['pnts'] = numpy.multiply(mult, a_init['pnts'])
        self.add_input('a', a_init['pnts'], desc="Bump Shape Control Points")
        #self.add_input('a', 0.2, desc="Bump Shape Control Points")

        if self.ooptions['use_area_con']:
            self.add_output('SA', 1.0, desc='Surface Area Constraint')
        else:
            self.add_output('TC', numpy.zeros(self.NC), desc='Thickness Constraints')
        self.add_output('Cd_m', 0.0, desc="Mean Drag Coefficient")
        self.add_output('Cd_v', 0.0, desc="Variance Drag Coefficient")
        self.add_output('Cd_r', 0.0, desc="Robust Drag Objective")
        self.add_output('EQ', numpy.zeros(int(len(a_init['pnts'])/2)), desc="Control Point Symmetry")

        self.add_output('N1', self.N1, desc="MLMC Samples at each level")
    
    def setup_partials(self):
        self.declare_partials('Cd_m','a', method='exact')
        self.declare_partials('Cd_v','a', method='exact')
        self.declare_partials('Cd_r','a', method='exact')
        if self.ooptions['use_area_con']:
            self.declare_partials('SA','a', method='exact')
        else:
            self.declare_partials('TC','a', method='exact')
        self.declare_partials('EQ','a', method='exact')
    
    def compute(self, inputs, outputs):
        # run the bump shape model
        #import pdb; pdb.set_trace()
        
        dvdict = {'pnts':inputs['a']}
        ntot = self.current_samples
        nslp = []
        nslt = []
        for k in range(self.Lmax): 
            nslp.append(len(self.cases[k][rank]))
            nslt.append(sum([len(self.cases[k][x]) for x in range(size)]))
        funcs = {}
        sump = []
        musp = []
        sum1 = []
        mus = []
        sumpm = []
        muspm = []
        summ = []
        musm = []
        E = numpy.zeros(self.Lmax)
        V = numpy.zeros(self.Lmax)
        for k in range(self.Lmax): 
            sump.append(0.0)
            musp.append(numpy.zeros(nslp[k]))
            sumpm.append(0.)
            muspm.append(numpy.zeros(nslp[k]))
            for i in range(nslp[k]):
                saconstsm = self.samplep[k][i].tolist()
                self.saconsts = saconstsm + self.saconstsb
                self.solvers[k][i].setOption('SAConsts', self.saconsts)
                self.solvers[k][i].DVGeo.setDesignVars(dvdict)
                self.aps[k][i].setDesignVars(dvdict)
                self.solvers[k][i](self.aps[k][i])
                self.solvers[k][i].evalFunctions(self.aps[k][i], funcs)
                astr = self.gridSol + "_" + str(self.cases[k][rank][i]) +"_cd"
                musp[k][i] = funcs[astr]
                sump[k] += funcs[astr]     
                if k > 0:
                    self.solvers[k][i+nslp[k]].setOption('SAConsts', self.saconsts)
                    self.solvers[k][i+nslp[k]].DVGeo.setDesignVars(dvdict)
                    self.aps[k][i+nslp[k]].setDesignVars(dvdict)
                    self.solvers[k][i+nslp[k]](self.aps[k][i+nslp[k]])
                    self.solvers[k][i+nslp[k]].evalFunctions(self.aps[k][i+nslp[k]], funcs)
                    astr = self.gridSol + "_" + str(self.cases[k][rank][i]) +"_m_cd"
                    muspm[k][i] = -funcs[astr]
                    sumpm[k] += -funcs[astr]       
                    
        # compute mean and variance
        for k in range(self.Lmax):
            sum1.append(comm.allreduce(sump[k]))
            mus.append(comm.allgather(musp[k]))
            summ.append(comm.allreduce(sumpm[k]))
            musm.append(comm.allgather(muspm[k]))
        #import pdb; pdb.set_trace()
            E[k] = (sum1[k]+summ[k])/nslt[k]
            sum2 = 0.
            for i in range(len(mus[k])): #loop over processors
                for j in range(len(mus[k][i])): #loop over samples on processors
                    if k > 0:
                        sum2 += ((mus[k][i][j] + musm[k][i][j])-E[k])**2
                    else:
                        sum2 += (mus[k][i][j]-E[k])**2
            V[k] = sum2/nslt[k]

        if rank == 0:
            self.DVCon.evalFunctions(funcs, includeLinear=True)
            self.DVCon2.evalFunctions(funcs, includeLinear=True)
            eq0 = funcs['eqs']
            sa0 = funcs['sas']
        else:
            eq0 = None
            sa0 = None
        eq = comm.bcast(eq0, root=0)
        sa = comm.bcast(sa0, root=0)

        outputs['Cd_m'] = sum(E)
        outputs['Cd_v'] = V[0] #by assumption
        rho = self.uoptions['rho']
        outputs['Cd_r'] = sum(E) + rho*math.sqrt(V[0])
        if self.ooptions['use_area_con']:
            outputs['SA'] = sa
        else:
            outputs['TC'] = funcs['tcs']
        
        outputs['EQ'] = eq
        outputs['N1'] = self.N1
        #outputs['Cd'] = inputs['a']*inputs['a']

    def compute_partials(self, inputs, J):

        dvdict = {'pnts':inputs['a']}
        funcs = {}
        funcSens = {}
        nslp = []
        nslt = []
        for k in range(self.Lmax): 
            nslp.append(len(self.cases[k][rank]))
            nslt.append(sum([len(self.cases[k][x]) for x in range(size)]))
        sump = []
        musp = []
        sum1 = []
        sum2 = []
        mus = []
        
        sumpm = []
        muspm = []
        summ = []
        musm = []

        pmup = []
        psump = [] #numpy.zeros(len(inputs['a']))
        pmu = []
        psum1 = []
        
        pmupm = []
        psumpm = [] #numpy.zeros(len(inputs['a']))
        pmum = []
        psumm = []

        E = numpy.zeros(self.Lmax)
        V = numpy.zeros(self.Lmax)
        #import pdb; pdb.set_trace()
        for k in range(self.Lmax): 
            sump.append(0.0)
            musp.append(numpy.zeros(nslp[k]))
            sumpm.append(0.)
            muspm.append(numpy.zeros(nslp[k]))
            pmup.append([])
            psump.append(numpy.zeros(len(inputs['a'])))
            pmupm.append([])
            psumpm.append(numpy.zeros(len(inputs['a'])))
            for i in range(nslp[k]):
                saconstsm = self.samplep[k][i].tolist()
                self.saconsts = saconstsm + self.saconstsb
                self.solvers[k][i].setOption('SAConsts', self.saconsts)
                self.solvers[k][i].DVGeo.setDesignVars(dvdict)
                self.aps[k][i].setDesignVars(dvdict)
                self.solvers[k][i](self.aps[k][i])
                self.solvers[k][i].evalFunctions(self.aps[k][i], funcs)
                self.solvers[k][i].evalFunctionsSens(self.aps[k][i], funcSens, ['cd'])
                astr = self.gridSol + "_" + str(self.cases[k][rank][i]) +"_cd"
                arr = [x*(1./nslt[k]) for x in funcSens[astr]['pnts']]
                musp[k][i] = funcs[astr]
                sump[k] += funcs[astr]
                pmup[k].append(arr[0])
                psump[k] += arr[0]    
                if k > 0:
                    self.solvers[k][i+nslp[k]].setOption('SAConsts', self.saconsts)
                    self.solvers[k][i+nslp[k]].DVGeo.setDesignVars(dvdict)
                    #import pdb; pdb.set_trace()
                    self.aps[k][i+nslp[k]].setDesignVars(dvdict)
                    self.solvers[k][i+nslp[k]](self.aps[k][i+nslp[k]])
                    self.solvers[k][i+nslp[k]].evalFunctions(self.aps[k][i+nslp[k]], funcs)
                    self.solvers[k][i+nslp[k]].evalFunctionsSens(self.aps[k][i+nslp[k]], funcSens, ['cd'])
                    astr = self.gridSol + "_" + str(self.cases[k][rank][i]) +"_m_cd"
                    arr = [x*(1./nslt[k]) for x in funcSens[astr]['pnts']]
                    muspm[k][i] = -funcs[astr]
                    sumpm[k] += -funcs[astr]
                    pmupm[k].append(-arr[0])
                    psumpm[k] -= arr[0]   


        # compute mean and variance
        psum2 = numpy.zeros(len(inputs['a']))
        for k in range(self.Lmax):
            sum1.append(comm.allreduce(sump[k]))
            mus.append(comm.allgather(musp[k]))
            summ.append(comm.allreduce(sumpm[k]))
            musm.append(comm.allgather(muspm[k]))
            psum1.append(comm.allreduce(psump[k]))
            pmu.append(comm.allgather(pmup[k]))
            psumm.append(comm.allreduce(psumpm[k]))
            pmum.append(comm.allgather(pmupm[k]))

            E[k] = (sum1[k]+summ[k])/nslt[k]
        sum2 = 0.
        for i in range(len(mus[0])): #loop over processors
            for j in range(len(mus[0][i])): #loop over samples on processors
                #import pdb; pdb.set_trace()
                sum2 += (mus[0][i][j]-E[0])**2
                temp = pmu[0][i][j]*nslt[0] - psum1[0]
                arr2 = [x*2*(mus[0][i][j]-E[0])/nslt[0] for x in temp]
                psum2 += arr2
        V = sum2/nslt[0]

        # compute variance sensitivity

        #import pdb; pdb.set_trace()
        if rank == 0:
            self.DVCon.evalFunctionsSens(funcSens, includeLinear=True)
            self.DVCon2.evalFunctionsSens(funcSens, includeLinear=True)
            eq0p = funcSens['eqs']['pnts']
            sa0p = funcSens['sas']['pnts']
        else:
            eq0p = None
            sa0p = None
        eqp = comm.bcast(eq0p, root=0)
        sap = comm.bcast(sa0p, root=0)
        
 
        J['Cd_m','a'] = sum(psum1) + sum(psumm)
        J['Cd_v','a'] = psum2
        rho = self.uoptions['rho']
        J['Cd_r','a'] = sum(psum1) + sum(psumm) + rho*(1./(2*math.sqrt(V)))*psum2
        #import pdb; pdb.set_trace()
        if self.ooptions['use_area_con']:
            J['SA','a'] = sap
        else:
            J['TC','a'] = funcSens['tcs']['pnts']
        J['EQ','a'] = eqp

       #J['Cd','a'][0] = 2*inputs['a']

    def MLMC(self):
        # Use an MLMC algorithm to determine an optimal sample distribution between existing mesh levels
        # We do this once before optimization, then compute statistics with the same set of samples at every iteration

        # start with initial samples
        # Get a set of UQ sample points (LHS), enough for each level at the start
        #sys.stdout = open(os.devnull, "w")
        
        # flow characteristics
        alpha = 0.0
        mach = self.ooptions['mach']#0.95
        Re = self.ooptions['Re']#50000
        Re_L = 1.0
        tempR = 540
        arearef = 2.0
        chordref = 1.0
        a_init = self.DVGeo.getValues()
        a_init['pnts'][:] = self.ooptions['DVInit']
        
        self.current_samples = self.NS0*self.Lmax
        if rank == 0:
            rank0sam = plate_sa_lhs.genLHS(s=self.current_samples)
        else:
            rank0sam = None
        self.sample = comm.bcast(rank0sam, root=0)

        # Scatter samples on each level, multi-point parallelism
        
        for i in range(self.Lmax):
            self.cases.append(divide_cases(self.NS0, size)) 
            for j in range(len(self.cases[i])):
                for k in range(len(self.cases[i][j])):
                    self.cases[i][j][k] += i*self.NS0
            #self.nsp.append(len(self.cases[i][rank]))#int(ns/size) # samples per processor
            self.samplep.append(self.sample[self.cases[i][rank]])
        #import pdb; pdb.set_trace()
        #self.samplep = self.sample[self.cases[rank]]#self.sample[(rank*self.nsp):(rank*self.nsp+(self.nsp))] #shouldn't really need to "scatter" per se
        #import pdb; pdb.set_trace()
        # for i in range(self.Lmax):
        #     assert len(self.samplep[i]) == self.nsp[i]
        
        # Actually create solvers (and aeroproblems?) (and mesh?) now
        nslp = []
        nslt = []
        for k in range(self.Lmax): 
            alist = []
            slist = []
            mlist = []
            alist2 = []
            slist2 = []
            mlist2 = []
            nslp.append(len(self.cases[k][rank]))
            nslt.append(sum([len(self.cases[k][x]) for x in range(size)]))
            for i in range(nslp[k]):
                namestr = self.gridSol + "_" + str(self.cases[k][rank][i])

                # create meshes
                leveloptions = self.woptions
                leveloptions['gridFile'] = self.meshnames[k] 
                mlist.append(USMesh(options=leveloptions, comm=MPI.COMM_SELF))

                # create aeroproblems 
                aloptions = self.aoptions
                aloptions['gridFile'] = self.meshnames[k] 
                alist.append(AeroProblem(name=namestr, alpha=alpha, mach=mach, reynolds=Re, reynoldsLength=Re_L, T=tempR, areaRef=arearef, chordRef=chordref, evalFuncs=['cd']))
                time.sleep(0.1) # this solves a few problems for some reason
                # create solvers
                slist.append(ADFLOW(options=aloptions, comm=MPI.COMM_SELF))
                
                # if not self.ooptions['run_once']:
                #     saconstsm = self.samplep[i].tolist()
                # else:
                saconstsm = self.samplep[k][i].tolist()
                self.saconsts = saconstsm + self.saconstsb
                slist[i].setOption('SAConsts', self.saconsts)
                slist[i].setDVGeo(self.DVGeo)
                slist[i].setMesh(mlist[i])
                coords = slist[i].getSurfaceCoordinates(groupName=slist[i].allWallsGroup)
                slist[i].DVGeo.addPointSet(coords, 'coords')

                if k > 0: #create additional solvers at higher levels for the estimators
                     # create meshes
                    namestr = self.gridSol + "_" + str(self.cases[k][rank][i]) + "_m"
                    leveloptions = self.woptions
                    leveloptions['gridFile'] = self.meshnames[k-1] 
                    mlist2.append(USMesh(options=leveloptions, comm=MPI.COMM_SELF))
                    # create aeroproblems 
                    aloptions = self.aoptions
                    aloptions['gridFile'] = self.meshnames[k-1] 
                    alist2.append(AeroProblem(name=namestr, alpha=alpha, mach=mach, reynolds=Re, reynoldsLength=Re_L, T=tempR, areaRef=arearef, chordRef=chordref, evalFuncs=['cd']))
                    time.sleep(0.1) # this solves a few problems for some reason
                    # create solvers
                    slist2.append(ADFLOW(options=aloptions, comm=MPI.COMM_SELF))
                    slist2[i].setOption('SAConsts', self.saconsts)
                    slist2[i].setDVGeo(self.DVGeo)
                    slist2[i].setMesh(mlist2[i])
                    coords = slist[i].getSurfaceCoordinates(groupName=slist2[i].allWallsGroup)
                    slist2[i].DVGeo.addPointSet(coords, 'coords')
                
            self.aps.append(alist)
            self.solvers.append(slist)
            self.meshes.append(mlist)
            if k > 0:
                self.aps[k] += alist2
                self.solvers[k] += slist2
                self.meshes[k] += mlist2
        #import pdb; pdb.set_trace()
        # start looping over mesh levels
        L = 0
        M = 4.0 #0.5 #refinement factor?
        converged = 0
        eps = self.uoptions['vartol']
        sum1 = []
        mus = []
        sump = []
        musp = []
        sumpm = []
        muspm = []
        summ = []
        musm = []
        E = []
        V = []
        N1 = []
        while ~converged & L < self.Lmax:
            # compute start up samples to estimate variance
            dvdict = {'pnts':a_init['pnts']}
            funcs = {}
            nslp = []
            nslt = []
            for k in range(self.Lmax): 
                nslp.append(len(self.cases[k][rank]))
                nslt.append(sum([len(self.cases[k][x]) for x in range(size)]))
            sump.append(0.)
            musp.append(numpy.zeros(nslp[L]))
            sumpm.append(0.)
            muspm.append(numpy.zeros(nslp[L]))

            for i in range(nslp[L]):
                # just do this again in case
                saconstsm = self.samplep[L][i].tolist()
                self.saconsts = saconstsm + self.saconstsb
                self.solvers[L][i].setOption('SAConsts', self.saconsts)
                self.solvers[L][i].DVGeo.setDesignVars(dvdict)
                self.aps[L][i].setDesignVars(dvdict)
                self.solvers[L][i](self.aps[L][i])
                self.solvers[L][i].evalFunctions(self.aps[L][i], funcs)
                astr = self.gridSol + "_" + str(self.cases[L][rank][i]) +"_cd"
                musp[L][i] = funcs[astr]
                sump[L] += funcs[astr]   
                #import pdb; pdb.set_trace()
                if L > 0:
                    self.solvers[L][i+nslp[L]].setOption('SAConsts', self.saconsts)
                    self.solvers[L][i+nslp[L]].DVGeo.setDesignVars(dvdict)
                    self.aps[L][i+nslp[L]].setDesignVars(dvdict)
                    self.solvers[L][i+nslp[L]](self.aps[L][i+nslp[L]])
                    self.solvers[L][i+nslp[L]].evalFunctions(self.aps[L][i+nslp[L]], funcs)
                    astr = self.gridSol + "_" + str(self.cases[L][rank][i]) +"_m_cd" 
                    muspm[L][i] = -funcs[astr]
                    sumpm[L] += -funcs[astr]       
                
            # compute mean and variance estimate from start up samples
            sum1.append(comm.allreduce(sump[L]))
            mus.append(comm.allgather(musp[L]))
            summ.append(comm.allreduce(sumpm[L]))
            musm.append(comm.allgather(muspm[L]))

            #import pdb; pdb.set_trace()

            # mean at each level
            E = numpy.zeros(L+1)
            for l in range(L+1):
                E[l] = (sum1[l]+summ[l])/nslt[l]

            # variance at each level
            V = numpy.zeros(L+1)
            for l in range(L+1):
                sum2 = 0.
                for i in range(len(mus[l])): #range(size):
                    for j in range(len(mus[l][i])): #range(self.nsp):
                        if l > 0:
                            sum2 += ((mus[l][i][j] + musm[l][i][j])-E[l])**2
                        else:
                            sum2 += (mus[l][i][j]-E[l])**2
                V[l] = sum2/nslt[l]

            #import pdb; pdb.set_trace()
            # now determine the optimal number of samples at each level
            N1.append(0.)
            worksum = 0
            for l in range(L+1):
                worksum += numpy.sqrt(V[l]*(M**l))
            for l in range(L+1):
                nlf = 2*numpy.sqrt(V[l]/(M**l)) * worksum / (eps*eps)
                nlfm = max(nslt[l], math.ceil(nlf))
                N1[l] = nlfm

            # now compute and generate additional samples at each level
            # first partition samples  NEVERMIND (just do it once at each level, no need to repeat)
            # create the extra number of solvers at each (the current) level

            # need to loop everything from here on

            for l in range(L+1):
                alist = self.aps[l][0:nslp[l]]
                slist = self.solvers[l][0:nslp[l]]
                mlist = self.meshes[l][0:nslp[l]]
                if l > 0:
                    alist2 = self.aps[l][nslp[l]:]
                    slist2 = self.solvers[l][nslp[l]:]
                    mlist2 = self.meshes[l][nslp[l]:]
                
                self.naddedtot[l] = N1[l] - nslt[l]
                self.current_samples += self.naddedtot[l]
                #import pdb; pdb.set_trace()
                if rank == 0:
                    rank0sam = plate_sa_lhs.genLHS(s=self.current_samples)
                else:
                    rank0sam = None
                self.sample = comm.bcast(rank0sam, root=0)
                
                if self.naddedtot[l] > 0:
                    temp = divide_cases(self.naddedtot[l], size) 
                    for i in range(len(temp)):
                        for j in range(len(temp[i])):
                            temp[i][j] += self.current_samples - self.naddedtot[l] #self.Lmax*self.NS0 + sum(self.naddedtot[0:L])
                else:
                    temp = []
                if len(temp):
                    for ns in range(size):
                        self.cases[l][ns] += temp[ns] #append
                nslpnew = len(self.cases[l][rank])
                nsltnew = sum([len(self.cases[l][x]) for x in range(size)])
                #self.nsp[L] = len(self.cases[L][rank]) #int(ns/size) # samples per processor
                self.samplep[l] = self.sample[self.cases[l][rank]]
                
                for i in range(nslp[l], nslpnew): #need it to be just the extra cases
                    #import pdb; pdb.set_trace()
                    namestr = self.gridSol + "_" + str(self.cases[l][rank][i])
    
                    # create meshes
                    leveloptions = self.woptions
                    leveloptions['gridFile'] = self.meshnames[l] 
                    mlist.append(USMesh(options=leveloptions, comm=MPI.COMM_SELF))
    
                    # create aeroproblems
                    aloptions = self.aoptions
                    aloptions['gridFile'] = self.meshnames[l] 
                    alist.append(AeroProblem(name=namestr, alpha=alpha, mach=mach, reynolds=Re, reynoldsLength=Re_L, T=tempR, areaRef=arearef, chordRef=chordref, evalFuncs=['cd']))
                    time.sleep(0.1) # this solves a few problems for some reason
                    # create solvers
                    slist.append(ADFLOW(options=aloptions, comm=MPI.COMM_SELF))
    
                    saconstsm = self.samplep[l][i].tolist()
                    self.saconsts = saconstsm + self.saconstsb
                    slist[i].setOption('SAConsts', self.saconsts)
                    slist[i].setDVGeo(self.DVGeo)
                    slist[i].setMesh(mlist[i])
                    coords = slist[i].getSurfaceCoordinates(groupName=slist[i].allWallsGroup)
                    slist[i].DVGeo.addPointSet(coords, 'coords')
                    time.sleep(0.1)
                    if l > 0: #create additional solvers at higher levels for the estimators
                        # create meshes
                        #import pdb; pdb.set_trace()
                        namestr = self.gridSol + "_" + str(self.cases[l][rank][i]) + "_m"
                        leveloptions = self.woptions
                        leveloptions['gridFile'] = self.meshnames[l-1] 
                        mlist2.append(USMesh(options=leveloptions, comm=MPI.COMM_SELF))
                        # create aeroproblems 
                        aloptions = self.aoptions
                        aloptions['gridFile'] = self.meshnames[l-1]
                        alist2.append(AeroProblem(name=namestr, alpha=alpha, mach=mach, reynolds=Re, reynoldsLength=Re_L, T=tempR, areaRef=arearef, chordRef=chordref, evalFuncs=['cd']))
                        time.sleep(0.1) # this solves a few problems for some reason
                        # create solvers
                        slist2.append(ADFLOW(options=aloptions, comm=MPI.COMM_SELF))
                        slist2[i].setOption('SAConsts', self.saconsts)
                        slist2[i].setDVGeo(self.DVGeo)
                        slist2[i].setMesh(mlist2[i])
                        coords = slist[i].getSurfaceCoordinates(groupName=slist2[i].allWallsGroup)
                        slist2[i].DVGeo.addPointSet(coords, 'coords')
                nslp[l] = nslpnew
                nslt[l] = nsltnew
                
    
                self.aps[l] = alist
                self.solvers[l] = slist
                self.meshes[l] = mlist
                if l > 0:
                    self.aps[l] += alist2
                    self.solvers[l] += slist2
                    self.meshes[l] += mlist2
    
                # compute remaining samples
                sump[l] = 0
                sumpm[l] = 0
                musp[l] = numpy.zeros(nslp[l])
                muspm[l] = numpy.zeros(nslp[l])
                for i in range(nslp[l]):
                    # just do this again in case
                    saconstsm = self.samplep[l][i].tolist()
                    self.saconsts = saconstsm + self.saconstsb
                    self.solvers[l][i].setOption('SAConsts', self.saconsts)
                    self.solvers[l][i].DVGeo.setDesignVars(dvdict)
                    self.aps[l][i].setDesignVars(dvdict)
                    self.solvers[l][i](self.aps[l][i])
                    self.solvers[l][i].evalFunctions(self.aps[l][i], funcs)
                    astr = self.gridSol + "_" + str(self.cases[l][rank][i]) +"_cd"
                    musp[l][i] = funcs[astr]
                    sump[l] += funcs[astr]   
                    #import pdb; pdb.set_trace()
                    if l > 0:
                        self.solvers[l][i+nslp[l]].setOption('SAConsts', self.saconsts)
                        self.solvers[l][i+nslp[l]].DVGeo.setDesignVars(dvdict)
                        self.aps[l][i+nslp[l]].setDesignVars(dvdict)
                        self.solvers[l][i+nslp[l]](self.aps[l][i+nslp[l]])
                        self.solvers[l][i+nslp[l]].evalFunctions(self.aps[l][i+nslp[l]], funcs)
                        astr = self.gridSol + "_" + str(self.cases[l][rank][i]) +"_m_cd" 
                        muspm[l][i] = -funcs[astr]
                        sumpm[l] += -funcs[astr]     
    
                # compute mean and variance estimate from all samples
                sum1[l] = comm.allreduce(sump[l])
                mus[l] = comm.allgather(musp[l])
                summ[l] = comm.allreduce(sumpm[l])
                musm[l] = comm.allgather(muspm[l])
    
                # mean at each level
                E[l] = (sum1[l]+summ[l])/nslt[l]
    
                # variance at each level
                sum2 = 0.
                for i in range(len(mus[l])): #range(size):
                    for j in range(len(mus[l][i])): #range(self.nsp):
                        if l > 0:
                            sum2 += ((mus[l][i][j] + musm[l][i][j])-E[l])**2
                        else:
                            sum2 += (mus[l][i][j]-E[l])**2
                V[l] = sum2/nslt[l]
            
            
            # if L == 1:
            #     import pdb; pdb.set_trace()
            L += 1
        #import pdb; pdb.set_trace()
        #sys.stdout = sys.__stdout__
        if rank == 0:
            print("MLMC Completed, Samples per level: ", N1)
        self.N1 = N1   
        #import pdb; pdb.set_trace()
      
            # test for convergence
            # don't actually need this, won't ever end early
            # range = -1:0
            # if L > 1 & M**L >= 4:
            #     con = M**(range.*suml(2,L+1+range)./suml(1,L+1+range))
            #     converged = (max(abs(con)) < (M-1)*eps/sqrt(2))

        # once done, we have aps, solvers, meshes, which is all we need

    def dist_samples(self):
        # If we already have the number of samples, just create as many solvers as needed at each level
        # Just do this after running MLMC() anyway 

        # flow characteristics
        alpha = 0.0
        mach = self.ooptions['mach']#0.95
        Re = self.ooptions['Re']#50000
        Re_L = 1.0
        tempR = 540
        arearef = 2.0
        chordref = 1.0
        a_init = self.DVGeo.getValues()
        a_init['pnts'][:] = self.ooptions['DVInit']
        
        self.current_samples = sum(self.N1)
        if rank == 0:
            rank0sam = plate_sa_lhs.genLHS(s=self.current_samples)
        else:
            rank0sam = None
        self.sample = comm.bcast(rank0sam, root=0)
        #import pdb; pdb.set_trace()
        # Scatter samples on each level, multi-point parallelism
        self.cases = []
        self.samplep = []
        for i in range(self.Lmax):
            self.cases.append(divide_cases(self.N1[i], size)) 
            for j in range(len(self.cases[i])):
                for k in range(len(self.cases[i][j])):
                    self.cases[i][j][k] += sum(self.N1[0:i])
            #self.nsp.append(len(self.cases[i][rank]))#int(ns/size) # samples per processor
            self.samplep.append(self.sample[self.cases[i][rank]])
        
        # Actually create solvers (and aeroproblems?) (and mesh?) now
        self.aps = []
        self.solvers = []
        self.meshes = []
        nslp = []
        nslt = []
        for k in range(self.Lmax): 
            alist = []
            slist = []
            mlist = []
            alist2 = []
            slist2 = []
            mlist2 = []
            nslp.append(len(self.cases[k][rank]))
            nslt.append(sum([len(self.cases[k][x]) for x in range(size)]))
            for i in range(nslp[k]):
                namestr = self.gridSol + "_" + str(self.cases[k][rank][i])

                # create meshes
                leveloptions = self.woptions
                leveloptions['gridFile'] = self.meshnames[k] 
                mlist.append(USMesh(options=leveloptions, comm=MPI.COMM_SELF))

                # create aeroproblems 
                aloptions = self.aoptions
                aloptions['gridFile'] = self.meshnames[k] 
                alist.append(AeroProblem(name=namestr, alpha=alpha, mach=mach, reynolds=Re, reynoldsLength=Re_L, T=tempR, areaRef=arearef, chordRef=chordref, evalFuncs=['cd']))
                time.sleep(0.1) # this solves a few problems for some reason
                # create solvers
                slist.append(ADFLOW(options=aloptions, comm=MPI.COMM_SELF))
                
                # if not self.ooptions['run_once']:
                #     saconstsm = self.samplep[i].tolist()
                # else:
                saconstsm = self.samplep[k][i].tolist()
                self.saconsts = saconstsm + self.saconstsb
                slist[i].setOption('SAConsts', self.saconsts)
                slist[i].setDVGeo(self.DVGeo)
                slist[i].setMesh(mlist[i])
                coords = slist[i].getSurfaceCoordinates(groupName=slist[i].allWallsGroup)
                slist[i].DVGeo.addPointSet(coords, 'coords')

                if k > 0: #create additional solvers at higher levels for the estimators
                     # create meshes
                    namestr = self.gridSol + "_" + str(self.cases[k][rank][i]) + "_m"
                    leveloptions = self.woptions
                    leveloptions['gridFile'] = self.meshnames[k-1] 
                    mlist2.append(USMesh(options=leveloptions, comm=MPI.COMM_SELF))
                    # create aeroproblems 
                    aloptions = self.aoptions
                    aloptions['gridFile'] = self.meshnames[k-1] 
                    alist2.append(AeroProblem(name=namestr, alpha=alpha, mach=mach, reynolds=Re, reynoldsLength=Re_L, T=tempR, areaRef=arearef, chordRef=chordref, evalFuncs=['cd']))
                    time.sleep(0.1) # this solves a few problems for some reason
                    # create solvers
                    slist2.append(ADFLOW(options=aloptions, comm=MPI.COMM_SELF))
                    slist2[i].setOption('SAConsts', self.saconsts)
                    slist2[i].setDVGeo(self.DVGeo)
                    slist2[i].setMesh(mlist2[i])
                    coords = slist[i].getSurfaceCoordinates(groupName=slist2[i].allWallsGroup)
                    slist2[i].DVGeo.addPointSet(coords, 'coords')
                
            self.aps.append(alist)
            self.solvers.append(slist)
            self.meshes.append(mlist)
            if k > 0:
                self.aps[k] += alist2
                self.solvers[k] += slist2
                self.meshes[k] += mlist2
        

def divide_cases(ncases, nprocs):
    """
    From parallel OpenMDAO beam problem example

    Divide up load cases among available procs.

    Parameters
    ----------
    ncases : int
        Number of load cases.
    nprocs : int
        Number of processors.

    Returns
    -------
    list of list of int
        Integer case numbers for each proc.
    """
    data = []
    for j in range(nprocs):
        data.append([])

    wrap = 0
    for j in range(ncases):
        idx = j - wrap
        if idx >= nprocs:
            idx = 0
            wrap = j

        data[idx].append(j)

    return data

