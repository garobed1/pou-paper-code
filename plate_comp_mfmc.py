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
from itertools import combinations
#from plate_comp_opts import aeroOptions, warpOptions, optOptions, uqOptions

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

class PlateComponentMFMC(om.ExplicitComponent):
    """Robust Bump Flow Problem, with LHS Multi Level Monte Carlo Samples"""
    def __init__(self, opts):
        super().__init__()

        # Get all the options we need
        self.aoptions = opts.aeroOptions
        self.woptions = opts.warpOptions
        self.ooptions = opts.optOptions
        self.uoptions = opts.uqOptions

    def initialize(self):
        # Need to modify this dictionary when we change the SA constants
        #import pdb; pdb.set_trace()
        sys.stdout = open(os.devnull, "w")

        self.Pr = 0.
        # Generate FFD and DVs
        if rank == 0:
            rank0dvg = pf.createFFD()
        else:
            rank0dvg = None
        self.DVGeo = comm.bcast(rank0dvg, root=0)

        # Get a full list of every mesh name we have, assume they are ordered properly by level
        self.meshnames = self.uoptions['gridFileLevels']
        self.Lmax = len(self.meshnames) #as many levels as we have meshes
        self.mord = [*range(self.Lmax)]
        self.mord.reverse()

        #if self.Lmax < 3:
        #    raise ValueError('Not enough meshes available, ' + self.Lmax + ' < 3')


        # Spalart Allmaras model constants, to be changed in UQ (4 for now)
        saconstsm = [0.41, 0.1355, 0.622, 0.66666666667]
        self.saconstsb = [7.1, 0.3, 2.0, 1.0, 2.0, 1.2, 0.5, 2.0]
        self.saconsts = saconstsm + self.saconstsb
        self.aoptions['SAConsts'] = self.saconsts
        #self.gridSol = f'{meshname}_{saconstsm}_sol'
        solname = self.ooptions['prob_name']
        self.gridSol = f'{solname}_sol'

        self.cases = []
        #self.nsp = []   #keep track per level
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
        self.a1 = None
        self.P = self.uoptions['P']
##########
        # call distribute samples here?
        if not self.uoptions['use-predetermined-samples']:
            self.MFMC()
        else:
            self.N1 = self.uoptions['predet-N1']
            self.a1 = self.uoptions['predet-a1']

        self.dist_samples()

##########

        # Set constraints, should only need one of those solvers, the meshes are all the same
        if rank == 0:
            self.DVCon = DVConstraints()
            self.DVCon2 = DVConstraints() 
            self.DVCon.setDVGeo(self.solvers[0].DVGeo.getFlattenedChildren()[1])
            self.DVCon2.setDVGeo(self.solvers[0].DVGeo)

            self.DVCon.setSurface(self.solvers[0].getTriangulatedMeshSurface(groupName='allSurfaces'))
            # set extra group for surface area condition
            self.DVCon2.setSurface(self.solvers[0].getTriangulatedMeshSurface(), name='wall')

            # DV should be same into page (not doing anything right now)
            #import pdb; pdb.set_trace()
            lIndex = self.solvers[0].DVGeo.getFlattenedChildren()[1].getLocalIndex(0)
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
        self.add_output('Cd_s', 0.0, desc="Stdev Drag Coefficient")
        self.add_output('Cd_r', 0.0, desc="Robust Drag Objective")
        self.add_output('EQ', numpy.zeros(int(len(a_init['pnts'])/2)), desc="Control Point Symmetry")

        self.add_output('N1', self.N1, desc="MFMC Samples at each level")
        self.add_output('a1', self.a1, desc="MFMC Coeffs at each level")
        self.add_output('Pr', 0., desc="MFMC Samples at each level")
    
    def setup_partials(self):
        self.declare_partials('Cd_m','a', method='exact')
        self.declare_partials('Cd_v','a', method='exact')
        self.declare_partials('Cd_s','a', method='exact')
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
            if k > 0:
                muspm.append(numpy.zeros(nslp[k-1]))
            else:
                muspm.append(numpy.zeros(nslp[k])) #not used
            for i in range(nslp[k]):
                saconstsm = self.samplep[self.Lmax-1][i].tolist()
                self.saconsts = saconstsm + self.saconstsb
                self.solvers[k].setOption('SAConsts', self.saconsts)
                self.solvers[k].DVGeo.setDesignVars(dvdict)
                self.aps[k][i].setDesignVars(dvdict)
                self.solvers[k](self.aps[k][i])
                self.solvers[k].evalFunctions(self.aps[k][i], funcs)
                astr = self.gridSol + "_" + str(self.cases[k][rank][i]) +"_cd"
                musp[k][i] = funcs[astr]
                sump[k] += funcs[astr]     
                if k > 0 and i < nslp[k-1]:
                    muspm[k][i] = -funcs[astr]
                    sumpm[k] += -funcs[astr]       
                    
        # compute mean and variance
        for k in range(self.Lmax):
            sum1.append(comm.allreduce(sump[k]))
            mus.append(comm.allgather(musp[k]))
            summ.append(comm.allreduce(sumpm[k]))
            musm.append(comm.allgather(muspm[k]))
        #import pdb; pdb.set_trace()
            if k > 0:
                E[k] = (sum1[k]/nslt[k]+summ[k]/nslt[k-1])
            else:
                E[k] = (sum1[k]/nslt[k])
            sum2 = 0.
            for i in range(len(mus[k])): #loop over processors
                for j in range(len(mus[k][i])): #loop over samples on processors
                    if k > 0:
                        #sum2 = 0
                        sum2 += ((mus[k][i][j] - sum1[k]/nslt[k])**2)/nslt[k]
                        if j < len(mus[k-1][i]):
                            sum2 -= ((musm[k][i][j] - summ[k]/nslt[k-1])**2)/nslt[k-1] 
                    else:
                        sum2 += ((mus[k][i][j]-E[k])**2)/nslt[k]
            V[k] = sum2

        #import pdb; pdb.set_trace()
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

        outputs['Cd_m'] = numpy.dot(self.a1, E)
        outputs['Cd_v'] = numpy.dot(self.a1, V) #by assumption
        outputs['Cd_s'] = math.sqrt(numpy.dot(self.a1, V))
        rho = self.uoptions['rho']
        outputs['Cd_r'] = rho*numpy.dot(self.a1, E) + (1.-rho)*math.sqrt(numpy.dot(self.a1, V))
 #       import pdb; pdb.set_trace()
        if self.ooptions['use_area_con']:
            outputs['SA'] = sa
        else:
            outputs['TC'] = funcs['tcs']
        
        outputs['EQ'] = eq
        outputs['N1'] = self.N1
        outputs['a1'] = self.a1
        outputs['Pr'] = self.Pr
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
            if k > 0:
                muspm.append(numpy.zeros(nslp[k-1]))
            else:
                muspm.append(numpy.zeros(nslp[k])) #not used
            pmup.append([])
            psump.append(numpy.zeros(len(inputs['a'])))
            pmupm.append([])
            psumpm.append(numpy.zeros(len(inputs['a'])))
            for i in range(nslp[k]):
                saconstsm = self.samplep[self.Lmax-1][i].tolist()
                self.saconsts = saconstsm + self.saconstsb
                self.solvers[k].setOption('SAConsts', self.saconsts)
                self.solvers[k].DVGeo.setDesignVars(dvdict)
                self.aps[k][i].setDesignVars(dvdict)
                self.solvers[k](self.aps[k][i])
                self.solvers[k].evalFunctions(self.aps[k][i], funcs)
                self.solvers[k].evalFunctionsSens(self.aps[k][i], funcSens, ['cd'])
                astr = self.gridSol + "_" + str(self.cases[k][rank][i]) +"_cd"
                arr = [x*(1./nslt[k]) for x in funcSens[astr]['pnts']]
                musp[k][i] = funcs[astr]
                sump[k] += funcs[astr]
                pmup[k].append(arr[0])
                psump[k] += arr[0]    
                if k > 0 and i < nslp[k-1]:
                    arrm = [x*(1./nslt[k-1]) for x in funcSens[astr]['pnts']]
                    muspm[k][i] = -funcs[astr]
                    sumpm[k] += -funcs[astr]
                    pmupm[k].append(-arrm[0])
                    psumpm[k] -= arrm[0]   


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

            if k > 0:
                E[k] = (sum1[k]/nslt[k]+summ[k]/nslt[k-1])
            else:
                E[k] = (sum1[k]/nslt[k])
            sum2 = 0.
            for i in range(len(mus[k])): #loop over processors
                for j in range(len(mus[k][i])): #loop over samples on processors
                    if k > 0:
                        sum2 += ((mus[k][i][j] - sum1[k]/nslt[k])**2)/nslt[k]
                        if j < len(mus[k-1][i]):
                            sum2 -= ((musm[k][i][j] - summ[k]/nslt[k-1])**2)/nslt[k-1]
                    else:
                        sum2 += ((mus[k][i][j]-E[k])**2)/nslt[k]
            V[k] = sum2
                
        # compute variance sensitivity
        psum2 = []
        for k in range(self.Lmax):
            psum2.append(numpy.zeros(len(inputs['a'])))
            for i in range(len(mus[k])): #loop over processors
                for j in range(len(mus[k][i])): #loop over samples on processors
                    if k > 0:
                        temp = pmu[k][i][j]*nslt[k] - psum1[k]
                        arr2 = [x*2*(mus[k][i][j] - sum1[k]/nslt[k])/nslt[k] for x in temp]
                        psum2[k] += arr2
                        if j < len(mus[k-1][i]):
                            temp = -pmum[k][i][j]*nslt[k-1] - -psumm[k]
                            arr2 = [x*2*(-musm[k][i][j] - -summ[k]/nslt[k-1])/nslt[k-1] for x in temp]
                            psum2[k] -= arr2
                    else:
                        temp = pmu[k][i][j]*nslt[k] - psum1[k]
                        arr2 = [x*2*(mus[k][i][j]-E[k])/nslt[k] for x in temp]
                        psum2[k] += arr2
            psum1[k] *= self.a1[k]
            psumm[k] *= self.a1[k]
            psum2[k] *= self.a1[k]

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
        J['Cd_v','a'] = sum(psum2)
        J['Cd_s','a'] = (1./(2*math.sqrt(numpy.dot(self.a1, V))))*sum(psum2)
        rho = self.uoptions['rho']
        J['Cd_r','a'] = rho*sum(psum1) + rho*sum(psumm) + (1.-rho)*(1./(2*math.sqrt(numpy.dot(self.a1, V))))*sum(psum2)
        if self.ooptions['use_area_con']:
            J['SA','a'] = sap
        else:
            J['TC','a'] = funcSens['tcs']['pnts']
        J['EQ','a'] = eqp

       #J['Cd','a'][0] = 2*inputs['a']

    def MFMC(self):
        # Use an MFMC algorithm to determine optimal sample distribution and coefficients among mesh levels
        # We do this once before optimization, then compute statistics with the same set of samples and coeffs at every iteration

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
            rank0sam = plate_sa_lhs.genLHS(s=self.current_samples, mcs = self.uoptions['MCPure'])
        else:
            rank0sam = None
        self.sample = comm.bcast(rank0sam, root=0)

        N1 = []
        a1 = numpy.zeros(self.Lmax)
        r1 = numpy.zeros(self.Lmax)

        # Scatter samples on each level, multi-point parallelism
        for i in range(self.Lmax):
            self.cases.append(divide_cases(self.NS0, size)) 
            for j in range(len(self.cases[i])):
                for k in range(len(self.cases[i][j])):
                    self.cases[i][j][k] += i*self.NS0
            #self.nsp.append(len(self.cases[i][rank]))#int(ns/size) # samples per processor
            self.samplep.append(self.sample[self.cases[i][rank]])
        #import pdb; pdb.set_trace()
        
        # Create solvers for the preliminary data
        nslp = []
        nslt = []
        for k in range(self.Lmax): 
            alist = []
            slist = []
            mlist = []
            nslp.append(len(self.cases[k][rank]))
            nslt.append(sum([len(self.cases[k][x]) for x in range(size)]))
            for i in range(nslp[k]):
                namestr = self.gridSol + "_" + str(self.cases[k][rank][i])
                alist.append(AeroProblem(name=namestr, alpha=alpha, mach=mach, reynolds=Re, reynoldsLength=Re_L, T=tempR, areaRef=arearef, chordRef=chordref, evalFuncs=['cd']))
            # create meshes
            leveloptions = self.woptions
            leveloptions['gridFile'] = self.meshnames[self.mord[k]] 
            #import pdb; pdb.set_trace()
            mlist = USMesh(options=leveloptions, comm=MPI.COMM_SELF)

            aloptions = self.aoptions
            aloptions['gridFile'] = self.meshnames[self.mord[k]] 
            # create solvers
            time.sleep(0.1) # this solves a few problems for some reason
            slist = ADFLOW(options=aloptions, comm=MPI.COMM_SELF)
            slist.setOption('SAConsts', self.saconsts)
            slist.setDVGeo(self.DVGeo)
            slist.setMesh(mlist)
            coords = slist.getSurfaceCoordinates(groupName=slist.allWallsGroup)
            slist.DVGeo.addPointSet(coords, 'coords')
                
            self.aps.append(alist)
            self.solvers.append(slist)
            self.meshes.append(mlist)


        # Solve the preliminary samples
        
        # start looping over mesh levels
        sumt = []
        sumtp = []
        nslp = []
        nslt = []
        sum1 = []
        mus = []
        sump = []
        musp = []
        sumpm = []
        muspm = []
        summ = []
        musm = []
        Et = numpy.zeros(self.Lmax)
        E = numpy.zeros(self.Lmax)
        V = numpy.zeros(self.Lmax)
        S = numpy.zeros(self.Lmax)
        N1 = []
        for k in range(self.Lmax): 
            nslp.append(len(self.cases[k][rank]))
            nslt.append(sum([len(self.cases[k][x]) for x in range(size)]))
            dvdict = {'pnts':a_init['pnts']}
            funcs = {}
            sumtp.append(0.0)
            sump.append(0.)
            musp.append(numpy.zeros(nslp[k]))
            sumpm.append(0.)
            muspm.append(numpy.zeros(nslp[k]))
            for i in range(nslp[k]):
                saconstsm = self.samplep[0][i].tolist()
                self.saconsts = saconstsm + self.saconstsb
                self.solvers[k].setOption('SAConsts', self.saconsts)
                self.solvers[k].DVGeo.setDesignVars(dvdict)
                self.aps[k][i].setDesignVars(dvdict)
                pc0 = time.process_time()
                self.solvers[k](self.aps[k][i])
                self.solvers[k].evalFunctions(self.aps[k][i], funcs)
                pc1 = time.process_time()
                astr = self.gridSol + "_" + str(self.cases[k][rank][i]) +"_cd"
                musp[k][i] = funcs[astr]
                sump[k] += funcs[astr]
                sumtp[k] += (pc1-pc0)   
      
        # compute mean and variance estimate from start up samples
        for k in range(self.Lmax):
            sumt.append(comm.allreduce(sumtp[k]))
            sum1.append(comm.allreduce(sump[k]))
            mus.append(comm.allgather(musp[k]))
            summ.append(comm.allreduce(sumpm[k]))
            musm.append(comm.allgather(muspm[k]))
            mus[k] = numpy.concatenate(mus[k][:])
            musm[k] = numpy.concatenate(musm[k][:])
            # mean at each level
            Et[k] = sumt[k]/nslt[k]
            E[k] = (sum1[k])/nslt[k] #+summ[k]
            sum2 = 0.
            for i in range(len(mus[k])): #loop over processors
                sum2 += (mus[k][i]-E[k])**2
            V[k] = sum2/nslt[k]
            S[k] = math.sqrt(V[k])

        # compute correlation matrix and rearrange models if necessary
        ordered = False
        while not ordered:
            rho = numpy.corrcoef(mus)
            ordered = True # check if contradicted
            #tarr = rho[0,1:]
            for k in range(self.Lmax-2):
                test = rho[0,1+k]**2 - rho[0,2+k]**2
                if test < 0:
                    ordered = False
            tarr = -rho[0,:]**2

            if not ordered:
                sind = numpy.argsort(tarr)
                #import pdb; pdb.set_trace()
                self.mord[:] = [self.mord[i] for i in sind]
                E[:] = [E[i] for i in sind]
                Et[:] = [Et[i] for i in sind]
                V[:] = [V[i] for i in sind]
                S[:] = [S[i] for i in sind]
                mus[:] = [mus[i] for i in sind]
        rho0 = rho[0]

        # model selection
        combs = []
        NM = self.Lmax
        order = [*range(NM)] #not the same as mord, refers to mord's indices
        for k in range(1,NM+1):
            temp = combinations(order, k)
            for j in temp:
                if (0 in list(j)): #get only lists that contain the highest level
                    combs.append(list(j))

        # loop over candidates
        vMC = V[0]*Et[0]/10 # p = 10
        vs = vMC
        Ms = [0]
        for o in range(len(combs)):
            comb = combs[o]
            kn = len(comb)
            F = True
            if kn > 1:
                for k in range(kn-1): # check feasability
                    feas = Et[comb[k]]/Et[comb[k+1]]
                    if k == kn-2:
                        feas -= (rho0[comb[k]]**2 - rho0[comb[k+1]]**2)/(rho0[comb[k+1]]**2)
                    else:
                        feas -= (rho0[comb[k]]**2 - rho0[comb[k+1]]**2)/(rho0[comb[k+1]]**2 - rho0[comb[k+2]]**2)
                    if feas < 0:    
                        F = False
                if F == False:
                    continue
            # if feasible, check if variance is beaten
            v = 0.
            for k in range(kn):
                if k == kn-1:
                    v += math.sqrt(Et[comb[k]]*(rho0[comb[k]]**2))
                else:
                    v += math.sqrt(Et[comb[k]]*(rho0[comb[k]]**2 - rho0[comb[k+1]]**2))
            v = v**2
            v *= V[0]/10
            if v < vs:
                Ms = comb
                vs = v

        # take Ms indices of mord as the new mord
        self.mord = [self.mord[i] for i in Ms]
        self.Lmax = len(self.mord)

        # now compute N1 and a1 using sigma, rho, w, and p
        a1 = numpy.zeros(self.Lmax)
        r1 = numpy.zeros(self.Lmax)
        Et = [Et[i] for i in Ms]
        E = [E[i] for i in Ms]
        V = [V[i] for i in Ms]
        S = [S[i] for i in Ms]
        for k in range(self.Lmax):
            a1[k] = S[0]*rho0[k]/S[k]
            
            if k == 0:
                r1[k] = 1
            elif k == self.Lmax-1:
                work = Et[0]*(rho0[k]**2)
                work /= Et[k]*(1 - rho0[1]**2)
                r1[k] = math.sqrt(work)
            else:
                work = Et[0]*(rho0[k-1]**2 - rho0[k]**2)
                work /= Et[k]*(1 - rho0[1]**2)
                r1[k] = math.sqrt(work)

        for k in range(self.Lmax):
            N1.append(0)

        nsf0 = self.P/numpy.dot(Et,r1)
        N1[0] = math.ceil(nsf0)
        for k in range(self.Lmax):
            nsf = nsf0*r1[k]
            N1[k] = math.ceil(nsf)

        # limit the number of samples on the last one to pass the sanity check, shouldnt need this anymore
        sanity = numpy.dot(N1, Et)
        if sanity > 1.2*self.P:
            N1n = (self.P - numpy.dot(N1[0:self.Lmax-2], Et[0:self.Lmax-2]))/Et[self.Lmax-1]
            N1[self.Lmax-1] = math.ceil(N1n)

        # compute the MSE beating MC condition
        cond = 0.
        for k in range(self.Lmax):
            if k == self.Lmax - 1:
                cond += math.sqrt((Et[k]/Et[0])*(rho0[k]**2))
            else:
                cond += math.sqrt((Et[k]/Et[0])*(rho0[k]**2 - rho0[k+1]**2))

        self.Pr = numpy.dot(N1, Et)
        #import pdb; pdb.set_trace()
        self.N1 = N1
        self.a1 = a1   
        if rank == 0:
            print("MFMC Completed, Samples per level: ", N1)

         

        # call dist_samples after this

    def dist_samples(self):
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
        
        self.current_samples = self.N1[self.Lmax-1]
        if rank == 0:
            rank0sam = plate_sa_lhs.genLHS(s=self.current_samples, mcs = self.uoptions['MCPure'])
        else:
            rank0sam = None
        self.sample = comm.bcast(rank0sam, root=0)
        #import pdb; pdb.set_trace()
        # Scatter samples on each level, multi-point parallelism
        self.cases = []
        self.css = []
        self.samplep = []
        for i in range(self.Lmax):
            div = divide_cases(self.N1[i], size)
            div2 = divide_cases(self.N1[i], size)
            self.cases.append(div)
            self.css.append(div2)
            for j in range(len(self.cases[i])):
                for k in range(len(self.cases[i][j])):
                    self.cases[i][j][k] += sum(self.N1[0:i]) # just for aeroproblem numbering
            self.samplep.append([]) 
        self.samplep[self.Lmax-1] = self.sample[self.css[self.Lmax-1][rank]]

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
            nslp.append(len(self.cases[k][rank]))
            nslt.append(sum([len(self.cases[k][x]) for x in range(size)]))
            for i in range(nslp[k]):
                namestr = self.gridSol + "_" + str(self.cases[k][rank][i])
                alist.append(AeroProblem(name=namestr, alpha=alpha, mach=mach, reynolds=Re, reynoldsLength=Re_L, T=tempR, areaRef=arearef, chordRef=chordref, evalFuncs=['cd']))
            # create meshes
            leveloptions = self.woptions
            leveloptions['gridFile'] = self.meshnames[self.mord[k]] 
            mlist = USMesh(options=leveloptions, comm=MPI.COMM_SELF)

            aloptions = self.aoptions
            aloptions['gridFile'] = self.meshnames[self.mord[k]] 
            time.sleep(0.1) # this solves a few problems for some reason
            # create solvers
            slist = ADFLOW(options=aloptions, comm=MPI.COMM_SELF)
            slist.setDVGeo(self.DVGeo)
            slist.setMesh(mlist)
            coords = slist.getSurfaceCoordinates(groupName=slist.allWallsGroup)
            slist.DVGeo.addPointSet(coords, 'coords')

            self.aps.append(alist)
            self.solvers.append(slist)
            self.meshes.append(mlist)

        

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

