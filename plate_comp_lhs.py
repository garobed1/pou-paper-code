import numpy
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

class PlateComponentLHS(om.ExplicitComponent):
    """Robust Bump Flow Problem"""
    def initialize(self):
        # Need to modify this dictionary when we change the SA constants
        #import pdb; pdb.set_trace()
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

        # starting flat mesh
        meshname = self.aoptions['gridFile']
        gridFile = meshname
        
        # flow characteristics
        alpha = 0.0
        mach = self.ooptions['mach']#0.95
        Re = self.ooptions['Re']#50000
        Re_L = 1.0
        temp = 540
        arearef = 2.0
        chordref = 1.0

        # Spalart Allmaras model constants, to be changed in UQ (4 for now)
        saconstsm = [0.41, 0.1355, 0.622, 0.66666666667]
        self.saconstsb = [7.1, 0.3, 2.0, 1.0, 2.0, 1.2, 0.5, 2.0]
        self.saconsts = saconstsm + self.saconstsb
        self.aoptions['SAConsts'] = self.saconsts
        #self.gridSol = f'{meshname}_{saconstsm}_sol'
        solname = self.ooptions['prob_name']
        self.gridSol = f'{solname}_sol'

        # Aerodynamic problem description
        #self.ap = AeroProblem(name=self.gridSol, alpha=alpha, mach=mach, reynolds=Re, reynoldsLength=Re_L, T=temp, areaRef=arearef, chordRef=chordref, 
        #evalFuncs=['cd'])

        # Get a set of UQ sample points (LHS)
        if self.ooptions['run_once']:
            self.sample = self.uoptions['dist']
        else:
            self.sample = plate_sa_lhs.genLHS(s=self.uoptions['NS'])

        # Set up mesh warping
        #self.mesh = USMesh(options=self.woptions, comm=MPI.COMM_SELF)

        # Scatter samples, multi-point parallelism
        ns = self.uoptions['NS']
        self.cases = divide_cases(ns, size)
        self.nsp = len(self.cases[rank])#int(ns/size) # samples per processor

        self.samplep = self.sample[self.cases[rank]]#self.sample[(rank*self.nsp):(rank*self.nsp+(self.nsp))] #shouldn't really need to "scatter" per se
        #import pdb; pdb.set_trace()
        assert len(self.samplep) == self.nsp

        #import pdb; pdb.set_trace()
        #self.CFDSolver.DVGeo.getFlattenedChildren()[1].writePlot3d("ffdp_opt_def.xyz")
        #import pdb; pdb.set_trace()

        # create copies of the original mesh (need this for idwarp?)
        # if rank == 0:
        #     #basemesh = USMesh(options=self.woptions, comm=MPI.COMM_SELF)
        #     for i in range(ns):
        #         meshname = self.ooptions['prob_name'] + "_" + str(i) + ".cgns"
        #         # basemesh.writeGrid(fileName=meshname)
        #         subprocess.run(["cp",self.woptions['gridFile'],meshname])
        #     dummy = 1
        # else:
        #     dummy = 2
        
        # dummysum = comm.allgather(dummy)
        
        # Actually create solvers (and aeroproblems?) (and mesh?) now
        self.aps = []
        self.solvers = []
        self.meshes = []

        #self.mesh = USMesh(options=self.woptions, comm=MPI.COMM_SELF)
        for i in range(self.nsp):
            namestr = self.gridSol + "_" + str(self.cases[rank][i])
            
            # create meshes 
            self.meshes.append(USMesh(options=self.woptions, comm=MPI.COMM_SELF))
            
            # create aeroproblems 
            self.aps.append(AeroProblem(name=namestr, alpha=alpha, mach=mach, reynolds=Re, reynoldsLength=Re_L, T=temp, areaRef=arearef, chordRef=chordref, evalFuncs=['cd']))
            time.sleep(0.1) # this solves a few problems for some reason
            # create solvers
            self.solvers.append(ADFLOW(options=self.aoptions, comm=MPI.COMM_SELF))
            if not self.ooptions['run_once']:
                saconstsm = self.samplep[i].tolist()
            else:
                saconstsm = self.samplep[i]
            self.saconsts = saconstsm + self.saconstsb
            self.solvers[i].setOption('SAConsts', self.saconsts)
            self.solvers[i].setDVGeo(self.DVGeo)
            self.solvers[i].setMesh(self.meshes[i])
            print("what up %i", str(rank))
            coords = self.solvers[i].getSurfaceCoordinates(groupName=self.solvers[i].allWallsGroup)
            self.solvers[i].DVGeo.addPointSet(coords, 'coords')

        # Set constraints, should only need one of those solvers, the meshes are all the same
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


    def setup(self):
        #initialize shape and set deformation points as inputs
        a_init = self.solvers[0].DVGeo.getValues()
        a_init['pnts'][:] = self.ooptions['DVInit']
        # mult = numpy.linspace(1.0,1.5,num=int(0.5*len(a_init['pnts'])))
        # mult = numpy.concatenate((mult, mult))
        if self.ooptions['run_once']:
            a_init['pnts'] = self.ooptions['ro_shape']
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
        # evaluate each sample point
        print("hello")
        dvdict = {'pnts':inputs['a']}
        funcs = {}
        ns = self.uoptions['NS']
        sump = 0.
        musp = numpy.zeros(self.nsp)
        #self.aps[0].setDesignVars(dvdict)
        for i in range(self.nsp):
            self.solvers[i].DVGeo.setDesignVars(dvdict)
            self.aps[i].setDesignVars(dvdict)
            self.solvers[i](self.aps[i])
            self.solvers[i].evalFunctions(self.aps[i], funcs)
            astr = self.gridSol + "_" + str(self.cases[rank][i]) +"_cd"
            #import pdb; pdb.set_trace()
            musp[i] = funcs[astr]
            sump += funcs[astr]
        
        # compute mean and variance
        sum = comm.allreduce(sump)
        mus = comm.allgather(musp)
        #import pdb; pdb.set_trace()
        E = sum/ns
        sum2 = 0.
        for i in range(len(mus)): #range(size):
            for j in range(len(mus[i])): #range(self.nsp):
                sum2 += (mus[i][j]-E)**2
        V = sum2/ns

        self.DVCon.evalFunctions(funcs, includeLinear=True)
        self.DVCon2.evalFunctions(funcs, includeLinear=True)

        outputs['Cd_m'] = E
        outputs['Cd_v'] = V
        rho = self.uoptions['rho']
        outputs['Cd_r'] = E + rho*math.sqrt(V)
        if self.ooptions['use_area_con']:
            outputs['SA'] = funcs['sas']
        else:
            outputs['TC'] = funcs['tcs']
        
        outputs['EQ'] = funcs['eqs']

        #outputs['Cd'] = inputs['a']*inputs['a']

    def compute_partials(self, inputs, J):

        dvdict = {'pnts':inputs['a']}
        funcs = {}
        funcSens = {}
        ns = self.uoptions['NS']
        sump = 0.
        musp = numpy.zeros(self.nsp)
        pmup = []
        psump = numpy.zeros(len(inputs['a']))
        self.aps[0].setDesignVars(dvdict)
        #import pdb; pdb.set_trace()
        for i in range(self.nsp):
            self.solvers[i].DVGeo.setDesignVars(dvdict)
            self.aps[i].setDesignVars(dvdict)
            self.solvers[i].evalFunctions(self.aps[i], funcs)
            self.solvers[i].evalFunctionsSens(self.aps[i], funcSens, ['cd'])
            astr = self.gridSol + "_" + str(self.cases[rank][i])+"_cd"
            arr = [x*(1./ns) for x in funcSens[astr]['pnts']]
            sump += funcs[astr]
            musp[i] = funcs[astr]
            pmup.append(arr[0])
            psump += arr[0]

        sum = comm.allreduce(sump)
        mus = comm.allgather(musp)
        pmu = comm.allgather(pmup)
        psum = comm.allreduce(psump)

        #import pdb; pdb.set_trace()

        # compute variance sensitivity
        E = sum/ns
        sum2 = 0.
        psum2 = numpy.zeros(len(inputs['a']))
        for i in range(len(mus)): #range(size):
            for j in range(len(mus[i])): #range(self.nsp):
                sum2 += (mus[i][j]-E)**2

                temp = pmu[i][j] - psum
                arr2 = [x*2*(mus[i][j]-E) for x in temp]
                psum2 += arr2
        V = sum2/ns
        #import pdb; pdb.set_trace()

        self.DVCon.evalFunctionsSens(funcSens, includeLinear=True)
        self.DVCon2.evalFunctionsSens(funcSens, includeLinear=True)
 
        J['Cd_m','a'] = psum
        J['Cd_v','a'] = psum2
        rho = self.uoptions['rho']
        J['Cd_r','a'] = psum + rho*(1./(2*math.sqrt(V)))*psum2
        if self.ooptions['use_area_con']:
            J['SA','a'] = funcSens['sas']['pnts']
        else:
            J['TC','a'] = funcSens['tcs']['pnts']
        J['EQ','a'] = funcSens['eqs']['pnts']

       #J['Cd','a'][0] = 2*inputs['a']

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
