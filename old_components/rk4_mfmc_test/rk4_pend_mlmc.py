import numpy
import math
import sys, os
import openmdao.api as om
import plate_ffd as pf
import math
import plate_sa_lhs
import subprocess
import time
from mpi4py import MPI
from rk4_pend import rk4_pend_solver
from rk4_pend_opts import uqOptions

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

class RK4PendCompMLMC(om.ExplicitComponent):
    """Stochastic Damped Pendulum Problem, with LHS Multi Level Monte Carlo Samples"""
    def initialize(self):
        # Need to modify this dictionary when we change the SA constants
        #import pdb; pdb.set_trace()
        self.uoptions = uqOptions

        self.L = self.uoptions['L']
        self.m = self.uoptions['m']
        self.g = self.uoptions['g']
        self.crange = self.uoptions['crange']

        self.Lmax = self.uoptions['levels']

        self.mord = [*range(self.Lmax)]
        self.mord.reverse()

        #if self.Lmax < 3:
        #    raise ValueError('Not enough meshes available, ' + self.Lmax + ' < 3')

        self.NS0 = self.uoptions['NS0']
        self.current_samples = self.NS0*self.Lmax
        self.N1 = None
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

    def setup(self):
        self.add_output('uf_m', 0.0, desc="Mean Final Position")
        self.add_output('uf_v', 0.0, desc="Final Position Variance")
        self.add_output('N1', self.N1, desc="MLMC Samples at each level")
    

    def compute(self, inputs, outputs):
        # run the bump shape model
        #import pdb; pdb.set_trace()
        
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

        tspan = numpy.zeros(2)
        u0 = numpy.zeros(2)
        tspan[0] = 0.
        tspan[1] = self.uoptions['Tf']
        u0[0] = self.uoptions['u0'][0]
        u0[1] = self.uoptions['u0'][1]

        for k in range(self.Lmax): 
            sump.append(0.0)
            musp.append(numpy.zeros(nslp[k]))
            sumpm.append(0.)
            if k > 0:
                muspm.append(numpy.zeros(nslp[k-1]))
            else:
                muspm.append(numpy.zeros(nslp[k])) #not used
            for i in range(nslp[k]):
                num_steps = 2**(5+self.mord[k])
                c = self.csamplep[self.Lmax-1][i]
                SOL = self.solver.solve_for_uf(self.m, self.g, c, self.L, tspan, u0, num_steps)
                musp[k][i] = SOL
                sump[k] += SOL
                if k > 0 and i < nslp[k-1]:
                    muspm[k][i] = -SOL
                    sumpm[k] += -SOL
            #import pdb; pdb.set_trace()
            # if k > 0:
            #         num_steps = 2**(5+self.mord[k-1])
            #         SOL = self.solver.solve_for_uf(self.m, self.g, c, self.L, tspan, u0, num_steps)
            #         muspm[k][i] = -SOL
            #         sumpm[k] += -SOL   
                    
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
        outputs['uf_m'] = numpy.dot(self.a1, E)
        outputs['uf_v'] = numpy.dot(self.a1, V) #by assumption
        outputs['N1'] = self.N1
        #outputs['Cd'] = inputs['a']*inputs['a']


    def MFMC(self):
        # Use an MFMC algorithm to determine optimal sample distribution and coefficients
        # We do this once before optimization, then compute statistics with the same set of samples at every iteration

        # start with initial samples
        # Get a set of UQ sample points (LHS), enough for each level at the start
        #sys.stdout = open(os.devnull, "w")
        
        N1 = []
        

        a1 = numpy.zeros(self.Lmax)
        r1 = numpy.zeros(self.Lmax)

        self.current_samples = self.NS0*self.Lmax
        if rank == 0:
            csamp = self.crange[0] + (self.crange[1] - self.crange[0])*numpy.random.rand(self.current_samples)
        else:
            csamp = None
        self.csample = comm.bcast(csamp, root=0)

        # Scatter samples on each level, multi-point parallelism
        
        self.cases = []
        self.csamplep = []
        for i in range(self.Lmax):
            self.cases.append(divide_cases(self.NS0, size)) 
            for j in range(len(self.cases[i])):
                for k in range(len(self.cases[i][j])):
                    self.cases[i][j][k] += i*self.NS0
            self.csamplep.append(self.csample[self.cases[i][rank]])
  
        
        # Actually create solvers (and aeroproblems?) (and mesh?) now
        tspan = numpy.zeros(2)
        u0 = numpy.zeros(2)
        tspan[0] = 0.
        tspan[1] = self.uoptions['Tf']
        u0[0] = self.uoptions['u0'][0]
        u0[1] = self.uoptions['u0'][1]
        self.solver = rk4_pend_solver()

        sumt = []
        sumtp = []
        nslp = []
        nslt = []
        sump = []
        musp = []
        sum1 = []
        mus = []
        sumpm = []
        muspm = []
        summ = []
        musm = []
        Et = numpy.zeros(self.Lmax)
        E = numpy.zeros(self.Lmax)
        V = numpy.zeros(self.Lmax)
        S = numpy.zeros(self.Lmax)

        # solve for each level
        for k in range(self.Lmax): 
            nslp.append(len(self.cases[k][rank]))
            nslt.append(sum([len(self.cases[k][x]) for x in range(size)]))
            sump.append(0.0)
            sumtp.append(0.0)
            musp.append(numpy.zeros(nslp[k]))
            sumpm.append(0.)
            muspm.append(numpy.zeros(nslp[k]))
            for i in range(nslp[k]):
                num_steps = 2**(5+self.mord[k])
                c = self.csamplep[0][i]
                pc0 = time.process_time()
                SOL = self.solver.solve_for_uf(self.m, self.g, c, self.L, tspan, u0, num_steps)
                pc1 = time.process_time()
                musp[k][i] = SOL
                sump[k] += SOL
                sumtp[k] += (pc1-pc0)

        # compute mean and variance, and measure average computation time
        for k in range(self.Lmax):
            sumt.append(comm.allreduce(sumtp[k]))
            sum1.append(comm.allreduce(sump[k]))
            mus.append(comm.allgather(musp[k]))
            summ.append(comm.allreduce(sumpm[k]))
            musm.append(comm.allgather(muspm[k]))
            mus[k] = numpy.concatenate(mus[k][:])
            musm[k] = numpy.concatenate(musm[k][:])
        #import pdb; pdb.set_trace()
            Et[k] = sumt[k]/nslt[k]
            E[k] = (sum1[k])/nslt[k] #+summ[k]
            sum2 = 0.
            for i in range(len(mus[k])): #loop over processors
                #for j in range(len(mus[k][i])): #loop over samples on processors
                    # if k > 0:
                    #     sum2 += ((mus[k][i][j] + musm[k][i][j])-E[k])**2
                    # else:
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
            
            #import pdb; pdb.set_trace()

        # now select models
        # mset = []
        # msetind = []
        # mset.append(self.mord[0])
        # msetind.append(0)
        # vMC = V[0]*Et[0]/self.P
        # vS = vMC
        # for k in range(1, self.Lmax):
        #     mset.append(self.mord[k])
        #     msetind.append(k)
        #     K = len(mset)

        #     # work = Et[0]*(rho[0,k]**2)
        #     # work /= Et[k]*(1 - rho[0,1]**2)
        #     # if ()

        #     vsum = 0.
        #     for j in range(K):
        #         if j+1 >= K:
        #             vsum += math.sqrt(Et[msetind[j]]*(rho[0,msetind[j]]**2))
        #         else:
        #             vsum += math.sqrt(Et[msetind[j]]*(rho[0,msetind[j]]**2 - rho[0,msetind[j+1]]**2))
        #     vC = V[0]*(vsum**2)/self.P

        #     if vC > vS:
        #         mset.pop(K-1)
        #         msetind.pop(K-1)
        #     else:
        #         vS = vC

        # self.mord = mset
        # self.Lmax = len(mset)

        # import pdb; pdb.set_trace()
        
        

        # now compute N1 and a1 using sigma, rho, w, and p
        for k in range(self.Lmax):
            a1[k] = S[0]*rho[0,k]/S[k]
            
            if k == 0:
                r1[k] = 1
            elif k == self.Lmax-1:
                work = Et[0]*(rho[0,k]**2)
                work /= Et[k]*(1 - rho[0,1]**2)
                r1[k] = math.sqrt(work)
            else:
                work = Et[0]*(rho[0,k-1]**2 - rho[0,k]**2)
                work /= Et[k]*(1 - rho[0,1]**2)
                r1[k] = math.sqrt(work)

        for k in range(self.Lmax):
            N1.append(0)

        nsf = self.P/numpy.dot(Et,r1)
        N1[0] = math.ceil(nsf)
        for k in range(self.Lmax):
            nsf = N1[0]*r1[k]
            N1[k] = math.ceil(nsf)

        sanity = numpy.dot(N1, Et)

        
        self.N1 = N1
        self.a1 = a1
        #import pdb; pdb.set_trace()
        #import pdb; pdb.set_trace()
      


        # once done, we have aps, solvers, meshes, which is all we need

    def dist_samples(self):
        # If we already have the number of samples, just create as many solvers as needed at each level
        # Just do this after running MFMC() anyway 
        
        self.current_samples = sum(self.N1)
        if rank == 0:
            csamp = self.crange[0] + (self.crange[1] - self.crange[0])*numpy.random.rand(self.current_samples)
        else:
            csamp = None
        self.csample = comm.bcast(csamp, root=0)

        self.cases = []
        self.csamplep = []
        for i in range(self.Lmax):
            self.cases.append(divide_cases(self.N1[i], size)) 
            for j in range(len(self.cases[i])):
                for k in range(len(self.cases[i][j])):
                    self.cases[i][j][k] += sum(self.N1[0:i])
            #self.nsp.append(len(self.cases[i][rank]))#int(ns/size) # samples per processor
            self.csamplep.append(self.csample[self.cases[i][rank]])

        self.solver = rk4_pend_solver()
        

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

