import numpy as np
import copy

from matplotlib import pyplot as plt
from smt.utils.options_dictionary import OptionsDictionary
from smt.sampling_methods import LHS
from smt.surrogate_models import GEKPLS
from pougrad import POUMetric, POUSurrogate
from refinecriteria import ASCriteria
from scipy.linalg import lstsq, eig
from scipy.stats import qmc
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.optimize import Bounds
from sutils import linear, quadratic, quadraticSolve, quadraticSolveHOnly, symMatfromVec, maxEigenEstimate, boxIntersect


# Hessian estimation to generate an anisotropic mapping of the space
class AnisotropicRefine(ASCriteria):
    def __init__(self, model, grad, bounds, **kwargs):

        self.bads = None
        self.bad_list = None
        self.nbhd = None
        self.eigvals = None

        self.trx = None # use for sequential optimization for batches of points
        self.dminmax = None
        self.grad = grad
        self.bounds = bounds
        self.bnorms = None
        self.bpts = None
        self.numer = None

        self.metric = None
        self.mmodel = None #POU model of the anisotropy metric

        super().__init__(model, **kwargs)

        self.supports["obj_derivatives"] = True  
        
    def _init_options(self):
        #options: neighborhood, surrogate, exact
        self.options.declare("hessian", "neighborhood", types=str)

        #options: honly, full, arnoldi
        self.options.declare("interp", "arnoldi", types=str)

        #options: linear, quadratic
        self.options.declare("error", "linear", types=str)

        self.options.declare("improve", 0, types=int)

        #number of closest points to evaluate hessian eigenpair
        self.options.declare("neval", self.dim*2+1, types=int)

        #number of optimizations to try per point
        self.options.declare("multistart", 1, types=int)

        self.options.declare("bpen", False, types=bool)

        #options: inv, exp
        self.options.declare("objective","inv", types=str)
        
        self.options.declare("rscale", 0.5, types=float)

        self.options.declare("nscale", 1.0, types=float)

    def initialize(self, model=None, grad=None):
        
        # set up constraints
        # self.condict = {
        #     "type":"ineq",
        #     "fun":self.eval_constraint,
        #     "args":[],
        # }


        if(model is not None):
            self.model = copy.deepcopy(model)
            kx = 0
            self.dim = self.model.training_points[None][kx][0].shape[1]
            self.ntr = self.model.training_points[None][kx][0].shape[0]

        if(grad is not None):
            self.grad = grad

        self.nnew = self.options["improve"]#int(self.ntr*self.options["improve"])
        if(self.nnew == 0):
            self.nnew = 1

        trx = self.model.training_points[None][0][0]
        trf = self.model.training_points[None][0][1]
        trg = np.zeros_like(trx)
        trg = self.grad
        if(isinstance(self.model, GEKPLS)):
            for j in range(self.dim):
                trg[:,j] = self.model.training_points[None][j+1][1].flatten()

        self.trx = trx
        self.numer = self.options["nscale"]

        dists = pdist(trx)
        dists = squareform(dists)

        neval = self.options["neval"]
        if(self.options["interp"] == "arnoldi"):
            neval = self.dim


        # 1. Estimate the Hessian/the principal Hessian eigenpair about each point
        hess = []
        metric = np.zeros([self.ntr, self.dim, self.dim])
        nbhd = []
        indn = []

        # 1a. Determine the neighborhood to fit the Hessian for each point/evaluate the error
        # along with minimum distances
        mins = np.zeros(self.ntr)
        for i in range(self.ntr):
            ind = dists[i,:]
            ind = np.argsort(ind)
            indn.append(ind)
            mins[i] = dists[i,ind[1]]
        self.dminmax = max(mins)
        lmax = np.amax(dists)

        if(self.options["hessian"] == "neighborhood"):        
            for i in range(self.ntr):
                if(self.options["interp"] == "full"):
                    fh, gh, Hh = quadraticSolve(trx[i,:], trx[indn[i][1:neval+1],:], \
                                            trf[i], trf[indn[i][1:neval+1]], \
                                            trg[i,:], trg[indn[i][1:neval+1],:])

                if(self.options["interp"] == "honly"):
                    Hh = quadraticSolveHOnly(trx[i,:], trx[indn[i][1:neval+1],:], \
                                            trf[i], trf[indn[i][1:neval+1]], \
                                            trg[i,:], trg[indn[i][1:neval+1],:])
                    fh = trf[i]
                    gh = trg[i,:]

                if(self.options["interp"] == "full" or self.options["interp"] == "honly"):
                    hess.append(np.zeros([self.dim, self.dim]))
                    for j in range(self.dim):
                        for k in range(self.dim):
                            hess[i][j,k] = Hh[symMatfromVec(j,k,self.dim)]
                
                else: #arnoldi
                    evalm, evecm = maxEigenEstimate(trx[i,:], trx[indn[i][1:neval],:], \
                                                    trg[i,:], trg[indn[i][1:neval],:])

                    hess.append([evalm, evecm])

        if(self.options["hessian"] == "surrogate"):
            # 1a. Get the hessian as determined by the surrogate
            # central difference scheme
            h = 1e-5
            for i in range(self.ntr):
                hess.append(np.zeros((self.dim, self.dim)))
            
            for j in range(self.dim):
                xsp = np.copy(trx)
                xsm = np.copy(trx)
                xsp[:,j] += h
                xsm[:,j] -= h

                for k in range(self.dim):
                    hj = np.zeros(self.dim)
                    hj = self.model.predict_derivatives(xsp, k)
                    hj -= self.model.predict_derivatives(xsm, k)
                    for l in range(len(hess)):
                        hess[l][j,k] = hj[l]/h




        


        # 2. Take the highest eigenvalue/vector of each Hessian and generate the metric M
        opt_dir = []
        opt_val = []
        min_eig = [] # for symmetry
        if(self.options["interp"] == "arnoldi"):
            for i in range(self.ntr):
                opt_dir.append(hess[i][1])
                opt_val.append(hess[i][0])

        else:
            for i in range(self.ntr):
                H = hess[i]
                eigvals, eigvecs = eig(H)
                o = np.argsort(abs(eigvals))
                opt_dir.append(eigvecs[:,o[-1]])
                opt_val.append(eigvals[o[-1]])
                min_eig.append(eigvals[o[0]])


        for i in range(self.ntr):
            work = np.outer(opt_dir[i], opt_dir[i])
            metric[i,:,:] = np.abs(opt_val[i])*work
            metric[i,:,:] += np.abs(np.real(min_eig[i]))*(np.eye(self.dim) - work)

        metric /= sum(np.abs(np.real(opt_val)))/(len(opt_val)*1.)

        # Now train a POU surrogate on the metric

        # increase rho based on number of points, minimum distance
        #rho = 1.*self.dim*self.ntr#100.#self.ntr*100.#/(self.dim*1.)
        rho = self.options['rscale']*(self.ntr/self.dim)

        self.mmodel = POUMetric(rho=rho, metric=metric) # the metric are the actual training outputs, this is a bad workaround
        #self.mmodel.set_training_values(trx, np.ones(trx.shape[0]))
        self.mmodel.set_training_values(qmc.scale(trx, self.bounds[:,0], self.bounds[:,1], reverse=True), \
                                np.ones(trx.shape[0]))

        # h = 1e-6
        # zero = np.zeros([1,2])
        # step = np.zeros([1,2])
        # step[0,0] += h
        # ad = self.mmodel.predict_derivatives(zero)
        # fd = (self.mmodel.predict_values(step) - self.mmodel.predict_values(zero))/h

        # 3. For every point, sum the discrepancies between the linear (quadratic)
        # prediction in the neighborhood and the observation value
        err = np.zeros(self.ntr)

        # using the original neval here
        for i in range(self.ntr):
            #ind = indn[i]
            ind = dists[i,:]
            ind = np.argsort(ind)
            for key in ind[1:self.options["neval"]]:
                if(self.options["error"] == "linear" or self.options["interp"] == "arnoldi"):
                    fh = linear(trx[key], trx[i], trf[i], trg[:][i])
                else:
                    fh = quadratic(trx[key], trx[i], trf[i], trg[:][i], hess[i])
                err[i] += abs(trf[key] - fh)
            err[i] /= self.options["neval"]
            nbhd.append(ind[1:neval])

        emax = max(err)
        for i in range(self.ntr):
            # ADDING A DISTANCE PENALTY TERM
            err[i] /= emax
            err[i] *= 1. - mins[i]/lmax
            err[i] += mins[i]/self.dminmax

        # 3a. Pick some percentage of the "worst" points, use these to start the local optimizer
        badlist = np.argsort(err)
        badlist = badlist[-self.nnew:]
        bads = trx[badlist]
        # bad_nbhd = np.zeros([bads.shape[0], self.options["neval"]-1], dtype=int)
        # for i in range(bads.shape[0]):
        #     bad_nbhd[i,:] = nbhd[badlist[i]]

        # h = 1e-6
        # zero = 0.5*np.ones([2])
        # step = 0.5*np.ones([2])
        # step[0] += h
        # ad = self.eval_grad(zero, bounds)
        # fd = (self.evaluate(step, bounds) - self.evaluate(zero, bounds))/h
        # import pdb; pdb.set_trace()

        # import pdb; pdb.set_trace()
        # we have what we need
        self.bad_list = badlist
        self.bad_eigs = opt_val
        self.bads = bads
        #self.bad_nbhd = bad_nbhd
        self.bad_dirs = opt_dir
        

    def evaluate(self, x, bounds, dir=0):
        
        trx = qmc.scale(self.trx, bounds[:,0], bounds[:,1], reverse=True)
        #trx = self.bads
        #x = qmc.scale(x, bounds[:,0], bounds[:,1], reverse=True)
        m, n = trx.shape

        N = self.numer

        sum = 0
        sum = np.zeros(m)
        mwork = self.mmodel.predict_values(np.array([x]))
        for i in range(m):
            work = x-trx[i]
            #sum += 1./(np.sqrt(np.matmul(np.matmul(work, mwork), work)**2) + 1e-10)
            dist = np.matmul(np.matmul(work, mwork), work)

            if(self.options["objective"] == "inv"):
                sum[i] = N/(dist + 1e-10)
            elif(self.options["objective"] == "geom"):
                sum[i] = N/(np.power(dist, n/2.)  + 1e-10)
            elif(self.options["objective"] == "abs"):
                sum[i] = -dist
            else:    
                sum[i] = np.exp(-np.sqrt(dist))
            #sum += np.exp(-np.linalg.norm(work)**2)

        #ans = max(sum)# - np.linalg.norm(mwork)
        ans = np.sum(sum)/m
        #ATTEMPT 1
        #penalize distance to bound
        # mb, nb = self.bounds.shape
        # bsum = np.zeros(mb)
        # for i in range(mb):
        #     work = x - self.bounds[i]
        #     dist = np.matmul(np.matmul(work, mwork), self.bnorms[i])
        #     bsum[i] = 1./(dist**2 + 1e-10)

        mb = 0
        if(self.options["bpen"]):
            #ATTEMPT 2
            #penalize distance to bound
            # mb = len(self.bpts)
            # if(mb > 0):
            #     bsum = np.zeros(mb)
            #     for i in range(mb):
            #         #bpts = qmc.scale(np.array([self.bpts[i]]), bounds[:,0], bounds[:,1], reverse=True)[0]
            #         work = x - self.bpts[i]
            #         #dist = np.matmul(np.matmul(work, mwork), self.bnorms[i])
            #         dist = abs(np.dot(work, self.bnorms[i]))
            #         if(self.options["objective"] == "inv"):
            #             bsum[i] = N/(dist**2 + 1e-10)
            #         elif(self.options["objective"] == "geom"):
            #             bsum[i] = N/(np.power(dist, n)  + 1e-10)
            #         elif(self.options["objective"] == "abs"):
            #             bsum[i] = -dist**2
            #         else:
            #             bsum[i] = np.exp(-dist)

            #     ans += np.sum(bsum)/mb

            #ATTEMPT 3
            mb, nb = self.bpts.shape
            bsum = np.zeros(mb)
            for i in range(mb):
                work = x-self.bpts[i]
                #sum += 1./(np.sqrt(np.matmul(np.matmul(work, mwork), work)**2) + 1e-10)
                dist = np.matmul(np.matmul(work, mwork), work)

                if(self.options["objective"] == "inv"):
                    bsum[i] = N/(dist + 1e-10)
                elif(self.options["objective"] == "geom"):
                    bsum[i] = N/(np.power(dist, n/2.)  + 1e-10)
                elif(self.options["objective"] == "abs"):
                    bsum[i] = -dist
                else:    
                    bsum[i] = np.exp(-np.sqrt(dist))

            ans += np.sum(bsum)/mb

        return ans 


    def eval_grad(self, x, bounds, dir=0):
        
        trx = qmc.scale(self.trx, bounds[:,0], bounds[:,1], reverse=True)
        #trx = self.bads
        m, n = trx.shape

        N = self.numer

        sum = 0
        sum = np.zeros(m)
        mwork = self.mmodel.predict_values(np.array([x]))
        dmwork = self.mmodel.predict_derivatives(np.array([x]))
        dsum = np.zeros(n)
        for i in range(m):
            work = x-trx[i]

            leftprod = np.matmul(work, mwork)
            rightprod = np.matmul(mwork, work)
            dist = np.matmul(leftprod, work)
            ddist = np.zeros(n)
            ddist += rightprod[0]
            ddist += leftprod[0]

            for j in range(n):
                ddist[j] += np.matmul(np.matmul(work, dmwork[0,j,:,:]), work)

            if(self.options["objective"] == "inv"):
                sum[i] = N/(dist + 1e-10)    
                dsum += (-N/((dist + 1e-10)**2))*ddist/m
            elif(self.options["objective"] == "geom"):
                sum[i] = N/(np.power(dist, n/2.)  + 1e-10)
                dsum += (-N/((np.power(dist, n/2.) + 1e-10)**2))*((n/2.)*np.power(dist, n/2.-1.))*ddist/m
            elif(self.options["objective"] == "abs"):
                sum[i] = -dist
                dsum += -ddist/m
            else:
                sum[i] = np.exp(-np.sqrt(dist))
                dsum += np.exp(-np.sqrt(dist))*(-1./(2.*np.sqrt(dist)))*ddist/m

        #ATTEMPT 1
        #penalize distance to bound
        # mb, nb = self.bounds.shape
        # bsum = np.zeros(mb)
        # dbsum = np.zeros(n)
        # for i in range(mb):
        #     work = x - self.bounds[i]

        #     leftprod = np.matmul(work, mwork)
        #     rightprod = np.matmul(mwork, self.bnorms[i])
        #     dist = np.matmul(np.matmul(work, mwork), self.bnorms[i])
        #     ddist = np.zeros(n)
        #     ddist += rightprod[0]
        #     for j in range(n):
        #         ddist[j] += np.matmul(np.matmul(work, dmwork[0,j,:,:]), self.bnorms[i])

        #     bsum[i] = 1./(dist**2 + 1e-10)
        #     dbsum += (-1./((dist**2 + 1e-10)**2))*ddist*2*dist

        if(self.options["bpen"]):
            #ATTEMPT 2
            #penalize distance to bound
            # mb = len(self.bpts)
            # if(mb > 0):
            #     bsum = np.zeros(mb)
            #     dbsum = np.zeros(n)
            #     for i in range(mb):
            #         bpts = qmc.scale(np.array([self.bpts[i]]), bounds[:,0], bounds[:,1], reverse=True)[0]
            #         work = x - bpts

            #         # leftprod = np.matmul(work, mwork)
            #         # rightprod = np.matmul(mwork, self.bnorms[i])
            #         #dist = np.matmul(np.matmul(work, mwork), self.bnorms[i])
            #         dist = abs(np.dot(work, self.bnorms[i]))
            #         ddist = np.zeros(n)
            #         ddist += abs(self.bnorms[i])
            #         # for j in range(n):
            #         #     ddist[j] += np.matmul(np.matmul(work, dmwork[0,j,:,:]), self.bnorms[i])

            #         if(self.options["objective"] == "inv"):
            #             bsum[i] = N/(dist**2 + 1e-10)
            #             dbsum += (-N/((dist**2 + 1e-10)**2))*ddist*2*dist/mb
            #         elif(self.options["objective"] == "geom"):
            #             bsum[i] = N/(np.power(dist, n)  + 1e-10)
            #             dbsum += (-N/((np.power(dist, n) + 1e-10)**2))*((n)*np.power(dist, n-1.))*ddist/mb
            #         elif(self.options["objective"] == "abs"):
            #             bsum[i] = -dist**2
            #             dbsum += -2*dist*ddist/mb
            #         else:
            #             bsum[i] = np.exp(-dist)
            #             dbsum += np.exp(-dist)*(-1.)*ddist/mb
            #     dsum += dbsum

            #ATTEMPT 3
            mb, nb = self.bpts.shape

            N = self.numer

            bsum = np.zeros(mb)
            dbsum = np.zeros(nb)
            for i in range(mb):
                work = x-self.bpts[i]

                leftprod = np.matmul(work, mwork)
                rightprod = np.matmul(mwork, work)
                dist = np.matmul(leftprod, work)
                ddist = np.zeros(n)
                ddist += rightprod[0]
                ddist += leftprod[0]

                for j in range(n):
                    ddist[j] += np.matmul(np.matmul(work, dmwork[0,j,:,:]), work)

                if(self.options["objective"] == "inv"):
                    bsum[i] = N/(dist + 1e-10)    
                    dbsum += (-N/((dist + 1e-10)**2))*ddist/mb
                elif(self.options["objective"] == "geom"):
                    bsum[i] = N/(np.power(dist, n/2.)  + 1e-10)
                    dbsum += (-N/((np.power(dist, n/2.) + 1e-10)**2))*((n/2.)*np.power(dist, n/2.-1.))*ddist/mb
                elif(self.options["objective"] == "abs"):
                    bsum[i] = -dist
                    dbsum += -ddist/mb
                else:
                    bsum[i] = np.exp(-np.sqrt(dist))
                    dbsum += np.exp(-np.sqrt(dist))*(-1./(2.*np.sqrt(dist)))*ddist/mb

            dsum += dbsum

        return dsum 




    def pre_asopt(self, bounds, dir=0):
        
        xc = self.bads[dir]
        xdir = self.bad_dirs[dir]
        trx = self.trx
        m, n = trx.shape

        # Store points at centroids of the bounds
        # center = np.zeros(n)
        # for i in range(n):
        #     center[i] = (bounds[i,1]-bounds[i,0])/2. + bounds[i,0]
        # #self.bounds = bounds
        # self.bounds = np.zeros([2*n, n])
        # for i in range(n):
        #     self.bounds[i,:] = center
        #     self.bounds[i,i] = bounds[i,0] 
        #     self.bounds[i+n,:] = center
        #     self.bounds[i+n,i] = bounds[i,1] 
        
        # Store boundary normals
        self.bnorms = np.zeros([2*n, n])
        for i in range(n):
            self.bnorms[i,i] = -1.
            self.bnorms[i+n,i] = 1.



        # need to update this for batches
        #dists = pdist(trx, lambda u, v: np.matmul(np.matmul((u-v), self.mmodel.predict_values(np.array([u]))), (u-v)))
        dists = pdist(trx)
        dists = squareform(dists)
        mins = np.zeros(m)
        for i in range(m):
            ind = dists[i,:]
            ind = np.argsort(ind)
            mins[i] = dists[i,ind[1]]
        self.dminmax = max(mins)
        # S = 0.5*self.dminmax



        # find which points are on boundaries 
        # self.bpts = []
        # self.bnorms = []
            
        # co = 0
        # for i in range(m):
        #     on_bound = False
        #     work = abs(trx[i,:] - bounds[:,0]) #lb
        #     work2 = abs(trx[i,:] - bounds[:,1]) #ub
        #     if((work < 1e-8).any() or (work2 < 1e-8).any()):
        #         on_bound = True

        #     if(on_bound):
        #         self.bpts.append(qmc.scale(np.array([trx[i,:]]), bounds[:,0], bounds[:,1], reverse=True)[0])
        #         self.bnorms.append(np.zeros([n]))
        #         for j in range(n):
        #             if(abs(trx[i,j] - bounds[j ,0]) < 1e-8):
        #                 self.bnorms[co][j] = -1.
        #             if(abs(trx[i,j] - bounds[j ,1]) < 1e-8):
        #                 self.bnorms[co][j] = 1.
        #         co += 1

        sbounds = np.zeros_like(bounds)
        sbounds[:,1] = 1.
        bigbounds = 2*sbounds - 0.5
        bsampling = LHS(xlimits=bigbounds, criterion='m')
        bpts = bsampling(self.ntr*(2**self.dim))
        mb, nb = bpts.shape
        mblist = []
        for i in range(mb):
            in_bound = False
            work = bpts[i,:] - sbounds[:,0]
            work2 = bpts[i,:] - sbounds[:,1]
            if((work > 0.).all() and (work2 < 0.).all()):
                in_bound = True

            if(in_bound):
                mblist.append(i)
        
        # pop out interior points
        bpts = np.delete(bpts, mblist, axis=0)
        self.bpts = bpts


        # h = 1e-6
        # zero = 0.5*np.ones([2])
        # step = 0.5*np.ones([2])
        # step[0] += h
        # ad = self.eval_grad(zero, bounds)
        # fd = (self.evaluate(step, bounds) - self.evaluate(zero, bounds))/h
        # import pdb; pdb.set_trace()

        ndir = 100
        # x = np.linspace(bounds[0][0], bounds[0][1], ndir)
        # y = np.linspace(bounds[1][0], bounds[1][1], ndir)
        x = np.linspace(0., 1., ndir)
        y = np.linspace(0., 1., ndir)

        X, Y = np.meshgrid(x, y)
        F  = np.zeros([ndir, ndir])


        for i in range(ndir):
            for j in range(ndir):
                xi = np.zeros([2])
                xi[0] = x[i]
                xi[1] = y[j]
                F[i,j]  = self.evaluate(xi, bounds)

        cs = plt.contour(Y, X, F, levels = np.linspace(0.,500.,25))
        plt.colorbar(cs)
        trxs = qmc.scale(self.trx, bounds[:,0], bounds[:,1], reverse=True)
        plt.plot(trxs[0:-1,0], trxs[0:-1,1], 'bo')
        plt.plot(trxs[-1,0], trxs[-1,1], 'ro')
        plt.savefig("refine_contour_2.png")

        plt.clf()

        import pdb; pdb.set_trace()

        sampling = LHS(xlimits=bounds, criterion='m')
        ntries = self.options["multistart"]
        if(ntries > 1):
            xc = sampling(ntries)
        else: 
            xc = np.random.rand(n)*(bounds[:,1] - bounds[:,0]) + bounds[:,0]
            xc = np.array([xc])

        # for i in range(n):
        #     xc[i] = xc[i]*(bounds[i,1] - bounds[i,0]) + bounds[i,0]
        return xc, bounds# + 0.001*self.dminmax+randvec, bounds



    def post_asopt(self, x, bounds, dir=0):

        # add x to trx, for constraint purposes when dealing with batches
        #import pdb; pdb.set_trace()
        self.trx = np.append(self.trx, np.array([x]), axis=0)
        #import pdb; pdb.set_trace()
        return x



    # def eval_constraint(self, x, dir=0):

    #     trx = self.trx
    #     m, n = trx.shape
        
    #     # # find a cluster threshold (max of minimum distances, Aute 2013)
    #     S = 0.5*self.dminmax

    #     con = np.ones(m)
    #     #mwork = self.mmodel.predict_values(np.array([x]))
    #     for i in range(m):
    #         #work = x-trx[i]
    #         #con[i] = np.matmul(np.matmul(work, mwork), work)
    #         #con[i] = np.sqrt(con[i])
    #         con[i] = np.linalg.norm(x-trx[i])

    #     return con - S










