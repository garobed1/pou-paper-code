import numpy as np
import copy

from matplotlib import pyplot as plt
from smt.utils.options_dictionary import OptionsDictionary
from smt.sampling_methods import LHS
from smt.surrogate_models import GEKPLS, KPLS, KRG
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
        self.vmodel = None #surrogate of surrogate, variances modified by metric

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

        #options: inv, exp, geom, mvar
        self.options.declare("objective","inv", types=str)

        #use continuous tead error measure
        self.options.declare("objerr", False, types=bool)
        
        self.options.declare("rscale", 0.5, types=float)

        self.options.declare("nscale", 1.0, types=float)

    def initialize(self, model=None, grad=None):

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
                #min_eig.append(1e-5)

        for i in range(self.ntr):
            work = np.outer(opt_dir[i], opt_dir[i])
            metric[i,:,:] = np.abs(opt_val[i])*work
            metric[i,:,:] += np.abs(np.real(min_eig[i]))*(np.eye(self.dim) - work)

        metric /= sum(np.abs(np.real(opt_val)))/(len(opt_val)*1.)

        # if using kriging variance, train a Kriging model that uses the metric distances in the correlation
        if(self.options["objective"] == "mvar"):
            self.vmodel = KRG(
                metric_warp=metric, 
                corr=model.options["corr"], 
                poly=model.options["poly"],
                n_start=model.options["n_start"],
                print_global=False)
            self.vmodel.set_training_values(trx, trf)
            self.vmodel.train()

        # increase rho based on number of points, minimum distance
        #rho = 1.*self.dim*self.ntr#100.#self.ntr*100.#/(self.dim*1.)
        rho = self.options['rscale']*(self.ntr/self.dim)

        self.mmodel = POUMetric(rho=rho, metric=metric) # the metric are the actual training outputs, this is a bad workaround
        #self.mmodel.set_training_values(trx, np.ones(trx.shape[0]))
        self.mmodel.set_training_values(qmc.scale(trx, self.bounds[:,0], self.bounds[:,1], reverse=True), \
                                np.ones(trx.shape[0]))


    def evaluate(self, x, bounds, dir=0):
        
        trx = qmc.scale(self.trx, bounds[:,0], bounds[:,1], reverse=True)
        trf = self.model.training_points[None][0][1]
        m, n = trx.shape

        N = self.numer
        C = 5.

        if(self.options["objective"] == "mvar"):
            ans = self.vmodel.predict_variances(np.array([x]))
        else:
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

            ans = np.sum(sum)/m
        
        # only loop over previous batch, since we only have those gradients
        if(self.options["objerr"]):
            mg = self.grad.shape[0]
            esum = np.zeros(mg)
            for i in range(mg):
                work = x-trx[i]
                fhat = self.model.predict_values(np.array([x]))
                ft = linear(x, trx[i], trf[i], self.grad[i])
                esum[i] = np.abs(fhat - ft)*np.exp(-np.linalg.norm(work)*C)
                
            import pdb; pdb.set_trace()
            ans -= np.sum(esum)
        
        
        mb = 0
        if(self.options["bpen"]):
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
        trf = self.model.training_points[None][0][1]
        m, n = trx.shape

        N = self.numer
        C = 5.

        if(self.options["objective"] == "mvar"):
            #ans = self.vmodel.predict_variances(x)
            dsum = self.vmodel.predict_variance_derivatives(np.array([x]))
        else:
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
                    dsum += (-N/((dist + 1e-10)**2))*ddist/m# + dN/(dist + 1e-10)
                elif(self.options["objective"] == "geom"):
                    sum[i] = N/(np.power(dist, n/2.)  + 1e-10)
                    dsum += (-N/((np.power(dist, n/2.) + 1e-10)**2))*((n/2.)*np.power(dist, n/2.-1.))*ddist/m
                elif(self.options["objective"] == "abs"):
                    sum[i] = -dist
                    dsum += -ddist/m
                else:
                    sum[i] = np.exp(-np.sqrt(dist))
                    dsum += np.exp(-np.sqrt(dist))*(-1./(2.*np.sqrt(dist)))*ddist/m


        # only loop over previous batch, since we only have those gradients
        if(self.options["objerr"]):
            mg = self.grad.shape[0]
            esum = np.zeros(mg)
            desum = np.zeros(n)
            dfhat = np.zeros(n)
            for i in range(mg):
                work = x-trx[i]
                fhat = self.model.predict_values(np.array([x]))
                for j in range(self.dim):
                    dfhat[j] = self.model.predict_derivatives(np.array([x]), j)
                desum += np.abs(dfhat - self.grad[i])*np.exp(-np.linalg.norm(work)*C)
                ft = linear(x, trx[i], trf[i], self.grad[i])
                esum[i] = np.abs(fhat - ft)*np.exp(-np.linalg.norm(work)*C)
                desum += esum[i]*(-C*x/np.linalg.norm(work))
                
            dsum -= desum

        if(self.options["bpen"]):
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
        trf = self.model.training_points[None][0][1]
        # use surrogate values for the unevaluated points
        extra = trx.shape[0]-trf.shape[0]
        if extra:
            trf = np.append(trf, self.model.predict_values(trx[-extra:]))

        m, n = trx.shape

        # Store boundary normals
        self.bnorms = np.zeros([2*n, n])
        for i in range(n):
            self.bnorms[i,i] = -1.
            self.bnorms[i+n,i] = 1.


        if(self.options["objective"] == "mvar"):
            self.vmodel.options["metric_warp"] = self.mmodel.predict_values(trx)
            self.vmodel.set_training_values(trx, trf)
            self.vmodel.train()
        
        # h = 1e-6
        # zero = 0.5*np.ones([2])
        # step = 0.5*np.ones([2])
        # step[0] += h
        # ad = self.eval_grad(zero, bounds)
        # fd = (self.evaluate(step, bounds) - self.evaluate(zero, bounds))/h
        # import pdb; pdb.set_trace()

        # ndir = 100
        # # x = np.linspace(bounds[0][0], bounds[0][1], ndir)
        # # y = np.linspace(bounds[1][0], bounds[1][1], ndir)
        # x = np.linspace(0., 1., ndir)
        # y = np.linspace(0., 1., ndir)

        # X, Y = np.meshgrid(x, y)
        # F  = np.zeros([ndir, ndir])


        # for i in range(ndir):
        #     for j in range(ndir):
        #         xi = np.zeros([2])
        #         xi[0] = x[i]
        #         xi[1] = y[j]
        #         F[i,j]  = self.evaluate(xi, bounds)

        # cs = plt.contour(Y, X, F, levels = np.linspace(0.,500.,25))
        # plt.colorbar(cs)
        # trxs = qmc.scale(self.trx, bounds[:,0], bounds[:,1], reverse=True)
        # plt.plot(trxs[0:-1,0], trxs[0:-1,1], 'bo')
        # plt.plot(trxs[-1,0], trxs[-1,1], 'ro')
        # plt.savefig("refine_contour_2.png")

        # plt.clf()

        #import pdb; pdb.set_trace()

        sampling = LHS(xlimits=bounds, criterion='m')
        ntries = self.options["multistart"]
        if(ntries > 1):
            xc = sampling(ntries)
        else: 
            xc = np.random.rand(n)*(bounds[:,1] - bounds[:,0]) + bounds[:,0]
            xc = np.array([xc])

        return xc, bounds



    def post_asopt(self, x, bounds, dir=0):

        # add x to trx, for constraint purposes when dealing with batches
        self.trx = np.append(self.trx, np.array([x]), axis=0)

        return x











