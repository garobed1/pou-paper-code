import numpy as np
import copy

from matplotlib import pyplot as plt
from smt.utils.options_dictionary import OptionsDictionary
from smt.sampling_methods import LHS
from smt.surrogate_models import GEKPLS, KPLS, KRG
from pougrad import POUError, POUErrorVol, POUMetric, POUSurrogate
from refinecriteria import ASCriteria
from scipy.linalg import lstsq, eig
from scipy.stats import qmc
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.optimize import Bounds
from sutils import linear, quadratic, quadraticSolve, quadraticSolveHOnly, symMatfromVec, maxEigenEstimate, boxIntersect


# Refine based on a first-order Taylor approximation using the gradient
class TaylorRefine(ASCriteria):
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
        self.numer = 1.0

        self.tmodel = None #First-Order POU Surrogate, use to measure nonlinearity

        super().__init__(model, **kwargs)

        self.supports["obj_derivatives"] = True  
        
    def _init_options(self):

        self.options.declare("improve", 0, types=int)

        #number of optimizations to try per point
        self.options.declare("multistart", 1, types=int)

        self.options.declare("rscale", 0.5, types=float)

        #options: linear, quadratic
        self.options.declare("taylor_error", "linear", types=str)

        self.options.declare("volume_weight", False, types=bool)

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
        
        # Generate the error model
        rho = self.options['rscale']*(self.ntr/self.dim)
        if(self.options['volume_weight']):
            self.tmodel = POUErrorVol(rho=rho, xscale=self.bounds)
        else:
            self.tmodel = POUError(rho=rho, xscale=self.bounds)
        self.tmodel.set_training_values(qmc.scale(trx, self.bounds[:,0], self.bounds[:,1], reverse=True), trf)
        for j in range(self.dim):
            self.tmodel.set_training_derivatives(qmc.scale(trx, self.bounds[:,0], self.bounds[:,1], reverse=True), trg[:,j], j)


    def evaluate(self, x, bounds, dir=0):
        
        trx = qmc.scale(self.trx, bounds[:,0], bounds[:,1], reverse=True)
        trf = self.model.training_points[None][0][1]

        m, n = trx.shape
        N = self.numer

        ans = -self.tmodel.predict_values(np.array([x]), self.model)*(10**n)

        # for batches, loop over already added points to prevent clustering
        for i in range(dir):
            ind = self.ntr + i
            work = x - trx[ind]
            ans += N/(np.dot(work, work) + 1e-10)

        return ans 


    def eval_grad(self, x, bounds, dir=0):
        
        trx = qmc.scale(self.trx, bounds[:,0], bounds[:,1], reverse=True)
        trf = self.model.training_points[None][0][1]

        m, n = trx.shape
        N = self.numer

        ans = -self.tmodel.predict_derivatives(np.array([x]), self.model)*(10**n)

        # for batches, loop over already added points to prevent clustering
        for i in range(dir):
            ind = self.ntr + i
            work = x - trx[ind]
            #dwork = np.eye(n)
            d2 = np.dot(work, work)
            dd2 = 2*work
            term = N/(d2 + 1e-10)
            ans += -N/((d2 + 1e-10)**2)*dd2
        
        return ans




    def pre_asopt(self, bounds, dir=0):
        
        trx = self.trx
        m, n = trx.shape

        # ### FD CHECK
        # h = 1e-6
        # zero = 0.5*np.ones([2])
        # step = 0.5*np.ones([2])
        # step[0] += h
        # ad = self.eval_grad(zero, bounds)
        # fd1 = (self.evaluate(step, bounds) - self.evaluate(zero, bounds))/h
        # step = 0.5*np.ones([2])
        # step[1] += h
        # fd2 = (self.evaluate(step, bounds) - self.evaluate(zero, bounds))/h
        # fd = [fd1[0], fd2[0]]
        # import pdb; pdb.set_trace()

        # ndir = 75
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
        #         F[i,j]  = self.evaluate(xi, bounds, dir=dir)

        # cs = plt.contour(Y, X, F, levels = np.linspace(np.min(F), 0., 25))
        # plt.colorbar(cs)
        # trxs = qmc.scale(self.trx, bounds[:,0], bounds[:,1], reverse=True)
        # plt.plot(trxs[0:-1,0], trxs[0:-1,1], 'bo')
        # plt.plot(trxs[-1,0], trxs[-1,1], 'ro')
        # plt.savefig("taylor_rc.png")

        # plt.clf()

        sampling = LHS(xlimits=bounds, criterion='m')
        ntries = self.options["multistart"]
        if(ntries > 1):
            xc = sampling(ntries)
        else: 
            xc = np.random.rand(n)*(bounds[:,1] - bounds[:,0]) + bounds[:,0]
            xc = np.array([xc])

        errs = np.zeros(ntries)
        xc_scale = qmc.scale(xc, bounds[:,0], bounds[:,1], reverse=True)
        for i in range(ntries):
            errs[i] = self.evaluate(xc_scale[i], bounds, dir=0)

        # For batches, set a numerator based on the scale of the error
        self.numer = abs(np.mean(errs))/100.

        return xc, bounds# + 0.001*self.dminmax+randvec, bounds



    def post_asopt(self, x, bounds, dir=0):

        self.trx = np.append(self.trx, np.array([x]), axis=0)

        return x










# Refine based on a first-order Taylor approximation using the gradient, with an exploration component
class TaylorExploreRefine(TaylorRefine):
    def __init__(self, model, grad, bounds, **kwargs):

        super().__init__(model, grad, bounds, **kwargs)


        
    def _init_options(self):
        super()._init_options()

        self.options.declare("objective","inv", types=str)

    def evaluate(self, x, bounds, dir=0):
        
        trx = qmc.scale(self.trx, bounds[:,0], bounds[:,1], reverse=True)
        trf = self.model.training_points[None][0][1]

        m, n = trx.shape
        N = self.numer

        # TODO: Need to normalize these in some way
        # error term
        ans = -self.tmodel.predict_values(np.array([x]), self.model)

        # distance term
        for i in range(m):
            work = x-trx[i]
            #dist = np.matmul(np.matmul(work, mwork), work)
            dist = np.linalg.norm(work)
            if(self.options["objective"] == "inv"):
                ans += N/(m*(dist + 1e-10))
            elif(self.options["objective"] == "abs"):
                ans += -dist/m

        return ans 


    def eval_grad(self, x, bounds, dir=0):
        
        trx = qmc.scale(self.trx, bounds[:,0], bounds[:,1], reverse=True)
        trf = self.model.training_points[None][0][1]

        m, n = trx.shape
        N = self.numer

        ans = -self.tmodel.predict_derivatives(np.array([x]), self.model)
            
        for i in range(m):
            work = x-trx[i]
            #dist = np.matmul(np.matmul(work, mwork), work)
            dist = np.linalg.norm(work)
            ddist = np.ones_like(work)
            if(self.options["objective"] == "inv"):
                ans += (-N/((dist + 1e-10)**2))*ddist/m
            elif(self.options["objective"] == "abs"):
                ans += -ddist/m

        return ans




    def pre_asopt(self, bounds, dir=0):
        
        trx = self.trx
        m, n = trx.shape

        ### FD CHECK
        # h = 1e-6
        # zero = 0.5*np.ones([2])
        # step = 0.5*np.ones([2])
        # step[0] += h
        # ad = self.eval_grad(zero, bounds)
        # fd = (self.evaluate(step, bounds) - self.evaluate(zero, bounds))/h
        # import pdb; pdb.set_trace()

        # ndir = 75
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

        # cs = plt.contour(Y, X, F, levels = np.linspace(-10,10.,25))
        # plt.colorbar(cs)
        # trxs = qmc.scale(self.trx, bounds[:,0], bounds[:,1], reverse=True)
        # plt.plot(trxs[0:-1,0], trxs[0:-1,1], 'bo')
        # plt.plot(trxs[-1,0], trxs[-1,1], 'ro')
        # plt.savefig("taylorexp_rc.png")

        # plt.clf()




        sampling = LHS(xlimits=bounds, criterion='m')
        ntries = self.options["multistart"]
        if(ntries > 1):
            xc = sampling(ntries)
        else: 
            xc = np.random.rand(n)*(bounds[:,1] - bounds[:,0]) + bounds[:,0]
            xc = np.array([xc])

        return xc, bounds# + 0.001*self.dminmax+randvec, bounds



    def post_asopt(self, x, bounds, dir=0):
        # Add new points to the distance term, but not the error term
        self.trx = np.append(self.trx, np.array([x]), axis=0)

        return x