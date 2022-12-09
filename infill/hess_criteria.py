import numpy as np
import copy

from matplotlib import pyplot as plt
from smt.utils.options_dictionary import OptionsDictionary
from smt.sampling_methods import LHS
from smt.surrogate_models import GEKPLS, KPLS, KRG
from pougrad import POUCV, POUError, POUErrorVol, POUMetric, POUSurrogate
from refinecriteria import ASCriteria
from scipy.linalg import lstsq, eig
from scipy.stats import qmc
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.optimize import Bounds
from sutils import innerMatrixProduct, linear, quadratic, quadraticSolve, quadraticSolveHOnly, symMatfromVec, estimate_pou_volume


"""
Refine based on a first-order Taylor approximation using the gradient. Pull 
Hessian estimates from the surrogate if available, to avoid unnecessary 
computation.
"""
class HessianRefine(ASCriteria):
    def __init__(self, model, grad, bounds, **kwargs):

        self.grad = grad
        self.bounds = bounds

        super().__init__(model, **kwargs)
        self.scaler = 0

        self.supports["obj_derivatives"] = True  
        
    def _init_options(self):
        declare = self.options.declare
        
        declare(
            "improve", 
            0, 
            types=int,
            desc="Number of points to generate before retraining"
        )

        declare(
            "multistart", 
            1, 
            types=int,
            desc="number of optimizations to try per point"
        )

        declare(
            "rscale", 
            0.5, 
            types=float,
            desc="scaling for error model hyperparameter"
        )

        declare(
            "neval", 
            3, 
            types=int,
            desc="number of closest points to evaluate hessian estimate"
        )


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

        trx = self.model.X_norma#model.training_points[None][0][0]
        trf = self.model.y_norma#training_points[None][0][1]
        trg = np.zeros_like(trx)
        if(isinstance(self.model, GEKPLS)):
            for j in range(self.dim):
                trg[:,j] = self.model.g_norma[:,j].flatten()
        else:
            trg = self.grad*(self.model.X_scale/self.model.y_std)


        self.trx = trx
        
        # Determine rho for the error model
        self.rho = self.options['rscale']*pow(self.ntr, 1./self.dim)

        # Generate kd tree for nearest neighbors lookup
        self.tree = KDTree(self.trx)

        # Check if the trained surrogate model has hessian data
        try:
            self.H = model.h

        except:
            indn = []
            nstencil = self.options["neval"]
            for i in range(self.ntr):
                dists, ind = self.tree.query(self.trx[i], nstencil)
                indn.append(ind)
            hess = []

            for i in range(self.ntr):
                Hh = quadraticSolveHOnly(self.trx[i,:], self.trx[indn[i][1:nstencil],:], \
                                         trf[i], trf[indn[i][1:nstencil]], \
                                         trg[i,:], trg[indn[i][1:nstencil],:])

                hess.append(np.zeros([self.dim, self.dim]))
                for j in range(self.dim):
                    for k in range(self.dim):
                        hess[i][j,k] = Hh[symMatfromVec(j,k,self.dim)]

            self.H = hess

        

    # Assumption is that the quadratic terms are the error
    def evaluate(self, x, bounds, dir=0):
        
        delta = self.model.options["delta"]

        # exhaustive search for closest sample point, for regularization
        D = cdist(np.array([x]), self.trx)
        mindist = min(D[0])

        numer = 0
        denom = 0

        for i in range(self.ntr):
            work = x - self.trx[i]
            dist = D[0][i] + delta#np.sqrt(D[0][i] + delta)
            local = 0.5*innerMatrixProduct(self.H[i], work)*self.dV[i]
            expfac = np.exp(-self.rho*(dist-mindist))
            numer += local*expfac
            denom += expfac

        y = numer/denom

        ans = -abs(y)

        # for batches, loop over already added points to prevent clustering
        for i in range(dir):
            ind = self.ntr + i
            work = x - self.trx[ind]
            ans += 1./(np.dot(work, work) + 1e-10)

        return ans 


    def eval_grad(self, x, bounds, dir=0):
        
        delta = self.model.options["delta"]

        # exhaustive search for closest sample point, for regularization
        D = cdist(np.array([x]), self.trx)
        mindist = min(D[0])

        numer = 0
        denom = 0
        dnumer = np.zeros(self.dim)
        ddenom = np.zeros(self.dim)
        dwork = np.ones(self.dim)

        for i in range(self.ntr):
            work = x - self.trx[i]
            dist = D[0][i] + delta#np.sqrt(D[0][i] + delta)
            ddist = work/D[0][i]
            local = 0.5*innerMatrixProduct(self.H[i], work)*self.dV[i]
            dlocal = np.dot(self.H[i], work)*self.dV[i]
            expfac = np.exp(-self.rho*(dist-mindist))
            dexpfac = -self.rho*expfac*ddist
            numer += local*expfac
            dnumer += local*dexpfac + dlocal*expfac
            denom += expfac
            ddenom += dexpfac

        y = numer/denom
        dy = (denom*dnumer - numer*ddenom)/(denom**2)

        ans = -np.sign(dy)*dy

        # for batches, loop over already added points to prevent clustering
        for i in range(dir):
            ind = self.ntr + i
            work = x - self.trx[ind]
            #dwork = np.eye(n)
            d2 = np.dot(work, work)
            dd2 = 2*work
            term = 1.0/(d2 + 1e-10)
            ans += -1.0/((d2 + 1e-10)**2)*dd2
        
        return ans




    def pre_asopt(self, bounds, dir=0):
        
        trx = self.trx
        m, n = trx.shape

        # factor in cell volume
        fakebounds = copy.deepcopy(bounds)
        fakebounds[:,0] = 0.
        fakebounds[:,1] = 1.
        self.dV = estimate_pou_volume(self.trx, fakebounds)


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
        # fd = [fd1, fd2]
        # import pdb; pdb.set_trace()
        if(self.options["print_rc_plots"]):

            if(n == 1):
                
                ndir = 200
                # x = np.linspace(bounds[0][0], bounds[0][1], ndir)
                # y = np.linspace(bounds[1][0], bounds[1][1], ndir)
                x = np.linspace(0., 1., ndir)
                F  = np.zeros([ndir]) 
                for i in range(ndir):
                    xi = np.zeros([1])
                    xi[0] = x[i]
                    F[i]  = -self.evaluate(xi, bounds, dir=dir)    
                if(self.ntr == 10):
                    self.scaler = np.max(F)  
                F /= np.abs(self.scaler)

                plt.rcParams['font.size'] = '16'
                ax = plt.gca()  
                plt.plot(x, F, label='Criteria')
                plt.xlim(-0.05, 1.05)
                plt.ylim(bottom=-0.015)
                plt.ylim(top=1.0)#np.min(F))
                trxs = self.trx#qmc.scale(self.trx, bounds[:,0], bounds[:,1], reverse=True)
                #plt.plot(trxs[0:-1,0], np.zeros(trxs[0:-1,0].shape[0]), 'bo')
                #plt.plot(trxs[-1,0], [0], 'ro')
                plt.plot(trxs[0:,0], np.zeros(trxs[0:,0].shape[0]), 'bo', label='Sample Locations')
                plt.legend(loc=0)
                plt.xlabel(r'$x_1$')
                plt.ylabel(r'$\psi_{\mathrm{Hess},%i}(x_1)$' % (self.ntr-10))
                plt.axvline(x[np.argmax(F)], color='k', linestyle='--', linewidth=1.2)
                plt.savefig(f"taylor_rc_1d_{self.ntr}.pdf", bbox_inches="tight")    
                plt.clf()

                xmod = np.linspace(bounds[0][0], bounds[0][1], ndir)
                trxmod = self.model.training_points[None][0][0]
                fmod = self.model.predict_values(xmod)
                from problem_picker import GetProblem
                origfunc = GetProblem('fuhgsh', 1)
                forig = origfunc(xmod)
                plt.plot(xmod, fmod, 'b', label='Model')
                plt.plot(xmod, forig, 'k', label='Original')
                plt.ylim(0,21)
                trf = self.model.training_points[None][0][1]
                plt.plot(trxmod, trf, 'bo', label='Sample Locations')
                plt.legend(loc=2)
                plt.xlabel(r'$x_1$')
                plt.ylabel(r'$\hat{f}_{POU,%i}(x_1)$' % (self.ntr-10))
                plt.axvline(xmod[np.argmax(F)], color='k', linestyle='--', linewidth=1.2)
                plt.savefig(f"taylor_md_1d_{self.ntr}.pdf", bbox_inches="tight")    
                plt.clf()
                import pdb; pdb.set_trace()

            if(n == 2):
                ndir = 75
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
                        F[i,j]  = self.evaluate(xi, bounds, dir=dir)    
                cs = plt.contourf(Y, X, F, levels = np.linspace(np.min(F), 0., 25))
                plt.colorbar(cs)
                trxs = self.trx #qmc.scale(self.trx, bounds[:,0], bounds[:,1], reverse=True)
                plt.plot(trxs[0:-1,0], trxs[0:-1,1], 'bo')
                plt.plot(trxs[-1,0], trxs[-1,1], 'ro')
                plt.savefig(f"taylor_rc_2d.pdf")    
                plt.clf()
        # import pdb; pdb.set_trace()


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








# SSA for POU Surrogate
class POUSSA(ASCriteria):
    def __init__(self, model, grad, bounds, **kwargs):

        self.grad = grad
        self.bounds = bounds

        super().__init__(model, **kwargs)

        self.supports["obj_derivatives"] = True  
        
    def _init_options(self):
        declare = self.options.declare
        
        declare(
            "improve", 
            0, 
            types=int,
            desc="Number of points to generate before retraining"
        )

        declare(
            "multistart", 
            1, 
            types=int,
            desc="number of optimizations to try per point"
        )
        
        declare(
            "eps", 
            0.01, 
            types=float,
            desc="non clustering parameter, minimum distance to existing samples"
        )



    def initialize(self, model=None, grad=None):
        
        # set up constraints
        self.condict = {
            "type":"ineq",
            "fun":self.eval_constraint,
            "args":[],
        }


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

        trx = self.model.X_norma#model.training_points[None][0][0]
        trf = self.model.y_norma#training_points[None][0][1]
        trg = np.zeros_like(trx)
        if(isinstance(self.model, GEKPLS)):
            for j in range(self.dim):
                trg[:,j] = self.model.g_norma[:,j].flatten()
        else:
            trg = self.grad*(self.model.X_scale/self.model.y_std)


        self.trx = trx
        

        # Generate kd tree for nearest neighbors lookup
        self.tree = KDTree(self.trx)

        self.cvmodel = POUCV(pmodel = self.model)
        self.cvmodel.options.update({"pmodel":self.model})

    def evaluate(self, x, bounds, dir=0):
        
        m = self.trx.shape[0]

        # compute CDM
        cdm = 0
        for i in range(m):
            work = x - self.trx[i]
            cdm += np.dot(work, work)
        
        y = cdm*self.cvmodel._predict_values(np.array([x]))**2

        ans = -abs(y)

        

        return ans 

    def eval_constraint(self, x, bounds, dir=0):
        
        m = self.trx.shape[0]

        xmin = self.tree.query(np.array([x]), 1)

        y = np.linalg.norm(x - xmin) - self.options["eps"]

        return y 


    def eval_grad(self, x, bounds, dir=0):
        
        # delta = self.model.options["delta"]

        # # exhaustive search for closest sample point, for regularization
        # D = cdist(np.array([x]), self.trx)
        # mindist = min(D[0])

        # numer = 0
        # denom = 0
        # dnumer = np.zeros(self.dim)
        # ddenom = np.zeros(self.dim)
        # dwork = np.ones(self.dim)

        # for i in range(self.ntr):
        #     work = x - self.trx[i]
        #     dist = D[0][i] + delta#np.sqrt(D[0][i] + delta)
        #     ddist = work/D[0][i]
        #     local = 0.5*innerMatrixProduct(self.H[i], work)*self.dV[i]
        #     dlocal = np.dot(self.H[i], work)*self.dV[i]
        #     expfac = np.exp(-self.rho*(dist-mindist))
        #     dexpfac = -self.rho*expfac*ddist
        #     numer += local*expfac
        #     dnumer += local*dexpfac + dlocal*expfac
        #     denom += expfac
        #     ddenom += dexpfac

        # y = numer/denom
        # dy = (denom*dnumer - numer*ddenom)/(denom**2)

        # ans = -np.sign(dy)*dy

        # # for batches, loop over already added points to prevent clustering
        # for i in range(dir):
        #     ind = self.ntr + i
        #     work = x - self.trx[ind]
        #     #dwork = np.eye(n)
        #     d2 = np.dot(work, work)
        #     dd2 = 2*work
        #     term = 1.0/(d2 + 1e-10)
        #     ans += -1.0/((d2 + 1e-10)**2)*dd2
        
        return None




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
        # fd = [fd1, fd2]
        # import pdb; pdb.set_trace()
        if(self.options["print_rc_plots"]):
            if(n == 1):
                ndir = 75
                # x = np.linspace(bounds[0][0], bounds[0][1], ndir)
                # y = np.linspace(bounds[1][0], bounds[1][1], ndir)
                x = np.linspace(0., 1., ndir)
                F  = np.zeros([ndir]) 
                for i in range(ndir):
                    xi = np.zeros([1])
                    xi[0] = x[i]
                    F[i]  = self.evaluate(xi, bounds, dir=dir)    
                plt.plot(x, F)
                plt.ylim(top=0.1)
                plt.ylim(bottom=np.min(F))
                trxs = self.trx#qmc.scale(self.trx, bounds[:,0], bounds[:,1], reverse=True)
                plt.plot(trxs[0:-1,0], np.zeros(trxs[0:-1,0].shape[0]), 'bo')
                plt.plot(trxs[-1,0], [0], 'ro')
                plt.savefig(f"cvssa_rc_1d.pdf")    
                plt.clf()
                # import pdb; pdb.set_trace()

            if(n == 2):
                ndir = 75
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
                        F[i,j]  = self.evaluate(xi, bounds, dir=dir)    
                cs = plt.contourf(Y, X, F, levels = np.linspace(np.min(F), 0., 25))
                plt.colorbar(cs)
                trxs = self.trx #qmc.scale(self.trx, bounds[:,0], bounds[:,1], reverse=True)
                plt.plot(trxs[0:-1,0], trxs[0:-1,1], 'bo')
                plt.plot(trxs[-1,0], trxs[-1,1], 'ro')
                plt.savefig(f"cvssa_rc_2d.pdf")    
                plt.clf()
        # import pdb; pdb.set_trace()


        sampling = LHS(xlimits=bounds, criterion='m')
        ntries = self.options["multistart"]
        if(ntries > 1):
            xc = sampling(ntries)
        else: 
            xc = np.random.rand(n)*(bounds[:,1] - bounds[:,0]) + bounds[:,0]
            xc = np.array([xc])


        return xc, bounds# + 0.001*self.dminmax+randvec, bounds



    def post_asopt(self, x, bounds, dir=0):

        self.trx = np.append(self.trx, np.array([x]), axis=0)

        return x






