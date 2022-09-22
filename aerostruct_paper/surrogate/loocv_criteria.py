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


# SFCVT
class POUSFCVT(ASCriteria):
    def __init__(self, model, grad, bounds, **kwargs):

        self.grad = grad
        self.bounds = bounds

        super().__init__(model, **kwargs)

        self.supports["obj_derivatives"] = False  
        
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

        # min dist constraint
        self.S = 0        

        # Generate kd tree for nearest neighbors lookup
        self.tree = KDTree(self.trx)

        self.cvmodel = POUCV(pmodel = self.model)
        self.cvmodel.options.update({"pmodel":self.model})

    def evaluate(self, x, bounds, dir=0):
        
     
        y = self.cvmodel._predict_values(np.array([x]))

        ans = -abs(y)

        

        return ans 

    def eval_constraint(self, x, bounds, dir=0):
        
        m = self.trx.shape[0]

        xmin = self.tree.query(np.array([x]), 1)[0]

        y = xmin - self.S#options["eps"]
        # import pdb; pdb.set_trace()
        return y 





    def pre_asopt(self, bounds, dir=0):
        
        trx = self.trx
        m, n = trx.shape

        mindists = np.sort(squareform(pdist(trx)))
        self.S = 0.5*np.max(mindists[:,1])
        # import pdb; pdb.set_trace()


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
                ndir = 400
                # x = np.linspace(bounds[0][0], bounds[0][1], ndir)
                # y = np.linspace(bounds[1][0], bounds[1][1], ndir)
                x = np.linspace(0., 1., ndir)
                F  = np.zeros([ndir]) 
                for i in range(ndir):
                    xi = np.zeros([1])
                    xi[0] = x[i]
                    F[i]  = self.evaluate(xi, bounds, dir=dir)    
                F /= np.abs(np.min(F))

                plt.rcParams['font.size'] = '16'
                ax = plt.gca()  
                plt.plot(x, F, label='Criteria')
                plt.xlim(-0.05, 1.05)
                plt.ylim(top=0.015)
                plt.ylim(bottom=-1.0)#np.min(F))
                trxs = self.trx#qmc.scale(self.trx, bounds[:,0], bounds[:,1], reverse=True)
                #plt.plot(trxs[0:-1,0], np.zeros(trxs[0:-1,0].shape[0]), 'bo')
                #plt.plot(trxs[-1,0], [0], 'ro')
                plt.plot(trxs[0:,0], np.zeros(trxs[0:,0].shape[0]), 'bo', label='Sample Locations')
                plt.legend(loc=3)
                plt.xlabel(r'$x_1$')
                plt.ylabel(r'$-\mathrm{RC}_{\mathrm{CV},%i}(x_1)$' % (self.ntr-10))
                wheret = np.full([ndir], True)
                for i in range(self.ntr):
                    ax.fill_betweenx([-1,0], trxs[i]-self.S, trxs[i]+self.S, color='r', alpha=0.2)
                    for j in range(ndir):
                        if(x[j] > trxs[i]-self.S and x[j] < trxs[i]+self.S):
                            wheret[j] = False
                valid = np.where(wheret)[0]
                plt.axvline(x[valid[np.argmin(F[valid])]], color='k', linestyle='--', linewidth=1.)
                plt.savefig(f"cvsf_rc_1d_{self.ntr}.pdf", bbox_inches="tight")  
                plt.clf()
                #import pdb; pdb.set_trace()

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
                plt.savefig(f"cvsf_rc_2d.pdf")    
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






