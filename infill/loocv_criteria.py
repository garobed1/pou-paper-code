import numpy as np
import copy

from matplotlib import pyplot as plt
from smt.utils.options_dictionary import OptionsDictionary
from smt.sampling_methods import LHS
from smt.surrogate_models import GEKPLS, KPLS, KRG
from surrogate.pougrad import POUCV
from infill.refinecriteria import ASCriteria
from scipy.spatial import KDTree
from scipy.stats import qmc
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.optimize import NonlinearConstraint
from utils.sutils import convert_to_smt_grads, print_rc_plots


# SFCVT
class POUSFCVT(ASCriteria):
    def __init__(self, model, grad, bounds, **kwargs):

        self.grad = grad
        self.bounds = bounds

        super().__init__(model, **kwargs)
        self.scaler = 0

        self.supports["obj_derivatives"] = False  
        self.supports["uses_constraints"] = True  
        
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
            print_rc_plots(n, bounds, "POUSFCVT", self)
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





"""
SFCVT, Original Unmodified Version for any surrogate
"""
class SFCVT(ASCriteria):
    def __init__(self, model, grad, bounds, **kwargs):
        super().__init__(model, **kwargs)
        self.S = None
        self.cvsurr = None # KRG surrogate used to interpolate between cv values

        self.grad = grad
        self.bounds = bounds

        super().__init__(model, **kwargs)
        self.scaler = 0

        self.supports["uses_constraints"] = True  

    def _init_options(self):
        declare = self.options.declare
        
        declare("approx", False, types=bool)
        
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

        # self.condict = {
        #     "type":"ineq",
        #     "fun":self.eval_constraint,
        #     "args":[],
        # }

        # in case the model gets updated externally by getxnew
        if(model is not None):
            self.model = copy.deepcopy(model)
            kx = 0
            self.dim = self.model.training_points[None][kx][0].shape[1]
            self.ntr = self.model.training_points[None][kx][0].shape[0]

        if(grad is not None):
            self.grad = grad

        # set up constraints
        self.condict = NonlinearConstraint(self.eval_constraint,
                        lb = 0., ub = np.inf)
        
        # Compute the error at each left-out point, don't bother storing left out models it's a waste
        loosm = None
        trx_true = self.model.training_points[None][0][0]
        abs_err_sc = np.zeros([self.ntr, 1])
        for i in range(self.ntr):
            loosm = copy.deepcopy(self.model)
            loosm.options.update({"print_global":False})

            kx = 0

            # Give each LOO model its training data, and retrain if not approximating
            trx = copy.deepcopy(loosm.training_points[None][kx][0])
            trf = copy.deepcopy(loosm.training_points[None][kx][1])
            trg = None
                
            trx = np.delete(trx, i, 0)
            trf = np.delete(trf, i, 0)
            
            loosm.set_training_values(trx, trf)
            if(loosm.supports["training_derivatives"]):
                trg = convert_to_smt_grads(self.model)
                trg = np.delete(trg, i, 0)
                convert_to_smt_grads(loosm, trx, trg)

            # if(self.options["approx"] == False):
            loosm.train()

            # now compute scaled absolute error 
            # import pdb; pdb.set_trace()
            M_i = self.model.predict_values(trx_true[i:i+1,:])[0]
            M_i_m = loosm.predict_values(trx_true[i:i+1,:])[0]
            abs_err_sc[i,:] = abs((M_i - M_i_m)/M_i)

        # min dist constraint
        self.S = 0

        # now generate the Kriging model
        self.cvsurr = KRG()
        self.cvsurr.options.update({"print_global":False})
        self.cvsurr.set_training_values(trx_true, abs_err_sc)
        self.cvsurr.train()

    def evaluate(self, x, bounds, dir=0):
        
        if(len(x.shape) != 2):
            x = np.array([x])

        # just evaluate the cvsurr, negative to maximize
        ans = -self.cvsurr.predict_values(x).flatten()[0]

        return ans # to work with optimizers

    def pre_asopt(self, bounds, dir=0):
        
        # NOTE: DO NOT USE X_norma HERE FOR GEK, IT INCLUDES EXTRA POINTS
        # trx = self.model.X_norma
        trx = qmc.scale(self.model.training_points[None][0][0], bounds[:,0], bounds[:,1], reverse=True)
        n = trx.shape[1]
        mindists = np.sort(squareform(pdist(trx)))
        self.S = 0.5*np.max(mindists[:,1])

        if(self.options["print_rc_plots"]):
            print_rc_plots(trx.shape[1], bounds, "SFCVT", self)

        sampling = LHS(xlimits=bounds, criterion='m')
        ntries = self.options["multistart"]
        if(ntries > 1):
            xc = sampling(ntries)
        else: 
            xc = np.random.rand(n)*(bounds[:,1] - bounds[:,0]) + bounds[:,0]
            xc = np.array([xc])


        return xc, bounds

    def post_asopt(self, x, bounds, dir=0):

        return x
        

    def eval_constraint(self, x, dir=0):
        t0 = qmc.scale(self.model.training_points[None][0][0], self.bounds[:,0], self.bounds[:,1], reverse=True)

        con = np.zeros(self.ntr)
        for i in range(self.ntr):
            con[i] = np.linalg.norm(x - t0[i])

        # import pdb; pdb.set_trace()
        return con - self.S














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






