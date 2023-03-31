import numpy as np
import openmdao.api as om
from utils.error import stat_comp
from utils.sutils import convert_to_smt_grads
from optimization.robust_objective import RobustSampler
import copy
"""
Compute some statistical measure of a model at a given design
"""
class StatCompComponent(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('name', default='name', desc="name, used for plot saving path")
        self.options.declare('surrogate', default=None, desc="surrogate model of the func. If not None, sampler is used for training data")


        self.options.declare('full_space', default=False, desc="if true, construct surrogate over the full space and add points successively. if not, construct surrogate over the uncertain space")
        
        self.options.declare('sampler', desc="object that tracks samples of the func")
        self.options.declare('pdfs', desc="prob dists of inputs")
        self.options.declare('func', desc="SMT Func handle", recordable=False)
        self.options.declare('eta', desc="mean to stdev ratio")
        self.options.declare('stat_type', desc="Type of robust function to compute")
        self.options.declare('print_surr_plots', default=False, desc="Print plots of 1D or 2D surrogates")

        self.surrogate = None
        self.sampler = None
        self.func = None
        self.eta = None
        self.stat_type = None
        self.pdfs = None

        self.jump = 0

        self.xtrain_act = None
        self.gtrain_act = None
        self.ftrain_act = None

        self.first_train = False

        self.xps = []
        self.objs = []
        self.func_calls = []

        self.surr_x = []
        self.surr_f = []

    def setup(self):
        
        self.surrogate = self.options["surrogate"]
        self.sampler = self.options["sampler"]
        self.func = self.options["func"]
        self.eta = self.options["eta"]
        self.stat_type = self.options["stat_type"]
        self.pdfs = self.options["pdfs"]

        # check the dimension of the surrogate
        # if self.surrogate:
        #     if

        # inputs
        self.add_input('x_d', shape=1,
                              desc='Current design point')
        
        self.add_output('musigma', shape=1,
                                   desc='mean + stdev at design point')

        self.declare_partials('*','*')

    def compute(self, inputs, outputs):

        x = inputs['x_d']
        eta = self.eta

        eval_sampler = self.sampler
        eval_N = self.sampler.N

        if self.jump == 0:
            self.jump = self.sampler.N

        # set surrogate computation
        if self.surrogate:
            eval_sampler = None
            eval_N = 5000*self.sampler.x_u_dim

        self.pdfs[1] = x
        moved = self.sampler.set_design(np.array([x]))
        self.sampler.generate_uncertain_points(self.sampler.N)
        
        call_track = copy.deepcopy(self.first_train)

        # train the surrogate if available AND we have moved AND we even bothered to generate points
        if self.surrogate and (moved or not self.first_train):
            # TODO: find a way to add samples
            # Choose between surrogate over just x_u,
            # or build full surrogate, add points successively

            # actual computation of training data
            xtrain = self.sampler.current_samples['x']
            if self.sampler.func_computed:
                ftrain = self.sampler.current_samples['f']
            else:
                ftrain = self.func(xtrain)
                self.sampler.set_evaluated_func(ftrain)

            if self.sampler.grad_computed:
                gtrain = self.sampler.current_samples['g']
            else:
                gtrain = convert_to_smt_grads(self.func, xtrain)
                self.sampler.set_evaluated_grad(gtrain)


            # train model
            # slice dimensions here
            if not self.options["full_space"]:
                self.xtrain_act = xtrain[:,self.sampler.x_u_ind]
                self.gtrain_act = gtrain[:,self.sampler.x_u_ind]
                self.ftrain_act = ftrain
            elif self.xtrain_act is None:
                self.xtrain_act = xtrain
                self.gtrain_act = gtrain
                self.ftrain_act = ftrain
            else:
                self.xtrain_act = np.append(self.xtrain_act, xtrain, axis=0)
                self.gtrain_act = np.append(self.gtrain_act, gtrain, axis=0)
                self.ftrain_act = np.append(self.ftrain_act, ftrain, axis=0)

            self.surrogate.set_training_values(self.xtrain_act, self.ftrain_act)
            convert_to_smt_grads(self.surrogate, self.xtrain_act, self.gtrain_act)
            self.surrogate.train()
            
            self.first_train = True


            if(self.options["print_surr_plots"]):
                import matplotlib.pyplot as plt
                n = self.sampler.x_u_dim
                if self.options["full_space"]:
                    n += self.sampler.x_d_dim
                # if(n == 1):

                #     ndir = 200
                #     # x = np.linspace(bounds[0][0], bounds[0][1], ndir)
                #     # y = np.linspace(bounds[1][0], bounds[1][1], ndir)
                #     x = np.linspace(0., 1., ndir)
                #     F  = np.zeros([ndir]) 
                #     for i in range(ndir):
                #         xi = np.zeros([1])
                #         xi[0] = x[i]
                #         F[i]  = -self.evaluate(xi, bounds, dir=dir)    
                #     if(self.ntr == 10):
                #         self.scaler = np.max(F)  
                #     F /= np.abs(self.scaler)

                #     plt.rcParams['font.size'] = '16'
                #     ax = plt.gca()  
                #     plt.plot(x, F, label='Criteria')
                #     plt.xlim(-0.05, 1.05)
                #     plt.ylim(bottom=-0.015)
                #     plt.ylim(top=1.0)#np.min(F))
                #     trxs = self.trx#qmc.scale(self.trx, bounds[:,0], bounds[:,1], reverse=True)
                #     #plt.plot(trxs[0:-1,0], np.zeros(trxs[0:-1,0].shape[0]), 'bo')
                #     #plt.plot(trxs[-1,0], [0], 'ro')
                #     plt.plot(trxs[0:,0], np.zeros(trxs[0:,0].shape[0]), 'bo', label='Sample Locations')
                #     plt.legend(loc=0)
                #     plt.xlabel(r'$x_1$')
                #     plt.ylabel(r'$\psi_{\mathrm{Hess},%i}(x_1)$' % (self.ntr-10))
                #     plt.axvline(x[np.argmax(F)], color='k', linestyle='--', linewidth=1.2)
                #     plt.savefig(f"taylor_rc_1d_{self.ntr}.pdf", bbox_inches="tight")    
                #     plt.clf()

                    # xmod = np.linspace(bounds[0][0], bounds[0][1], ndir)
                    # trxmod = self.model.training_points[None][0][0]
                    # fmod = self.model.predict_values(xmod)
                    # from problem_picker import GetProblem
                    # origfunc = GetProblem('fuhgsh', 1)
                    # forig = origfunc(xmod)
                    # plt.plot(xmod, fmod, 'b', label='Model')
                    # plt.plot(xmod, forig, 'k', label='Original')
                    # plt.ylim(0,21)
                    # trf = self.model.training_points[None][0][1]
                    # plt.plot(trxmod, trf, 'bo', label='Sample Locations')
                    # plt.legend(loc=2)
                    # plt.xlabel(r'$x_1$')
                    # plt.ylabel(r'$\hat{f}_{POU,%i}(x_1)$' % (self.ntr-10))
                    # plt.axvline(xmod[np.argmax(F)], color='k', linestyle='--', linewidth=1.2)
                    # plt.savefig(f"taylor_md_1d_{self.ntr}.pdf", bbox_inches="tight")    
                    # plt.clf()
                    # import pdb; pdb.set_trace()

                if(n == 2):
                    ndir = 100
                    xlimits = self.surrogate.options["bounds"]
                    xp = np.linspace(xlimits[0][0], xlimits[0][1], ndir)
                    y = np.linspace(xlimits[1][0], xlimits[1][1], ndir)  
                    X, Y = np.meshgrid(xp, y)
                    F  = np.zeros([ndir, ndir]) 
                    for i in range(ndir):
                        for j in range(ndir):
                            xi = np.zeros([2])
                            xi[0] = xp[i]
                            xi[1] = y[j]
                            F[i,j]  = self.surrogate.predict_values(np.array([xi]))
                    cs = plt.contourf(Y, X, F, levels = 25) #, levels = np.linspace(np.min(F), 0., 25)
                    plt.colorbar(cs)
                    trxs = self.surrogate.training_points[None][0][0] #qmc.scale(self.trx, bounds[:,0], bounds[:,1], reverse=True)
                    plt.plot(trxs[0:-self.jump,1], trxs[0:-self.jump,0], 'bo')
                    plt.plot(trxs[-self.jump:,1], trxs[-self.jump:,0], 'ro')
                    path = self.options["name"]
                    plt.savefig(f"./{path}/subprob_surr_2d_iter_{self.sampler.iter_max}.pdf")    
                    plt.clf()


                    # import pdb; pdb.set_trace()
                    # plot robust func
                    # ndir = 150
                    # yobj = np.zeros([ndir])
                    # for j in range(ndir):
                    #     pdfs_plot = copy.deepcopy(self.pdfs)
                    #     pdfs_plot[1] = xp[j]
                    #     yobj[j] = stat_comp(self.surrogate, self.func, 
                    #             stat_type=self.stat_type, 
                    #             pdfs=pdfs_plot, 
                    #             N=eval_N)[0]

                    # Plot original function
                    # cs = plt.plot(xp, y)
                    # plt.xlabel(r"$x_d$")
                    # plt.ylabel(r"$\mu_f(x_d)$")
                    # # plt.axvline(x_init, color='k', linestyle='--', linewidth=1.2)
                    # # plt.axvline(x_opt_1, color='r', linestyle='--', linewidth=1.2)
                    # #plt.legend(loc=1)
                    # plt.savefig(f"./{path}/subprob_surr_2d_obj_{self.sampler.iter_max}.pdf", bbox_inches="tight")
                    """
                    append data to a self. list so we can plot obj surrogate converging
                    """
                    # self.surr_x.append(xp)
                    # self.surr_f.append(yobj)


        res = stat_comp(self.surrogate, self.func, 
                                stat_type=self.stat_type, 
                                pdfs=self.pdfs, 
                                N=eval_N,
                                xdata=eval_sampler)
        fm = res[0]
        fs = res[1]

        # if(self.options["print_surr_plots"]):
        #     import matplotlib.pyplot as plt
        #     n += self.sampler.x_d_dim
            
        #     if(n == 1):
        #         plt.plot(x, fm)
        #         plt.savefig(f"./{path}/subprob_surr_2d_obj_{self.sampler.iter_max}.pdf", bbox_inches="tight")
        #         plt.clf()
        self.xps.append(copy.deepcopy(x[0]))
        self.objs.append(fm)
        if len(self.func_calls):
            if self.surrogate and not (moved or not call_track):
                self.func_calls.append(self.func_calls[-1])
            else:
                self.func_calls.append(self.func_calls[-1] + self.sampler.current_samples['x'].shape[0])
        else:
            self.func_calls.append(self.sampler.current_samples['x'].shape[0])


        outputs['musigma'] = eta*fm + (1-eta)*fs

    def compute_partials(self, inputs, partials):

        x = inputs['x_d']
        eta = self.eta

        eval_sampler = self.sampler
        eval_N = self.sampler.N

        # set surrogate computation
        if self.surrogate:
            eval_sampler = None
            eval_N = 5000*self.sampler.x_u_dim

        self.pdfs[1] = x
        moved = self.sampler.set_design(np.array([x]))
        self.sampler.generate_uncertain_points(self.sampler.N)

        # train the surrogate if available AND we have moved
        if self.surrogate and (moved or not self.first_train):
            # TODO: find a way to add samples
            # Choose between surrogate over just x_u,
            # or build full surrogate, add points successively

            # actual computation of training data
            xtrain = self.sampler.current_samples['x']
            ftrain = self.func(xtrain)
            gtrain = convert_to_smt_grads(self.func, xtrain)

            # give computed data to sampler
            self.sampler.set_evaluated_func(ftrain)
            self.sampler.set_evaluated_grad(gtrain)

            # train model
            # slice dimensions here
            if not self.options["full_space"]:
                self.xtrain_act = xtrain[:,self.sampler.x_u_ind]
                self.gtrain_act = gtrain[:,self.sampler.x_u_ind]
                self.ftrain_act = ftrain
            elif self.xtrain_act == None:
                self.xtrain_act = xtrain
                self.gtrain_act = gtrain
                self.ftrain_act = ftrain
            else:
                self.xtrain_act = np.append(self.xtrain_act, xtrain, axis=0)
                self.gtrain_act = np.append(self.gtrain_act, gtrain, axis=0)
                self.ftrain_act = np.append(self.ftrain_act, ftrain, axis=0)

            self.surrogate.set_training_values(self.xtrain_act, self.ftrain_act)
            convert_to_smt_grads(self.surrogate, self.xtrain_act, self.gtrain_act)
            self.surrogate.train()
            self.first_train = True

        gres = stat_comp(self.surrogate, self.func, 
                                get_grad=True, 
                                stat_type=self.stat_type, 
                                pdfs=self.pdfs, 
                                N=eval_N,
                                xdata=eval_sampler)
        gm = gres[0]
        gs = gres[1]

        partials['musigma','x_d'] = eta*gm + (1-eta)*gs

    def get_fidelity(self):
        # return current number of samples
        return self.sampler.current_samples['x'].shape[0]
    
    def refine_model(self, N):

        if isinstance(N, dict):
            self.sampler.add_data(N, replace_current=True)

        else:
            self.sampler.N += N
            self.jump = N
            self.sampler.refine_uncertain_points(N)

        # reset training since the design is the same and we haven't moved
        self.first_train = False