import sys, os
import copy
import pickle
from mpi4py import MPI
sys.path.insert(1,"../surrogate")

import numpy as np
import matplotlib.pyplot as plt
from error import rmse, meane
from direct_gek import DGEK
from grbf import GRBF
from problem_picker import GetProblem
from smt.surrogate_models import KPLS, GEKPLS, KRG
#from smt.surrogate_models.rbf import RBF
from pougrad import POUSurrogate
from smt.sampling_methods import LHS
from scipy.stats import qmc

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

"""
Perform adaptive sampling and estimate error
"""
prob  = "arctan"    #problem
plot  = -1

# Conditions
dim = 2      #problem dimension
corr  = "matern32" #kriging correlation
poly  = "linear"    #kriging regression 
ncomp = dim
extra = 2           #gek extra points
nt0 = 10
ntr = 10
batch = 10
tval = 1e-2
t0 = [1e-2]
t0g = [tval]
tb = [1e-5, 20]
tbg = tb#[tval, tval+1e-4]#[tval, tval]
Nerr = 5000       #number of test points to evaluate the error
dx = 1e-4
nruns = int((ntr-nt0)/batch)+1

# Problem Settings
trueFunc = GetProblem(prob, dim)
xlimits = trueFunc.xlimits
sampling = LHS(xlimits=xlimits, criterion='m')

# Error
xtest = None 
ftest = None
testdata = None
xtest = sampling(Nerr)
ftest = trueFunc(xtest)
testdata = [xtest, ftest]


nt = np.linspace(nt0, ntr, nruns, dtype=int)

xtrainK = []
ftrainK = []
gtrainK = []
for j in range(nruns):
    xtrainK.append(sampling(nt[j]))
    ftrainK.append(trueFunc(xtrainK[j]))
    gtrainK.append(np.zeros([nt[j],dim]))
    for i in range(dim):
        gtrainK[j][:,i:i+1] = trueFunc(xtrainK[j],i)



modeldge = DGEK(xlimits=xlimits)
modeldge.options.update({"theta0":t0g})
modeldge.options.update({"theta_bounds":tbg})
modeldge.options.update({"corr":corr})
modeldge.options.update({"poly":poly})
modeldge.options.update({"n_start":1})
modeldge.options.update({"print_prediction":False})
# modeldge.set_training_values(xtrainK[0], ftrainK[0])
# for i in range(dim):
#     modeldge.set_training_derivatives(xtrainK[0], gtrainK[0][:,i:i+1], i)
# modeldge.train()

if(dim > 1):
    modelgek = GEKPLS(xlimits=xlimits)
    modelgek.options.update({"theta0":t0g})
    modelgek.options.update({"theta_bounds":tbg})
    modelgek.options.update({"n_comp":ncomp})
    modelgek.options.update({"extra_points":extra})
    modelgek.options.update({"corr":corr})
    modelgek.options.update({"poly":poly})
    modelgek.options.update({"n_start":5})
    modelgek.options.update({"delta_x":dx})
    modelgek.options.update({"print_prediction":False})
else:
    modelgek = KRG()
    #modelgek.options.update({"hyper_opt":"TNC"})
    modelgek.options.update({"theta0":t0g})
    modelgek.options.update({"theta_bounds":tbg})
    modelgek.options.update({"corr":corr})
    modelgek.options.update({"poly":poly})
    modelgek.options.update({"n_start":5})
    modelgek.options.update({"print_prediction":False})

modelkpl = KPLS()
#modelkpl.options.update({"hyper_opt":"TNC"})
modelkpl.options.update({"theta0":t0})
modelkpl.options.update({"theta_bounds":tb})
modelkpl.options.update({"n_comp":ncomp})
modelkpl.options.update({"corr":corr})
modelkpl.options.update({"poly":poly})
modelkpl.options.update({"n_start":5})
modelkpl.options.update({"print_prediction":False})

modelkrg = KRG()
#modelkrg.options.update({"hyper_opt":"TNC"})
modelkrg.options.update({"theta0":t0})
modelkrg.options.update({"theta_bounds":tb})
modelkrg.options.update({"corr":corr})
modelkrg.options.update({"poly":poly})
modelkrg.options.update({"n_start":5})
modelkrg.options.update({"print_prediction":False})

errgek = []
errkpl = []
errkrg = []
errdge = []
mgek = []
mkpl = []
mkrg = []
mdge = []
for j in range(nruns):
    if(dim > 1):
        modelgek.set_training_values(xtrainK[j], ftrainK[j])
        for i in range(dim):
            modelgek.set_training_derivatives(xtrainK[j], gtrainK[j][:,i:i+1], i)
        modelgek.train()
    else:
        nex = xtrainK[j].shape[0]
        xaug = np.zeros([nex, 1])
        faug = np.zeros([nex, 1])
        for k in range(nex):
            xaug[k] = xtrainK[j][k] + dx
            faug[k] = ftrainK[j][k] + dx*gtrainK[j][k]
        xtot = np.append(xtrainK[j], xaug, axis=0)
        ftot = np.append(ftrainK[j], faug, axis=0)
        modelgek.set_training_values(xtot, ftot)
        modelgek.train()
    mgek.append(copy.deepcopy(modelgek))
    print("GEK Done")
    modelkpl.set_training_values(xtrainK[j], ftrainK[j])
    modelkpl.train()
    mkpl.append(copy.deepcopy(modelkpl))
    print("KPL Done")
    modelkrg.set_training_values(xtrainK[j], ftrainK[j])
    modelkrg.train()
    errgek.append(rmse(modelgek, trueFunc, N=Nerr, xdata=xtest, fdata=ftest))
    errkpl.append(rmse(modelkpl, trueFunc, N=Nerr, xdata=xtest, fdata=ftest))
    errkrg.append(rmse(modelkrg, trueFunc, N=Nerr, xdata=xtest, fdata=ftest))
    mkrg.append(copy.deepcopy(modelkrg))
    print("KRG Done")

    # modeldge.set_training_values(xtrainK[j], ftrainK[j])
    # for i in range(dim):
    #     modeldge.set_training_derivatives(xtrainK[j], gtrainK[j][:,i:i+1], i)
    # modeldge.train()
    # errdge.append(rmse(modeldge, trueFunc, N=Nerr, xdata=xtest, fdata=ftest))
    # mdge.append(copy.deepcopy(modeldge))
    # print("DGEK Done")

print("IGEK")
print(errgek)
print(modelgek.optimal_theta)
print("KPL")
print(errkpl)
print(modelkpl.optimal_theta)
print("KRG")
print(errkrg)
print(modelkrg.optimal_theta)
# print("DGEK")
# print(errdge)
# print(modeldge.optimal_theta)

if(nruns > 2):

    plt.loglog(nt, errgek, "m-", label=f'GEK')
    plt.loglog(nt, errkpl, 'b-', label='KPL')
    plt.loglog(nt, errkrg, 'r-', label='KRG')
    # plt.loglog(nt, errdge, 'c-', label='DGEK')
    plt.xlabel("Number of samples")
    plt.ylabel("NRMSE")
    plt.grid()
    plt.legend(loc=1)
    plt.savefig(f"./gek_test_err.png", bbox_inches="tight")
    plt.clf()


# # Plot points
if(dim == 1):  

    ndir = 75
    xlimits = trueFunc.xlimits
    x = np.linspace(xlimits[0][0], xlimits[0][1], ndir)

    Zgek = np.zeros([ndir, ndir])
    Zkpl = np.zeros([ndir, ndir])
    Zkrg = np.zeros([ndir, ndir])
    Zdge = np.zeros([ndir, ndir])
    Fgek = np.zeros([ndir, ndir])
    Fkpl = np.zeros([ndir, ndir])
    Fkrg = np.zeros([ndir, ndir])
    Fdge = np.zeros([ndir, ndir])
    TF = np.zeros([ndir, ndir])

    for i in range(ndir):
        xi = np.zeros([1,1])
        xi[0] = x[i]
        TF[i] = trueFunc(xi)
        Fgek[i] = mgek[plot].predict_values(xi)
        Fkpl[i] = mkpl[plot].predict_values(xi)
        Fkrg[i] = mkrg[plot].predict_values(xi)
        Fdge[i] = mdge[plot].predict_values(xi)
        Zgek[i] = abs(Fgek[i] - TF[i])
        Zkpl[i] = abs(Fkpl[i] - TF[i])
        Zkrg[i] = abs(Fkrg[i] - TF[i])
        Zdge[i] = abs(Fdge[i] - TF[i])

    # Plot Non-Adaptive Error
    plt.plot(x, TF, "-k")
    plt.plot(x, Fgek, "-m", label=f'IGEK')
    plt.plot(x, Fkpl, "-b", label=f'KPL')
    plt.plot(x, Fkrg, "-r", label=f'KRG')
    plt.plot(x, Fdge, "-c", label=f'DGEK')
    plt.xlabel(r"$x$")
    plt.ylabel(r"$f$")
    #plt.legend(loc=1)
    plt.plot(xtrainK[plot][:,0], ftrainK[plot][:], "o", fillstyle='full', markerfacecolor='b', markeredgecolor='b', label='Initial Samples')

    plt.savefig(f"./gek_1dplot.png", bbox_inches="tight")
    plt.clf()

    # Plot Non-Adaptive Error
    plt.plot(x, Zgek, "-m", label=f'IGEK')
    plt.plot(x, Zkpl, "-b", label=f'KPL')
    plt.plot(x, Zkrg, "-r", label=f'KRG')
    plt.plot(x, Zdge, "-c", label=f'DGEK')
    plt.xlabel(r"$x$")
    plt.ylabel(r"$err$")
    #plt.legend(loc=1)
    plt.plot(xtrainK[plot][:,0], np.zeros(xtrainK[plot].shape[0]), "o", fillstyle='full', markerfacecolor='b', markeredgecolor='b', label='Initial Samples')

    plt.savefig(f"./gek_1derr.png", bbox_inches="tight")

    plt.clf()

# # Plot points
if(dim == 2):  
    # Plot Error Contour
    #Contour
    ndir = 75
    xlimits = trueFunc.xlimits
    x = np.linspace(xlimits[0][0], xlimits[0][1], ndir)
    y = np.linspace(xlimits[1][0], xlimits[1][1], ndir)

    X, Y = np.meshgrid(x, y)
    Za = np.zeros([ndir, ndir])
    Va = np.zeros([ndir, ndir])
    V0 = np.zeros([ndir, ndir])
    Zgek = np.zeros([ndir, ndir])
    Zkpl = np.zeros([ndir, ndir])
    Zkrg = np.zeros([ndir, ndir])
    Zdge = np.zeros([ndir, ndir])
    Vk = np.zeros([ndir, ndir])
    Z0 = np.zeros([ndir, ndir])
    F  = np.zeros([ndir, ndir])
    TF = np.zeros([ndir, ndir])

    for i in range(ndir):
        for j in range(ndir):
            xi = np.zeros([1,2])
            xi[0,0] = x[i]
            xi[0,1] = y[j]
            TF[j,i] = trueFunc(xi)
            # Zgek[j,i] = abs(modelgek.predict_values(xi) - TF[j,i])
            Zkpl[j,i] = abs(modelkpl.predict_values(xi) - TF[j,i])
            Zkrg[j,i] = abs(modelkrg.predict_values(xi) - TF[j,i])
            Zdge[j,i] = abs(modeldge.predict_values(xi) - TF[j,i])

    # Plot Non-Adaptive Error
    cs = plt.contour(X, Y, Zkpl, levels = 40)
    plt.colorbar(cs, aspect=20)
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    #plt.legend(loc=1)
    plt.plot(xtrainK[-1][:,0], xtrainK[-1][:,1], "o", fillstyle='full', markerfacecolor='b', markeredgecolor='b', label='Initial Samples')
    plt.savefig(f"./kpl_errcona.png", bbox_inches="tight")

    plt.clf()

    plt.contour(X, Y, Zgek, levels = cs.levels)
    plt.colorbar(cs, )
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    #plt.legend(loc=1)
    plt.plot(xtrainK[-1][:,0], xtrainK[-1][:,1], "o", fillstyle='full', markerfacecolor='b', markeredgecolor='b', label='Initial Samples')
    plt.savefig(f"./gek_errcona.png", bbox_inches="tight")

    plt.clf()

    

    plt.contour(X, Y, Zkrg, levels = cs.levels)
    plt.colorbar(cs, )
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    #plt.legend(loc=1)
    plt.plot(xtrainK[-1][:,0], xtrainK[-1][:,1], "o", fillstyle='full', markerfacecolor='b', markeredgecolor='b', label='Initial Samples')
    plt.savefig(f"./krg_errcona.png", bbox_inches="tight")

    plt.clf()

    plt.contour(X, Y, Zdge, levels = cs.levels)
    plt.colorbar(cs, )
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    #plt.legend(loc=1)
    plt.plot(xtrainK[-1][:,0], xtrainK[-1][:,1], "o", fillstyle='full', markerfacecolor='b', markeredgecolor='b', label='Initial Samples')
    plt.savefig(f"./dgek_errcona.png", bbox_inches="tight")

    plt.clf()

