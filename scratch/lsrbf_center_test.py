'''

Test convergence of least-squares method when increasing the number of basis centers

'''
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
from lsrbf import LSRBF
from shock_problem import ImpingingShock
from example_problems import  QuadHadamard, MultiDimJump, MultiDimJumpTaper, FuhgP8, FuhgP9, FuhgP10, Peaks2D
from smt.problems import Branin, Sphere, LpNorm, Rosenbrock, WaterFlow, WeldedBeam, RobotArm, CantileverBeam, TensorProduct
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
prob  = "branin"    #problem
plot  = -1

# Conditions
dim = 2      #problem dimension
corr  = "squar_exp" #kriging correlation
poly  = "linear"    #kriging regression 
ncomp = dim
tcomp = False
extra = 2           #gek extra points
nc0 = 10*dim
ncr = 50*dim
ntrain = 30*dim
batch = 10
tval = 1e-2
t0 = [1., 3e-3]
t0g = [tval]#[tval, tval]
tb = [1e-6, 20.]
tbg = tb#[tval, tval]#[tval, tval]
Nerr = 5000       #number of test points to evaluate the error
dx = 1e-4
nruns = int((ncr-nc0)/batch)+1

# Problem Settings
alpha = 8.       #arctangent jump strength
if(prob == "arctan"):
    trueFunc = MultiDimJump(ndim=dim, alpha=alpha)
elif(prob == "arctantaper"):
    trueFunc = MultiDimJumpTaper(ndim=dim, alpha=alpha)
elif(prob == "rosenbrock"):
    trueFunc = Rosenbrock(ndim=dim)
elif(prob == "branin"):
    trueFunc = Branin(ndim=dim)
elif(prob == "sphere"):
    trueFunc = Sphere(ndim=dim)
elif(prob == "fuhgp8"):
    trueFunc = FuhgP8(ndim=dim)
elif(prob == "fuhgp9"):
    trueFunc = FuhgP9(ndim=dim)
elif(prob == "fuhgp10"):
    trueFunc = FuhgP10(ndim=dim)
elif(prob == "waterflow"):
    trueFunc = WaterFlow(ndim=dim)
elif(prob == "weldedbeam"):
    trueFunc = WeldedBeam(ndim=dim)
elif(prob == "robotarm"):
    trueFunc = RobotArm(ndim=dim)
elif(prob == "cantilever"):
    trueFunc = CantileverBeam(ndim=dim)
elif(prob == "hadamard"):
    trueFunc = QuadHadamard(ndim=dim)
elif(prob == "peaks"):
    trueFunc = Peaks2D(ndim=dim)
elif(prob == "tensorexp"):
    trueFunc = TensorProduct(ndim=dim, func="gaussian")
elif(prob == "shock"):
    xlimits = np.zeros([dim,2])
    xlimits[0,:] = [23., 27.]
    xlimits[1,:] = [0.36, 0.51]
    trueFunc = ImpingingShock(ndim=dim, input_bounds=xlimits, comm=MPI.COMM_SELF)
else:
    raise ValueError("Given problem not valid.")
xlimits = trueFunc.xlimits
sampling = LHS(xlimits=xlimits, criterion='m')
csampling = LHS(xlimits=xlimits)
#centers = csampling(ncent)

# Error
xtest = None 
ftest = None
testdata = None
xtest = sampling(Nerr)
ftest = trueFunc(xtest)
testdata = [xtest, ftest]

# centers
nc = np.linspace(nc0, ncr, nruns, dtype=int)
xcentK = []
for j in range(nruns):
    xcentK.append(csampling(nc[j]))

# training data
xtrainK = sampling(ntrain)
ftrainK = trueFunc(xtrainK)
gtrainK = np.zeros([ntrain,dim])
for i in range(dim):
    gtrainK[:,i:i+1] = trueFunc(xtrainK,i)

# LSRBF
modellrb = LSRBF()
modellrb.options.update({"t0":t0})
modellrb.options.update({"corr":corr})
# modellrb.options.update({"basis_centers":centers})
modellrb.options.update({"compute_theta":tcomp})
modellrb.options.update({"use_derivatives":False})
modellrb.options.update({"print_prediction":False})
# modellrb.set_training_values(xtrainK[0], ftrainK[0])
# for i in range(dim):
#     modellrb.set_training_derivatives(xtrainK[0], gtrainK[0][:,i:i+1], i)
# modellrb.train()

# LSRBF G
modellgb = LSRBF()
modellgb.options.update({"t0":t0})
modellgb.options.update({"corr":corr})
# modellgb.options.update({"basis_centers":centers})
modellgb.options.update({"compute_theta":tcomp})
modellgb.options.update({"use_derivatives":True})
modellgb.options.update({"print_prediction":False})
# modellgb.set_training_values(xtrainK[0], ftrainK[0])
# for i in range(dim):
#     modellgb.set_training_derivatives(xtrainK[0], gtrainK[0][:,i:i+1], i)
# modellgb.train()

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


modelkrg = KRG()
#modelkrg.options.update({"hyper_opt":"TNC"})
#modelkrg.options.update({"theta0":[t0]})
modelkrg.options.update({"theta_bounds":tb})
modelkrg.options.update({"corr":corr})
modelkrg.options.update({"poly":poly})
modelkrg.options.update({"n_start":5})
modelkrg.options.update({"print_prediction":False})

errlrb = []
errlgb = []
errgek = []
errkrg = []
errgrb = []
mlrb = []
mlgb = []
mgek = []
mkrg = []
mdge = []

# run other models for comparison
if(dim > 1):
    modelgek.set_training_values(xtrainK, ftrainK)
    for i in range(dim):
        modelgek.set_training_derivatives(xtrainK, gtrainK[:,i:i+1], i)
    modelgek.train()
else:
    nex = xtrainK.shape[0]
    xaug = np.zeros([nex, 1])
    faug = np.zeros([nex, 1])
    for k in range(nex):
        xaug[k] = xtrainK[k] + dx
        faug[k] = ftrainK[k] + dx*gtrainK[k]
    xtot = np.append(xtrainK, xaug, axis=0)
    ftot = np.append(ftrainK, faug, axis=0)
    modelgek.set_training_values(xtot, ftot)
    modelgek.train()
print("GEK Done")

modelkrg.set_training_values(xtrainK, ftrainK)
modelkrg.train()
errgek.append(rmse(modelgek, trueFunc, N=Nerr, xdata=xtest, fdata=ftest))
errkrg.append(rmse(modelkrg, trueFunc, N=Nerr, xdata=xtest, fdata=ftest))
print("KRG Done")



for j in range(nruns):


    #LSRBF
    modellrb.options.update({"basis_centers":xcentK[j]})
    modellrb.set_training_values(xtrainK, ftrainK)
    modellrb.train()
    mlrb.append(copy.deepcopy(modellrb))
    errlrb.append(rmse(modellrb, trueFunc, N=Nerr, xdata=xtest, fdata=ftest))
    print("LSRBF Done")
    
    ncent = int(xtrainK.shape[0]/2) + int(dim*xtrainK.shape[0]/2)
    centers = csampling(ncent)
    modellgb.options.update({"basis_centers":xcentK[j]})
    modellgb.set_training_values(xtrainK, ftrainK)
    for i in range(dim):
        modellgb.set_training_derivatives(xtrainK, gtrainK[:,i:i+1], i)
    modellgb.train()
    mlgb.append(copy.deepcopy(modellgb))
    errlgb.append(rmse(modellgb, trueFunc, N=Nerr, xdata=xtest, fdata=ftest))
    print("LSRBF G Done")

    


print("GEK")
print(errgek)
print(modelgek.optimal_theta)
print("KRG")
print(errkrg)
print(modelkrg.optimal_theta)
print("LSRBF")
print(errlrb)
print(modellrb.theta)
print("LSRBF G")
print(errlgb)
print(modellgb.theta)

#import pdb; pdb.set_trace()
# print(errdge)
# print(modeldge.optimal_theta)

plt.loglog(nc, errgek*nruns, "k-", label=f'GEK')
plt.loglog(nc, errkrg*nruns, 'r-', label='KRG')
plt.loglog(nc, errlrb, 'm-', label='LSRBF (no g)')
plt.loglog(nc, errlgb, 'c-', label='LSRBF (g)')
plt.xlabel("Number of centers")
plt.ylabel("NRMSE")
plt.grid()
plt.legend(loc=1)
plt.savefig(f"./lsrbf_cent_test_err.png", bbox_inches="tight")
plt.clf()


# # Plot points
if(dim == 1):  

    ndir = 75
    xlimits = trueFunc.xlimits
    x = np.linspace(xlimits[0][0], xlimits[0][1], ndir)

    Zgek = np.zeros([ndir, ndir])
    Zkrg = np.zeros([ndir, ndir])
    Zlrb = np.zeros([ndir, ndir])
    Zlgb = np.zeros([ndir, ndir])
    Fgek = np.zeros([ndir, ndir])
    Fkrg = np.zeros([ndir, ndir])
    Flrb = np.zeros([ndir, ndir])
    Flgb = np.zeros([ndir, ndir])
    TF = np.zeros([ndir, ndir])

    for i in range(ndir):
        xi = np.zeros([1,1])
        xi[0] = x[i]
        TF[i] = trueFunc(xi)
        Fgek[i] = modelgek.predict_values(xi)
        Fkrg[i] = modelkrg.predict_values(xi)
        Flrb[i] = mlrb[plot].predict_values(xi)
        Flgb[i] = mlgb[plot].predict_values(xi)
        Zgek[i] = abs(Fgek[i] - TF[i])
        Zkrg[i] = abs(Fkrg[i] - TF[i])
        Zlrb[i] = abs(Flrb[i] - TF[i])
        Zlgb[i] = abs(Flgb[i] - TF[i])

    # Plot Non-Adaptive Error
    plt.plot(x, TF, "-g")
    plt.plot(x, Fgek, "-k", label=f'GEK')
    plt.plot(x, Fkrg, "-r", label=f'KRG')
    plt.plot(x, Flrb, "-m", label=f'LSRBF (no g)')
    plt.plot(x, Flgb, "-c", label=f'LSRBF (g)')
    plt.xlabel(r"$x$")
    plt.ylabel(r"$f$")
    #plt.legend(loc=1)
    plt.plot(xtrainK, ftrainK, "o", fillstyle='full', markerfacecolor='b', markeredgecolor='b', label='Initial Samples')

    plt.savefig(f"./lsrbf_cent_1dplot.png", bbox_inches="tight")
    plt.clf()

    # Plot Non-Adaptive Error
    plt.plot(x, Zgek, "-k", label=f'GEK')
    plt.plot(x, Zkrg, "-r", label=f'KRG')
    plt.plot(x, Zlrb, "-m", label=f'LSRBF (no g)')
    plt.plot(x, Zlgb, "-c", label=f'LSRBF (g)')

    plt.xlabel(r"$x$")
    plt.ylabel(r"$err$")
    #plt.legend(loc=1)
    plt.plot(xtrainK, np.zeros(xtrainK), "o", fillstyle='full', markerfacecolor='b', markeredgecolor='b', label='Initial Samples')

    plt.savefig(f"./lsrbf_cent_1derr.png", bbox_inches="tight")

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
    Zlrb = np.zeros([ndir, ndir])
    Zlgb = np.zeros([ndir, ndir])
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
            Zgek[j,i] = abs(mgek[plot].predict_values(xi) - TF[j,i])
            Zkpl[j,i] = abs(mkpl[plot].predict_values(xi) - TF[j,i])
            Zkrg[j,i] = abs(mkrg[plot].predict_values(xi) - TF[j,i])
            Zlrb[j,i] = abs(mlrb[plot].predict_values(xi) - TF[j,i])
            Zlgb[j,i] = abs(mlgb[plot].predict_values(xi) - TF[j,i])

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
    plt.colorbar(cs)
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    #plt.legend(loc=1)
    plt.plot(xtrainK[-1][:,0], xtrainK[-1][:,1], "o", fillstyle='full', markerfacecolor='b', markeredgecolor='b', label='Initial Samples')
    plt.savefig(f"./gek_errcona.png", bbox_inches="tight")

    plt.clf()

    plt.contour(X, Y, Zlrb, levels = cs.levels)
    plt.colorbar(cs)
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    #plt.legend(loc=1)
    plt.plot(xtrainK[-1][:,0], xtrainK[-1][:,1], "o", fillstyle='full', markerfacecolor='b', markeredgecolor='b', label='Initial Samples')
    plt.savefig(f"./lsrbf_errcona.png", bbox_inches="tight")

    plt.clf()  

    plt.contour(X, Y, Zlgb, levels = cs.levels)
    plt.colorbar(cs)
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    #plt.legend(loc=1)
    plt.plot(xtrainK[-1][:,0], xtrainK[-1][:,1], "o", fillstyle='full', markerfacecolor='b', markeredgecolor='b', label='Initial Samples')
    plt.savefig(f"./lsrbfg_errcona.png", bbox_inches="tight")

    plt.clf()  


    plt.contour(X, Y, Zkrg, levels = cs.levels)
    plt.colorbar(cs)
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    #plt.legend(loc=1)
    plt.plot(xtrainK[-1][:,0], xtrainK[-1][:,1], "o", fillstyle='full', markerfacecolor='b', markeredgecolor='b', label='Initial Samples')
    plt.savefig(f"./krg_errcona.png", bbox_inches="tight")

    plt.clf()

