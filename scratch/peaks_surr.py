import sys
import numpy as np
sys.path.insert(1,"../surrogate")
from problem_picker import GetProblem
from defaults import DefaultOptOptions
from optimizers import optimize
from matplotlib import cm
import matplotlib.pyplot as plt
from smt.surrogate_models import KPLS, GEKPLS, KRG
from smt.sampling_methods import LHS
from error import rmse

prob = "peaks"
dim = 2

func = GetProblem(prob, dim)

npts = 30

plt.rcParams['font.size'] = '14'
plt.rc('legend',fontsize=14)

modelkrg = KRG()
#modelkrg.options.update({"hyper_opt":"TNC"})
modelkrg.options.update({"print_prediction":False})

modelgek = GEKPLS(xlimits = func.xlimits)
#modelkrg.options.update({"hyper_opt":"TNC"})
modelgek.options.update({"n_comp":dim})
modelgek.options.update({"extra_points":dim})
modelgek.options.update({"print_prediction":False})

sampling = LHS(xlimits=func.xlimits, criterion='m')
trx = sampling(npts)
trf = func(trx)
trg = np.zeros_like(trx)
for i in range(dim):
    trg[:,i:i+1] = func(trx,i)

modelkrg.set_training_values(trx, trf)
modelkrg.train()

modelgek.set_training_values(trx, trf)
for i in range(dim):
    modelgek.set_training_derivatives(trx, trg[:,i:i+1], i)
modelgek.train()

# Plot contours
#contour
ndir = 150
xlimits = func.xlimits

x = np.linspace(xlimits[0][0], xlimits[0][1], ndir)
y = np.linspace(xlimits[1][0], xlimits[1][1], ndir)
X, Y = np.meshgrid(x, y)
Za = np.zeros([ndir, ndir])
Zk = np.zeros([ndir, ndir])
F  = np.zeros([ndir, ndir])
FG  = np.zeros([ndir, ndir])
TF = np.zeros([ndir, ndir])
for i in range(ndir):
    for j in range(ndir):
        xi = np.zeros([1,2])
        xi[0,0] = x[i]
        xi[0,1] = y[j]
        F[j,i]  = modelkrg.predict_values(xi)
        FG[j,i]  = modelgek.predict_values(xi)
        TF[j,i] = func(xi)


# Plot original function
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
surf = ax.plot_surface(X, Y, TF, cmap=cm.viridis)
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
ax.set_zlabel(r"$f(x)$")
ax.set_zlim(-6, 6)

#plt.legend(loc=1)
plt.savefig(f"peaks_true.pdf", bbox_inches="tight")
plt.clf()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d', computed_zorder=False)
surf = ax.plot_surface(X, Y, F, cmap=cm.viridis)
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
#plt.legend(loc=1)
ax.scatter(trx[:,0], trx[:,1], trf[:], c='r', alpha=1., label='Surrogate Samples')
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
ax.set_zlabel(r"$\hat{f}(x)$")
ax.set_zlim(-6, 6)
plt.legend()
plt.savefig(f"peaks_surr.pdf", bbox_inches="tight")
plt.clf()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d', computed_zorder=False)
surf = ax.plot_surface(X, Y, FG, cmap=cm.viridis)
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
#plt.legend(loc=1)
ax.scatter(trx[:,0], trx[:,1], trf[:], c='m', alpha=1., label='Surrogate Samples with Gradients')
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
ax.set_zlabel(r"$\hat{f}(x)$")
ax.set_zlim(-6, 6)
plt.legend()
plt.savefig(f"peaks_surr_g.pdf", bbox_inches="tight")
plt.clf()



print(rmse(modelkrg, func, N=10000))
print(rmse(modelgek, func, N=10000))