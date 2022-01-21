import numpy as np
import matplotlib.pyplot as plt
import pougrad

from smt.problems import Sphere
from smt.surrogate_models import gekpls

dim = 100

trueFunc = Sphere(ndim=dim)

ntr = 100
nte = 10

tr = np.random.rand(ntr,dim)*20 - 10
# tr = np.zeros([ntr,dim])
# tr[:,0] = np.linspace(-10,10,ntr)
gr = np.zeros([ntr,dim])

te = np.random.rand(nte, dim)*20 - 10
#te = np.sort(te,0)
fe = np.zeros(nte)

fr = trueFunc(tr)

for i in range(dim):
    gr[:,i:i+1] = trueFunc(tr,i)

fe = trueFunc(te)
pou = pougrad.POUSurrogate(tr,fr,gr,10)
gek = gekpls.GEKPLS(theta0=[1e-2], xlimits=trueFunc.xlimits)
gek.set_training_values(tr, fr)
for i in range(dim):
    gek.set_training_derivatives(tr, gr[:,i:i+1], i)
gek.train()

# test the surrogate
fesp = np.zeros(nte)
fesg = np.zeros(nte)

for i in range(nte):
    fesp[i] = pou.eval(te[i])
    fesg[i] = gek.predict_values(te[i:i+1,:])

errp = fesp - fe[:,0]
errg = fesg - fe[:,0]

print(errp)
print(errg)

#import pdb; pdb.set_trace()

# plt.plot(tr[:,0],fr[:,0])
# plt.plot(te[:,0],fes,"o")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.savefig("sphereplot.png")
