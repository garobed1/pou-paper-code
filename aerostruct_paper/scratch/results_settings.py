
header = "pou_hess_test"
skip_LHS = False
LHS_batch = 4
runs_per_proc = 1

# Problem Conditions
prob  = "arctan"    #problem
dim = 3     #problem dimension


# Surrogate Settings
stype = "pouhess"    #surrogate type
rtype = "hess"     #criteria type
corr  = "matern32"  #kriging correlation
poly  = "linear"    #kriging regression 
extra = dim           #gek extra points
t0 = [1e-0]
tb = [1e-2, 2e+1]
rho = 120            #POU parameter

# Adaptive Sampling Settings
nt0  = dim*5       #initial design size
ntr = dim*30       #number of points to add
ntot = nt0 + ntr    #total number of points
batch = 1#dim*2        #batch size for refinement, as a percentage of ntr
Nerr = 5000       #number of test points to evaluate the error
pperb = batch
pperbk = int(ntr/LHS_batch)
multistart = 25*dim     #aniso opt multistart
if(pperb == 0):
    pperb = 1

# Refinement Settings
neval = 1+2*dim
hess  = "neighborhood"
interp = "honly"
criteria = "distance"
perturb = True
bpen = False
obj = "inv"
rscale = 2.5 #0.5 for 2D
nscale = 10.0 #1.0 for 2D
nmatch = dim
opt = 'L-BFGS-B' #'SLSQP'#