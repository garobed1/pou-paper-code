
header = "pou_precci_test"
skip_LHS = False
LHS_batch = 6
runs_per_proc = 1

# Problem Conditions
prob  = "arctan"    #problem
dim = 1     #problem dimension


# Surrogate Settings
stype = "pouhess"    #surrogate type
# rtype = "poussa"#     #criteria type #poussa
rtype =  "hess"
corr  = "matern32"  #kriging correlation
poly  = "linear"    #kriging regression 
extra = dim           #gek extra points
t0 = [1e-0]
tb = [1e-5, 2e+1]
C = 5.5
rscale = 5.5 #0.5 for 2D
rho = 10          #POU parameter

# Adaptive Sampling Settings
nt0  = dim*10       #initial design size
ntr = dim*30       #number of points to add
ntot = nt0 + ntr    #total number of points
batch = 1#dim*2        #batch size for refinement, as a percentage of ntr
Nerr = 5000       #number of test points to evaluate the error
pperb = batch
pperbk = int(ntr/LHS_batch)
mstarttype = 2            # 0: No multistart
                          # 1: Start at the best out of a number of samples
                          # 2: Perform multiple optimizations
if(mstarttype == 1):   
    multistart = 50*dim
if(mstarttype == 2):
    multistart = 5*dim

if(pperb == 0):
    pperb = 1

# Refinement Settings
neval = 1+(dim+2)
hess  = "neighborhood"
interp = "honly"
criteria = "distance"
perturb = True
bpen = False
obj = "inv"
nscale = 10.0 #1.0 for 2D
nmatch = dim
opt = 'L-BFGS-B' #'SLSQP'#