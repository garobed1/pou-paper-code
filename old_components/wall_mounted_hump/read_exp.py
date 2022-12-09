
# function for reading experimental data from exp_dat.txt for the wall-mounted hump
# returns lists of centerspan x/c, Cp
def read_wmh_data():
    c = -1
    x = []
    cp = []
    with open('exp_dat.txt') as f:
        for line in f:
            c += 1
            if c < 2:
                continue
            xl, cpl = line.split()
            x.append(float(xl))
            cp.append(float(cpl))

    return x, cp

            

#read_wmh_data()