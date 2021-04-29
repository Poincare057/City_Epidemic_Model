import numpy as np
import matplotlib.pyplot as plt

#City Size (gridx and gridy initialized here are +2 more than the actual city size)
gridx = 28
gridy = 28

#Total time of simulation
T = 240

# degrees of influence
ind = 8

#Initializing relevant matrices
cases = np.zeros((gridx, gridy, T))
tot_cases  = np.zeros((gridx, gridy, T))

#tot_pop for convenience is generate using the same diffusion; this has no real meaning though
tot_pop = np.zeros((gridx, gridy, T))

p = np.zeros((ind, gridx, gridy, T))      
diff = np.zeros((gridx, gridy, T))

def growth_model(trans_rate, case_diff_c, fall):
    ''' Models the spread of epidemic with transmission rate 'trans_rate', case diffusion coefficient 'case_diff_c', exponential damping parameter 'fall' '''
    #Set relevant arrays to zeros
    global cases, tot_cases, tot_pop, p, diff
    cases = np.zeros((gridx, gridy, T))
    tot_cases  = np.zeros((gridx, gridy, T))
    tot_pop = np.zeros((gridx, gridy, T))
    p = np.zeros((ind, gridx, gridy, T))      
    diff = np.zeros((gridx, gridy, T))

    #population diffusion coefficient
    diff_c = 0.9

    #Points of index cases and population diffusion start
    no_src = 40
    sources = np.zeros((2, no_src))
    for i in range(no_src):
        sources[0][i] = np.random.randint(1, gridx-1)
        sources[1][i] = np.random.randint(1, gridy-1)
    common_start = 1746

    #IT IS ESSENTIAL FOR POPULATION AND CASES TO HAVE THE SAME INITIAL CONDITIONS!!!
    #Index case(s)
    for i in range(no_src):
        cases[int(sources[0][i])][int(sources[1][i])][0] = common_start
        tot_cases[int(sources[0][i])][int(sources[1][i])][0] = cases[int(sources[0][i])][int(sources[1][i])][0]

    #Initial condition(s) for tot_pop diffusion
    pop_fall = 0.13
    for i in range(no_src):
        tot_pop[int(sources[0][i])][int(sources[1][i])][0] = common_start

    #Initializing amount of diffusion for cases (Anisotropic)
    for i in range(gridx):
        for j in range(gridy):
            diff[i][j][0] = 0.001*np.random.randint(400, 1000)*case_diff_c
            s = diff[i][j][0]
            for k in range(ind - 1):
                #Highly Anistropic Diffusion
                p[k][i][j][0] = np.random.random()*s
                s -= p[k][i][j][0]
                
                #For isotropy
                #p[k][i][j][0] = diff[i][j][0]/8
            #p[ind-1] = diff[i][j][0]/8
                
            p[ind-1] = s
            
    #Exponential Decrease in amount of diffusion with time
    for i in range(gridx):
        for j in range(gridy):
            for t in range(1, T):
                diff[i][j][t] = diff[i][j][t-1]*np.exp(-fall)
                for k in range(ind):
                    p[k][i][j][t] = p[k][i][j][t-1]*np.exp(-fall)
                    
    #Assigns population on grid using isotropic diffusion; as mentioned before, the diffusion is a matter of convenience, not of any physical meaning
    for t in range(1, T):
        for i in range(1, gridx-1):
            for j in range(1, gridy-1):
                tot_pop[i][j][t] = tot_pop[i][j][t-1] + int(np.exp(-pop_fall*t)*0.125*(tot_pop[i+1][j+1][t-1]*diff_c + tot_pop[i][j+1][t-1]*diff_c +
                                             tot_pop[i-1][j+1][t-1]*diff_c + tot_pop[i-1][j][t-1]*diff_c +
                                             tot_pop[i-1][j-1][t-1]*diff_c + tot_pop[i][j-1][t-1]*diff_c +
                                             tot_pop[i+1][j-1][t-1]*diff_c + tot_pop[i+1][j][t-1]*diff_c))

    #Cases diffusion; highly anistropic and controlled by population
    for t in range(1, T):
        for i in range(1, gridx-1):
            for j in range(1, gridy-1):
                cases[i][j][t] = trans_rate*(tot_cases[i+1][j+1][t-1]*p[4][i+1][j+1][t-1] + tot_cases[i][j+1][t-1]*p[5][i][j+1][t-1] +
                                             tot_cases[i-1][j+1][t-1]*p[6][i-1][j+1][t-1] + tot_cases[i-1][j][t-1]*p[7][i-1][j][t-1] +
                                             tot_cases[i-1][j-1][t-1]*p[0][i-1][j-1][t-1] + tot_cases[i][j-1][t-1]*p[1][i][j-1][t-1] +
                                             tot_cases[i+1][j-1][t-1]*p[2][i+1][j-1][t-1] + tot_cases[i+1][j][t-1]*p[3][i+1][j][t-1])
                
                #Prevents no of cases from exceeding population in area
                if tot_cases[i][j][t-1] + int(cases[i][j][t]) < tot_pop[i][j][T-1]:
                    tot_cases[i][j][t] = tot_cases[i][j][t-1] + int(cases[i][j][t])
                else:
                    tot_cases[i][j][t] = tot_cases[i][j][t-1]


#Functions to plot and view the data
def plot_tot_cases(t):
    ar = np.zeros((gridx,gridy))
    for i in range(gridx):
        for j in range(gridy):
            ar[i][j] = tot_cases[i][j][t]
    plt.contourf(np.transpose(ar))
    plt.show()

def plot_tot_pop(t):
    ar = np.zeros((gridx,gridy))
    for i in range(gridx):
        for j in range(gridy):
            ar[i][j] = tot_pop[i][j][t]
    plt.contourf(np.transpose(ar))
    plt.show()

def net_cases(t):
    ''' total number of cases after t days '''
    s = 0
    for i in range(1, gridx):
        for j in range(1, gridy):
            s += tot_cases[i][j][t]
    return s

def tot_cases_view(t):
    ''' prints out the total number of cases recorded at every grid point up to time t '''
    for i in range(1, gridx):
        for j in range(1, gridy):
            print(tot_cases[i][j][t], end = " ")
        print()

def tot_pop_view(t):
    ''' prints out population at every grid point at time t '''
    for i in range(1, gridx):
        for j in range(1, gridy):
            print(tot_pop[i][j][t], end = " ")
        print()
        
def net_pop(t):
    ''' total population of city at time t '''
    s = 0
    for i in range(1, gridx):
        for j in range(1, gridy):
            s += tot_pop[i][j][t]
    return s

def daily_cases():
    ''' returns list where each element is the number of new cases on a day; actual day corresponds to list index + 1 '''
    d_cases = []
    for t in range(1, T):
        d_cases += [net_cases(t)-net_cases(t-1)]
    return d_cases

def net_cases_list():
    ''' returns list where each element is the total number of cases recorded up till time t; day corresponds to list index '''
    n_cases = []
    for t in range(T):
        n_cases += [net_cases(t)]
    return n_cases

def print_daily_cases():
    for t in range(1, T):
            if t%10 == 0:
                    print()
            print(daily_cases()[t-1], end = " ")


## The following estimate the constant for the exponential growth in the number of daily cases. As of 20th April it is 0.07101 in India. 
def expinf(lamb, d_cases):
	d_cases_exp = [d_cases[0]]                 #Note that d_cases[0] is actually the new cases recorded on day 1 (In this case 11 April 2021)
	for i in range(1, len(d_cases)):
		d_cases_exp += [d_cases[0]*2.718282**(lamb*i)]
	return d_cases_exp

def mse(data, data1):
    ''' returns mean squared error of entries in data and data1 '''
    err = 0
    for i in range(len(data)):
        err += (data[i]-data1[i])**2
    return err/len(data)
    
def findexp(d_cases):
    ''' exponential regression of d_cases'''
    minl = 1000000
    miner = 100000000000000000000000
    for i in range(1000, 9000):
	    meansq = mse(d_cases, expinf(i/100000, d_cases))
	    if meansq < miner:
		    minl = i
		    miner = meansq
    return minl/100000

def plot_daily_cases(peak):
    plt.plot(daily_cases())
    plt.plot(expinf(findexp(daily_cases()[0:peak]), daily_cases()[0:peak]))
    plt.show()

def plot_net_cases():
    plt.plot(net_cases_list())
    plt.show()

##Actually finding the 'best' model; fitting it to the first N days of exponential.

#To record the predictions of the 'best' model
b_cases = np.zeros((gridx, gridy, T))
b_tot_cases  = np.zeros((gridx, gridy, T))
b_tot_pop = np.zeros((gridx, gridy, T))
b_p = np.zeros((ind, gridx, gridy, T))      
b_diff = np.zeros((gridx, gridy, T))
b_daily_cases = np.zeros(T-1)
b_net_cases = np.zeros(T)
    
def infer_parameters():
    global b_cases, b_tot_cases, b_tot_pop, b_p, b_diff, b_daily_cases, b_net_cases
    minerr = 10000000000000000000000000000
    best_case_diff = 0
    best_fall = 0 
    for i in range(5, 15):
        for j in range(10, 40):
            print(i, j)

            #executing the model for the given parameters
            growth_model(1.65, i/100, j/1000)
            
            print(net_pop(239))         #debug
            
            err = mse(expinf(0.07101, daily_cases()[0:10]), daily_cases()[0:10])
            
            if err < minerr:
                minerr = err
                best_case_diff = i/100
                best_fall = j/1000
                b_cases = cases
                b_tot_cases = tot_cases
                b_p = p
                b_diff = diff
                
    print(best_case_diff, best_fall)

    #Setting to zeros
    b_daily_cases = np.zeros(T-1)
    b_net_cases = np.zeros(T)
    
    for t in range(T):
        for i in range(gridx):
            for j in range(gridy):
                b_net_cases[t] += tot_cases[i][j][t]
    
    for t in range(1, T):
        b_daily_cases[t-1] = b_net_cases[t] - b_net_cases[t-1]

    plt.plot(b_daily_cases)
    plt.show()
                

#These are just debug functions to check that cases <= population everywhere    
def compat(t):
    for i in range(1, gridx):
        for j in range(1, gridy):
            if tot_pop[i][j][T-1] < tot_cases[i][j][t]:
                print(i, j, tot_pop[i][j][T-1], tot_cases[i][j][t])
                return False
    return True
def fullcompat():
    for t in range(1, T):
        if compat(t) == False:
            return compat(t)
    return True
