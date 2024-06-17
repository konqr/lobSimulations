#%%
import pickle
import numpy as np
import pandas as pd
import time
import numpy as np
import os 

#perf_counter() measures the real amount of time for a process to take, as if you used a stop watch. Includes I/O and sleeping

# start = time.perf_counter_ns()
# time.sleep(1)
# end = time.perf_counter_ns()
# print("it took" + str((end-start)/10**9)+ " secs.")


def powerLawKernel(x, alpha = 1., t0 = 1., beta = -2.):
    if x < t0: return 0
    return alpha*(x**beta)

def powerLawCutoff(time, alpha, beta, gamma):
    # alpha = a*beta*(gamma - 1)
    funcEval = alpha/((1 + gamma*time)**beta)
    # funcEval[time < t0] = 0
    return funcEval

def powerLawKernelIntegral(x1, x2, alpha = 1., t0 = 1., beta = -2.):
    return (x2/(1+beta))*powerLawKernel(x2, alpha = alpha, t0=t0, beta=beta) - (x1/(1+beta))*powerLawKernel(x1, alpha = alpha, t0=t0, beta=beta)

       
def pairtoindex(i: int, j: int) -> int:        #maps (i,j) to a unique index in [0 .. 143]
    if 0 <= i < 12 and 0 <= j < 12:
        return i * 12 + j 
    else:
        raise ValueError("Indices must be in the range [0, 11]") #maps unique 0<= j <=143 to corresponding index (i, j)

def indextopair(index: int) -> tuple[int, int] :
    if 0<=index<=143:
        return (index//12, index%12)
    else: 
        raise ValueError("Index must be in the range [0, 143]")
num_nodes=12
def preprocessdata(paramsPath: str, todPath: str):
    """Takes in params and todpath and spits out corresponding vectorised numpy arrays
    
    Returns:
    tod: a [12, 13] matrix containing values of f(Q_t), the time multiplier for the 13 different 30 min bins of the trading day.
    params=[kernelparams, baselines]
        kernelparams: an array of [12, 12] matrices consisting of mask, alpha, beta, gamma. the item at arr[i][j] corresponds to the corresponding value from params[cols[i] + "->" + cols[j]]
        baselines: a vector of dim=(num_nodes, 1) consisting of baseline intensities
    """
    
    cols = ["lo_deep_Ask", "co_deep_Ask", "lo_top_Ask","co_top_Ask", "mo_Ask", "lo_inspread_Ask" ,
            "lo_inspread_Bid" , "mo_Bid", "co_top_Bid", "lo_top_Bid", "co_deep_Bid","lo_deep_Bid" ]
    with open(todPath, "rb") as f:
        data = pickle.load(f)
    tod=np.zeros(shape=(num_nodes, 13))
    for i in range(num_nodes):
        tod[i]=[data[cols[i]][k] for k in range(13)]
        
    with open(paramsPath, "rb") as f:
        data=pickle.load(f)
    baselines=np.zeros(shape=(num_nodes, 1)) #vectorising baselines
    for i in range(num_nodes):
        baselines[i]=data.pop(cols[i], None)
    
    
    #params=[mask, alpha, beta, gamma] where each is a 12x12 matrix
    mask, alpha, beta, gamma=[np.zeros(shape=(12, 12)) for _ in range(4)]
    for i in range(num_nodes):
        for j in range(num_nodes):
            kernelParams = data.get(cols[i] + "->" + cols[j], None)
            mask[i][j]=kernelParams[0]
            alpha[i][j]=kernelParams[1][0]
            beta[i][j]=kernelParams[1][1]
            gamma[i][j]=kernelParams[1][2]
    kernelparams=[mask, alpha, beta, gamma] 
    params=[kernelparams, baselines] 
    return tod, params
tod, params=preprocessdata(paramsPath='fake_ParamsInferredWCutoff_sod_eod_true', todPath='fakeData_Params_sod_eod_dictTOD_constt')
 

#%%
def thinningOgataIS2(T, params, tod, num_nodes=12, maxJumps = None, s = None, n = None, Ts = None, spread=None, beta = 0.7479, avgSpread = 0.0169,lamb= None):
    """ 
    Arguments:
    T: timelimit of simulation process
    params=[kernelparams, baselines]
        kernelparams: an array of [12, 12] matrices consisting of mask, alpha, beta, gamma. the item at arr[i][j] corresponds to the corresponding value from params[cols[i] + "->" + cols[j]]
        baselines: a vector of dim=(num_nodes, 1) consisting of baseline intensities
    tod: a [12, 13] matrix containing values of f(Q_t), the time multiplier for the 13 different 30 min bins of the trading day.
    num_nodes: #of different processes
    """
    
    """Setting up thinningOgata params"""
    cols = ["lo_deep_Ask", "co_deep_Ask", "lo_top_Ask","co_top_Ask", "mo_Ask", "lo_inspread_Ask" ,
            "lo_inspread_Bid" , "mo_Bid", "co_top_Bid", "lo_top_Bid", "co_deep_Bid","lo_deep_Bid" ]
    baselines =params[1].copy()
    mat = np.zeros((num_nodes, num_nodes))
    if s is None: 
        s = 0
    hourIndex = min(12,int(s//1800)) #1800 seconds in 30 minutes
    todmult=tod[:, hourIndex].reshape((12, 1))
    #todMult*kernelParams[0]*kernelParams[1][0]/((-1 + kernelParams[1][1])*kernelParams[1][2]) 
    mat=todmult*params[0][0]*params[0][1]/((params[0][2]-1) *params[0][3])
    baselines[5] = ((spread/avgSpread)**beta)*baselines[5]
    baselines[6] = ((spread/avgSpread)**beta)*baselines[6]
    specRad = np.max(np.linalg.eig(mat)[0])
    print("spectral radius = ", specRad)
    specRad = np.max(np.linalg.eig(mat)[0]).real
    if specRad < 1 : specRad = 0.99 #  # dont change actual specRad if already good
    """Initializing variable values"""
    numJumps = 0
    if n is None: n = num_nodes*[0]
    if Ts is None: Ts = num_nodes*[()]
    Ts_new = num_nodes*[()]
    if spread is None: spread = 1
    
    """calculating initial values of lamb_bar"""
    decays=(0.99/specRad) * todmult * baselines
    """array([[0.06079649],
       [0.06079649],
       [0.03039825],
       [0.04053099],
       [0.05066374],
       [1.28128916],
       [1.28128916],
       [0.04053099],
       [0.05066374],
       [0.04053099],
       [0.04053099],
       [0.07092924]])"""
    lamb_bar=np.sum(decays) #3.04895025
    
    while s<=T:
        u=np.random.uniform(0, 1)
        if lamb_bar==0:
            s+=0.1  # wait for some time
        else:
            w=max(1e-7, -1 * np.log(u)/lamb_bar) # floor at 0.1 microsec
            s+=w
        """Recalculating lambdas with new candidate"""
        hourIndex = min(12,int(s//1800))
        todmult=tod[:, hourIndex].reshape((12, 1))
        decays=(0.99/specRad) * todmult * baselines
        
        for i in range(num_nodes):
            points=Ts[i]#all the old points in a specific dimension
            for point in points:
                if s - point >= 500: continue
                if s - point < 1e-4: continue
                for j in range(len(Ts)):
                    
                
    return None
        
#%%
def thinningOgataIS(T, paramsPath, todPath, num_nodes = 12, maxJumps = None, s = None, n = None, Ts = None, spread=None, beta = 0.7479, avgSpread = 0.0169,lamb= None):
    """
    Parameters:
    T: 
    paramsPath: path to params: which is a dict containing (alpha, beta, gamma)
    todPath: path to tod: which is a [12, 13] matrix containing values of f(Q_t), the time multiplier for the 13 different 30 min bins of the trading day.
    num_nodes: number of dimensions/processes in the hawkes simulator
    s: current time candidate
    n: vector count of number of points in each dimension
    Ts: set of points for each dimension/process
    spread
    beta
    avgspread
    lamb
    """
    if maxJumps is None: maxJumps = np.inf
    tryer = 0
    while tryer < 5: # retry on pickle clashes
        try:
            with open(paramsPath, "rb") as f:
                params = pickle.load(f)
            with open(todPath, "rb") as f:
                tod = pickle.load(f)
            tryer =  6
        except:
            time.sleep(1)
            tryer +=1
            continue
    """Set up Variables for ThinningOgata"""
    cols = ["lo_deep_Ask", "co_deep_Ask", "lo_top_Ask","co_top_Ask", "mo_Ask", "lo_inspread_Ask" ,
            "lo_inspread_Bid" , "mo_Bid", "co_top_Bid", "lo_top_Bid", "co_deep_Bid","lo_deep_Bid" ]
    baselines = num_nodes*[0]
    mat = np.zeros((num_nodes, num_nodes))
    if s is None: 
        s = 0
    hourIndex = min(12,int(s//1800)) #1800 seconds in 30 minutes
    for i in range(num_nodes):
        for j in range(num_nodes):
            kernelParams = data.get(cols[i] + "->" + cols[j], None)
            if kernelParams is None: continue
            if np.isnan(kernelParams[1][2]): continue
            # print(cols[i] + "->" + cols[j])
            # print((kernelParams[0]*np.exp(kernelParams[1][0]) , kernelParams[1][1] , kernelParams[1][2]))
            todMult = tod[cols[j]][hourIndex]
            #implement vectorisation and broadcasting for this
            mat[i][j]  = todMult*kernelParams[0]*kernelParams[1][0]/((-1 + kernelParams[1][1])*kernelParams[1][2]) # alpha/(beta -1)*gamma   
        #baselines[i] = data[cols[i]]
    baselines[5] = ((spread/avgSpread)**beta)*baselines[5]
    baselines[6] = ((spread/avgSpread)**beta)*baselines[6]
    specRad = np.max(np.linalg.eig(mat)[0])
    print("spectral radius = ", specRad)
    specRad = np.max(np.linalg.eig(mat)[0]).real
    if specRad < 1 : specRad = 0.99 #  # dont change actual specRad if already good
    
    
    
    numJumps = 0
    if n is None: n = num_nodes*[0]
    if Ts is None: Ts = num_nodes*[()]
    Ts_new = num_nodes*[()]
    if spread is None: spread = 1
    
    """Compute initial value of lambbar"""
    if lamb is None:
        print(baselines)

        decays = baselines.copy()
        for i in range(len(Ts)):
            todMult = tod[cols[j]][hourIndex]*0.99/specRad
            decays[i] = todMult*decays[i]
        if (type(baselines[0]) == np.float64)or(type(baselines[0]) == float):
            lamb = sum(decays)
        else:
            lamb = sum(np.array(baselines)[:,0])
    
    """Begin thinningOgata simulation loop"""
    while s <= T:
        lambBar = lamb
        print(lambBar)
        u = np.random.uniform(0,1)
        if lambBar == 0:
            s += 0.1 # wait for some time
        else:
            w = np.max([1e-7,-1*np.log(u)/lambBar]) # floor at 0.1 microsec
            """Update candidate point"""
            s += w
        
        """calculating sum of baselines"""
        decays = baselines.copy()
        hourIndex = min(12,int(s//1800))
        for i in range(len(Ts)):
            todMult = tod[cols[i]][hourIndex]*0.99/specRad
            decays[i] = todMult*decays[i]
        
        """summation term"""
        for i in range(len(Ts)): #num_nodes
            taus = Ts[i]
            idx = np.searchsorted(taus, s - 10)
            for tau in taus[idx:]:
                if s - tau >= 10: continue
                #if s - tau < 1e-4: continue
                for j in range(len(Ts)):
                    kernelParams = params.get(cols[i] + "->" + cols[j], None)
                    todMult = tod[cols[j]][hourIndex]*0.99/specRad
                    if kernelParams is None: continue
                    if np.isnan(kernelParams[1][2]): continue
                    # decay = todMult*powerLawKernel(s - tau, alpha = kernelParams[0]*np.exp(kernelParams[1][0]), t0 = kernelParams[1][2], beta = kernelParams[1][1])
                    # print(decay)
                    decay = todMult*powerLawCutoff(s - tau, kernelParams[0]*kernelParams[1][0], kernelParams[1][1], kernelParams[1][2])
                    decays[j] += decay
        decays = [np.max([0, d]) for d in decays]
        decays[5] = ((spread/avgSpread)**beta)*decays[5]
        decays[6] = ((spread/avgSpread)**beta)*decays[6]
        if 100*np.round(spread, 2) < 2 : decays[5] = decays[6] = 0
        print(decays)
        lamb = sum(decays)
        print(lamb)
        
        
        
        """Testing candidate point"""
        D = np.random.uniform(0,1)
        if D*lambBar <= lamb: #accepted
            print(w)
            "Assign candidate point to a process by ratio of intensities"
            k = 0
            while D*lambBar >= sum(decays[:k+1]):
                k+=1
            # instantaneous lamb jumps
            if k in [5,6]:
                spread = spread - 0.01
                
            newdecays = len(cols)*[0]
            for i in range(len(Ts)):
                kernelParams = params.get(cols[k] + "->" + cols[i], None)
                todMult = tod[cols[i]][hourIndex]*0.99/specRad
                if kernelParams is None: continue
                if np.isnan(kernelParams[1][2]): continue
                # decay = todMult*powerLawKernel(0, alpha = kernelParams[0]*np.exp(kernelParams[1][0]), t0 = kernelParams[1][2], beta = kernelParams[1][1])
                decay = todMult*powerLawCutoff(0, kernelParams[0]*kernelParams[1][0], kernelParams[1][1], kernelParams[1][2])
                # print(decay)
                newdecays[i] += decay
            newdecays = [np.max([0, d]) for d in newdecays]
            newdecays[5] = ((spread/avgSpread)**beta)*newdecays[5]
            newdecays[6] = ((spread/avgSpread)**beta)*newdecays[6]
            if 100*np.round(spread, 2) < 2 : newdecays[5] = newdecays[6] = 0
            lamb += sum(newdecays)
            print(lamb)
            n[k] += 1
            if len(Ts[k]) > 0:
                T_Minus1 = Ts[k][-1]
            else:
                T_Minus1 = 0
            decays = np.array(baselines.copy())
            hourIndex = min(12,int(s//1800))
            decays[5] = ((spread/avgSpread)**beta)*decays[5]
            decays[6] = ((spread/avgSpread)**beta)*decays[6]
            decays = decays*(s-T_Minus1)
            
            """Updating history and returns"""
            tau = decays[k]
            Ts[k] += (s,)
            Ts_new[k] += (s,)
            numJumps += 1
            if numJumps >= maxJumps:
                return s,n,Ts, Ts_new, tau, lamb
    return s,n, Ts, Ts_new, -1, lamb
    
    
    

#%%
#paramvals:
""" T= 100 
paramsPath = "fake_ParamsInferredWCutoff_sod_eod_true"
todPath = "fakeData_Params_sod_eod_dictTOD_constt"
s=0
n=None
timestamps=None
spread=lob0[0]['Ask_touch'][0] - lob0[0]['Bid_touch'][0]
beta=1.0
avgSpread = .01
lamb=None
s, n, timestamps, timestamps_this, tau, lamb = thinningOgataIS(T, paramsPath, todPath, maxJumps = 1, s = s, n = n, Ts = timestamps, spread=spread, beta = beta, avgSpread = avgSpread, lamb = lamb)
 """
paramsPath = "fake_ParamsInferredWCutoff_sod_eod_true"
todPath = "fakeData_Params_sod_eod_dictTOD_constt"
with open(paramsPath, "rb") as f:
    params=pickle.load(f)
with open(todPath, "rb") as f:
    tod = pickle.load(f)
print(type(params), type(tod))
for j in tod.keys():
    print(j,": ", tod[j])