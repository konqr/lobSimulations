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
#%%
num_nodes=12
def preprocessdata(paramsPath: str, todPath: str):
    """Takes in params and todpath and spits out corresponding vectorised numpy arrays
    
    Returns:
    tod: a [12, 13] matrix containing values of f(Q_t), the time multiplier for the 13 different 30 min bins of the trading day.
    params=[kernelparams, baselines]
        kernelparams: an array of [12, 12] matrices consisting of mask, alpha0, beta, gamma. the item at arr[i][j] corresponds to the corresponding value from params[cols[i] + "->" + cols[j]]
        baselines: a vector of dim=(num_nodes, 1) consisting of baseline intensities
    """
    
    cols = ["lo_deep_Ask", "co_deep_Ask", "lo_top_Ask","co_top_Ask", "mo_Ask", "lo_inspread_Ask" ,
            "lo_inspread_Bid" , "mo_Bid", "co_top_Bid", "lo_top_Bid", "co_deep_Bid","lo_deep_Bid" ]
    os.path.exists("")
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

# def calcIntensities(Ts, s, baselines, mask, alpha, beta, gamma, memo):
#     powerLawCutoff(alpha=)
    



if os.path.exists("hawkes"):
    pass
else:
    os.chdir("..")
tod, params=preprocessdata(paramsPath='fake_ParamsInferredWCutoff_sod_eod_true', todPath='fakeData_Params_sod_eod_dictTOD_constt')
 

#%%
def thinningOgataIS2(T, params, tod, num_nodes=12, maxJumps = None, s = None, n = None, Ts = None, timeseries=None, spread=None, beta = 0.7479, avgSpread = 0.0169,lamb= None):
    """ 
    Arguments:
    T: timelimit of simulation process
    params=[kernelparams, baselines]
        kernelparams: an array of [12, 12] matrices consisting of mask, alpha0, beta, gamma. the item at arr[i][j] corresponds to the corresponding value from params[cols[i] + "->" + cols[j]]
        baselines: a vector of dim=(num_nodes, 1) consisting of baseline intensities
    tod: a [12, 13] matrix containing values of f(Q_t), the time multiplier for the 13 different 30 min bins of the trading day.
    num_nodes: #of different processes
    timeseries: the sequence of all point-events arranged in the form (t, m) where m is the #of the dimension
    """
    numJumps = 0
    if n is None: n = num_nodes*[0]
    if Ts is None: Ts = num_nodes*[[]]
    if spread is None: spread = 1
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

    
    """calculating initial values of lamb_bar"""
    if lamb is None:
        decays=(0.99/specRad) * todmult * baselines
        lamb=np.sum(decays) #3.04895025
    left=0
    if timeseries is None:
        timeseries=[]
        
    """simulation loop"""
    while s<=T:
        """Assign lamb_bar"""
        lamb_bar=lamb 
        """generate random u"""
        u=np.random.uniform(0, 1)
        if lamb_bar==0:
            s+=0.1  # wait for some time
        else:
            w=max(1e-7, -1 * np.log(u)/lamb_bar) # floor at 0.1 microsec
            s+=w
            
        """Recalculating baseline lambdas sum with new candidate"""
        hourIndex = min(12,int(s//1800))
        todmult=tod[:, hourIndex].reshape((12, 1)) * (0.99/specRad)
        decays=todmult * baselines
        """Summing cross excitations for previous points"""
        while s-timeseries[left]>=10:
            left+=1 
        for point in timeseries[left:]:
            kern=powerLawCutoff(time=s-point[0], alpha=params[0][0][point[1]]*params[0][1][point[1]], beta=params[0][2][point[1]], gamma=params[0][3][point[1]])
            print(kern.shape) #should be (12, 1)
            kern=kern.reshape((12, 1))
            decays+=todmult*kern
        decays=np.maximum(decays, 0)
        decays[5] = ((spread/avgSpread)**beta)*decays[5]
        decays[6] = ((spread/avgSpread)**beta)*decays[6]
        if 100*np.round(spread, 2) < 2 : 
            decays[5] = decays[6] = 0
        print(decays)
        lamb = sum(decays)
        print(lamb)
        
        
        """Testing candidate point"""
        D=np.random.uniform(0, 1)
        if D*lamb_bar<=lamb:
            """Accepted so assign candidate point to a process by a ratio of intensities"""
            k=0
            total=decays[k]
            while D*lamb_bar >= total:
                k+=1
                total+=decays[k]
            """dimension is cols[k]"""   
            """Update values of lambda for next simulation loop and append point to Ts"""
            if k in [5, 6]:
                spread=spread-0.01

            """Precalc next value of lambda_bar"""    
            newdecay=todmult * params[0][0][k]*params[0][1][k]
            newdecays=np.maximum(newdecays, 0)
            newdecays[5] = ((spread/avgSpread)**beta)*newdecays[5]
            newdecays[6] = ((spread/avgSpread)**beta)*newdecays[6]
            if 100*np.round(spread, 2) < 2 : newdecays[5] = newdecays[6] = 0
            lamb+= np.sum(newdecays)


            n[k]+=1
            timeseries.append((s, k))
            Ts[k].append(s)
            
            if len(Ts[k]) > 0:
                T_Minus1 = Ts[k][-1]
            else:
                T_Minus1 = 0
            decays = np.array(baselines.copy())
            hourIndex = np.min([12,int(np.floor(s/1800))])
            decays[5] = ((spread/avgSpread)**beta)*decays[5]
            decays[6] = ((spread/avgSpread)**beta)*decays[6]
            decays = decays*(s-T_Minus1)
            
            """Updating history and returns"""
            tau = decays[k]
            numJumps+=1
            if numJumps>=maxJumps:
                return s, n, Ts, tau, lamb, timeseries
            return s, n, Ts, tau, lamb, timeseries
# time=s-tau
# alpha=kernelParams[0]*kernelParams[1][0]
# beta=kernelParams[1][1]
# gamma=kernelParams[1][2]
#%%
if __name__== "__main__":
    paramsPath = "fake_ParamsInferredWCutoff_sod_eod_true"
    todPath = "fakeData_Params_sod_eod_dictTOD_constt"

    if os.path.exists("hawkes"):
    pass
else:
    os.chdir("..")
paramsPath = "fake_ParamsInferredWCutoff_sod_eod_true"
todPath = "fakeData_Params_sod_eod_dictTOD_constt"
T=10
s, n, timestamps, timestamps_this, tau, lamb = thinningOgataIS2(T, paramsPath, todPath, maxJumps = 1)    