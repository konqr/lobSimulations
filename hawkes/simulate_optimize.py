import pickle
import numpy as np
import pandas as pd
import time
import numpy as np
import os 

#perf_counter() measures the real amount of time for a process to take, as if you used a stop watch. Includes I/O and sleeping

start = time.perf_counter()
time.sleep(1)
end = time.perf_counter()
print("it took" + str(end-start)+ " nanosecs.")


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

def thinningOgataIS(T, paramsPath, todPath, num_nodes = 12, maxJumps = None, s = None, n = None, Ts = None, spread=None, beta = 0.7479, avgSpread = 0.0169,lamb= None):
    
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

    cols = ["lo_deep_Ask", "co_deep_Ask", "lo_top_Ask","co_top_Ask", "mo_Ask", "lo_inspread_Ask" ,
            "lo_inspread_Bid" , "mo_Bid", "co_top_Bid", "lo_top_Bid", "co_deep_Bid","lo_deep_Bid" ]
    baselines = num_nodes*[0]
    mat = np.zeros((num_nodes, num_nodes))
    if s is None: s = 0
    hourIndex = np.min([12,int(np.floor(s/1800))])
    for i in range(num_nodes):
        for j in range(num_nodes):
            kernelParams = params.get(cols[i] + "->" + cols[j], None)
            if kernelParams is None: continue
            if np.isnan(kernelParams[1][2]): continue
            # print(cols[i] + "->" + cols[j])
            # print((kernelParams[0]*np.exp(kernelParams[1][0]) , kernelParams[1][1] , kernelParams[1][2]))
            todMult = tod[cols[j]][hourIndex]
            mat[i][j]  = todMult*kernelParams[0]*kernelParams[1][0]/((-1 + kernelParams[1][1])*kernelParams[1][2]) # alpha/(beta -1)*gamma
        baselines[i] = params[cols[i]]
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
    if lamb is None:
        print(baselines)

        decays = baselines.copy()
        for i in range(len(Ts)):
            todMult = tod[cols[i]][hourIndex]*0.99/specRad
            decays[i] = todMult*decays[i]
        if (type(baselines[0]) == np.float64)or(type(baselines[0]) == float):
            lamb = sum(decays)
        else:
            lamb = sum(np.array(baselines)[:,0])
    while s <= T:
        lambBar = lamb
        print(lambBar)
        u = np.random.uniform(0,1)
        if lambBar == 0:
            s += 0.1 # wait for some time
        else:
            w = np.max([1e-7,-1*np.log(u)/lambBar]) # floor at 0.1 microsec
            s += w

        decays = baselines.copy()
        hourIndex = np.min([12,int(np.floor(s/1800))])

        for i in range(len(Ts)):
            todMult = tod[cols[i]][hourIndex]*0.99/specRad
            decays[i] = todMult*decays[i]
        for i in range(len(Ts)):
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
        D = np.random.uniform(0,1)
        if D*lambBar <= lamb: #accepted
            print(w)
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
            hourIndex = np.min([12,int(np.floor(s/1800))])
            decays[5] = ((spread/avgSpread)**beta)*decays[5]
            decays[6] = ((spread/avgSpread)**beta)*decays[6]
            decays = decays*(s-T_Minus1)
            tau = decays[k]
            Ts[k] += (s,)
            Ts_new[k] += (s,)
            numJumps += 1
            if numJumps >= maxJumps:
                return s,n,Ts, Ts_new, tau, lamb
    return s,n, Ts, Ts_new, -1, lamb


#params:
T= 100
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
