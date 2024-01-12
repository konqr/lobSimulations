# from tick.plot import plot_point_process
# from tick.hawkes import SimuHawkes, HawkesKernelPowerLaw
# import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd

import numpy as np

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

def thinningOgata(T, paramsPath, num_nodes = 12, maxJumps = None):
    if maxJumps is None: maxJumps = np.inf
    with open(paramsPath, "rb") as f:
        params = pickle.load(f)
    cols = ["lo_deep_Ask", "co_deep_Ask", "lo_top_Ask","co_top_Ask", "mo_Ask", "lo_inspread_Ask" ,
            "lo_inspread_Bid" , "mo_Bid", "co_top_Bid", "lo_top_Bid", "co_deep_Bid","lo_deep_Bid" ]
    baselines = num_nodes*[0]
    mat = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            kernelParams = params[cols[i] + "->" + cols[j]]
            # print(cols[i] + "->" + cols[j])
            # print((kernelParams[0]*np.exp(kernelParams[1][0]) , kernelParams[1][1] , kernelParams[1][2]))
            mat[i][j]  = kernelParams[0]*np.exp(kernelParams[1][0])/((-1 - kernelParams[1][1])*(kernelParams[1][2])**(-1 - kernelParams[1][1]))
        baselines[i] = params[cols[i]]
    # print("spectral radius = ", np.max(np.linalg.eig(mat)[0]))
    s = 0
    numJumps = 0
    n = num_nodes*[0]
    Ts = num_nodes*[()]
    if type(baselines[0]) == float:
        lamb = sum(baselines)
    else:
        lamb = sum(np.array(baselines)[:,0])
    while s <= T:
        lambBar = lamb
        u = np.random.uniform(0,1)
        w = -1*np.log(u)/lambBar
        s += np.max([1e-6,w])
        if type(baselines[0]) == float:
            decays = baselines.copy()
        else:
            hourIndex = np.min([12,int(np.floor(s/1800))])
            decays = np.array(baselines)[:,hourIndex]
        for i in range(len(Ts)):
            taus = Ts[i]
            for tau in taus:
                if s - tau >= 500: continue
                if s - tau < 1e-4: continue
                for j in range(len(Ts)):
                    kernelParams = params[cols[i] + "->" + cols[j]]
                    decay = powerLawKernel(s - tau, alpha = kernelParams[0]*np.exp(kernelParams[1][0]), t0 = kernelParams[1][2], beta = kernelParams[1][1])
                    decays[j] += decay
        decays = [np.max([0, d]) for d in decays]
        lamb = sum(decays)
        D = np.random.uniform(0,1)
        if D*lambBar <= lamb: #accepted
            k = 0
            while D*lambBar >= sum(decays[:k+1]):
                k+=1
            n[k] += 1
            Ts[k] += (s,)
            numJumps += 1
            if numJumps >= maxJumps:
                return n,Ts
    return n, Ts

def thinningOgataIS(T, paramsPath, todPath, num_nodes = 12, maxJumps = None, s = None, n = None, Ts = None, spread=None, beta = 0.41, lamb= None):
    if maxJumps is None: maxJumps = np.inf
    with open(paramsPath, "rb") as f:
        params = pickle.load(f)
    with open(todPath, "rb") as f:
        tod = pickle.load(f)
    cols = ["lo_deep_Ask", "co_deep_Ask", "lo_top_Ask","co_top_Ask", "mo_Ask", "lo_inspread_Ask" ,
            "lo_inspread_Bid" , "mo_Bid", "co_top_Bid", "lo_top_Bid", "co_deep_Bid","lo_deep_Bid" ]
    baselines = num_nodes*[0]
    mat = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            kernelParams = params.get(cols[i] + "->" + cols[j], None)
            if kernelParams is None: continue
            if np.isnan(kernelParams[1][2]): continue
            # print(cols[i] + "->" + cols[j])
            # print((kernelParams[0]*np.exp(kernelParams[1][0]) , kernelParams[1][1] , kernelParams[1][2]))
            mat[i][j]  = kernelParams[0]*kernelParams[1][0]/((-1 + kernelParams[1][1])*kernelParams[1][2]) # alpha/(beta -1)*gamma
        baselines[i] = params[cols[i]]
    baselines[5] = ((spread/0.0169)**beta)*baselines[5]
    baselines[6] = ((spread/0.0169)**beta)*baselines[6]
    print("spectral radius = ", np.max(np.linalg.eig(mat)[0]))
    if s is None: s = 0
    numJumps = 0
    if n is None: n = num_nodes*[0]
    if Ts is None: Ts = num_nodes*[()]
    Ts_new = num_nodes*[()]
    if spread is None: spread = 1
    if lamb is None:
        print(baselines)
        hourIndex = np.min([12,int(np.floor(s/1800))])
        decays = baselines.copy()
        for i in range(len(Ts)):
            todMult = tod[cols[i]][hourIndex]
            decays[i] = todMult*decays[i]
        if type(baselines[0]) == np.float64:
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
            todMult = tod[cols[i]][hourIndex]
            decays[i] = todMult*decays[i]
        for i in range(len(Ts)):
            taus = Ts[i]
            for tau in taus:
                if s - tau >= 1000: continue
                #if s - tau < 1e-4: continue
                for j in range(len(Ts)):
                    kernelParams = params.get(cols[i] + "->" + cols[j], None)
                    todMult = tod[cols[j]][hourIndex]
                    if kernelParams is None: continue
                    if np.isnan(kernelParams[1][2]): continue
                    # decay = todMult*powerLawKernel(s - tau, alpha = kernelParams[0]*np.exp(kernelParams[1][0]), t0 = kernelParams[1][2], beta = kernelParams[1][1])
                    # print(decay)
                    decay = todMult*powerLawCutoff(s - tau, kernelParams[0]*kernelParams[1][0], kernelParams[1][1], kernelParams[1][2])
                    decays[j] += decay
        decays = [np.max([0, d]) for d in decays]
        decays[5] = ((spread/0.0169)**beta)*decays[5]
        decays[6] = ((spread/0.0169)**beta)*decays[6]
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
                todMult = tod[cols[i]][hourIndex]
                if kernelParams is None: continue
                if np.isnan(kernelParams[1][2]): continue
                # decay = todMult*powerLawKernel(0, alpha = kernelParams[0]*np.exp(kernelParams[1][0]), t0 = kernelParams[1][2], beta = kernelParams[1][1])
                decay = todMult*powerLawCutoff(0, kernelParams[0]*kernelParams[1][0], kernelParams[1][1], kernelParams[1][2])
                # print(decay)
                newdecays[i] += decay
            newdecays = [np.max([0, d]) for d in newdecays]
            newdecays[5] = ((spread/0.0169)**beta)*newdecays[5]
            newdecays[6] = ((spread/0.0169)**beta)*newdecays[6]
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
            decays[5] = ((spread/0.0169)**beta)*decays[5]
            decays[6] = ((spread/0.0169)**beta)*decays[6]
            decays = decays*(s-T_Minus1)
            # for i in range(len(Ts)):
            #     taus = Ts[i]
            #     for tau in taus:
            #         if s - tau >= 1000: continue
            #         #if s - tau < 1e-4: continue
            #         kernelParams = params.get(cols[i] + "->" + cols[k], None)
            #         if kernelParams is None: continue
            #         if np.isnan(kernelParams[1][2]): continue
            #         todMult = tod[cols[k]][hourIndex]
            #         decay = todMult*powerLawKernelIntegral(T_Minus1 - tau, s - tau, alpha = kernelParams[0]*np.exp(kernelParams[1][0]), t0 = kernelParams[1][2], beta = kernelParams[1][1])
            #         decays[k] += decay
            tau = decays[k]
            Ts[k] += (s,)
            Ts_new[k] += (s,)
            numJumps += 1
            if numJumps >= maxJumps:
                return s,n,Ts, Ts_new, tau, lamb
    return s,n, Ts, Ts_new, -1, lamb

def simulate(T , paramsPath , todPath, Pis = None, Pi_Q0 = None, beta = 0.7479, spread0 = 3, price0 = 260):
    """
    :param T: time limit of simulations
    :param paramsPath: path of fitted params
    :param Pis: distribution of order sizes
    :param Pi_Q0: depleted queue size distribution
    """
    cols = ["lo_deep_Ask", "co_deep_Ask", "lo_top_Ask","co_top_Ask", "mo_Ask", "lo_inspread_Ask" ,
            "lo_inspread_Bid" , "mo_Bid", "co_top_Bid", "lo_top_Bid", "co_deep_Bid","lo_deep_Bid" ]
    if Pis == None:
        Pis = {
            'mo_Bid' : [8.16e-3, [(1, 0.072), (10, 0.04), (50, 0.028), (100, 0.427), (200, 0.051), (500, 0.07)]],
            'lo_top_Bid' : [2.09e-3, [(1, 0.02), (10, 0.069), (50, 0.005), (100, 0.6), (200, 0.036), (500, 0.054)]],
            'lo_deep_Bid' : [2.33e-3, [(1, 0.021), (10, 0.112), (50, 0.015), (100, 0.276), (200, 0.097), (500, 0.172)]]
        }
        Pis["mo_Ask"] = Pis["mo_Bid"]
        Pis["lo_top_Ask"] =  Pis["lo_inspread_Ask"] = Pis["lo_inspread_Bid"] = Pis["lo_top_Bid"]
        Pis["lo_deep_Ask"] = Pis["lo_deep_Bid"]
    if Pi_Q0 == None:
        Pi_Q0 = {
            "Ask_touch" : [0.0015, [(1, 0.013), (10, 0.016), (50, 0.004), (100, 0.166), (200, 0.133), (500, 0.04)]],
            "Ask_deep" : [0.0012, [(1, 0.002), (10, 0.004), (50, 0.001), (100, 0.042), (200, 0.046), (500, 0.057), (1000,0.031 )]]
        }
        Pi_Q0["Bid_touch"] = Pi_Q0["Ask_touch"]
        Pi_Q0["Bid_deep"] = Pi_Q0["Ask_deep"]

    if "CLSLogLin" not in paramsPath:
        n, timestamps = thinningOgata(T, paramsPath)

        sizes = {}
        dictTimestamps = {}
        for t, col in zip(timestamps, cols):
            if len(t) == 0: continue
            if "co" in col: # handle size of cancel order in createLOB
                size = 0
            else:
                pi = Pis[col] #geometric + dirac deltas; pi = (p, diracdeltas(i,p_i))
                p = pi[0]
                dd = pi[1]
                pi = np.array([p*(1-p)**k for k in range(1,10000)])
                pi = pi*(1-sum([d[1] for d in dd]))/sum(pi)
                for i, p_i in dd:
                    pi[i-1] = p_i + pi[i-1]
                pi = pi/sum(pi)
                cdf = np.cumsum(pi)
                a = np.random.uniform(0, 1, size = len(t))
                if type(a) != float:
                    size =[]
                    for i in a:
                        size.append(np.argmax(cdf>=i)+1)
                else:
                    size = np.argmax(cdf>=a)+1
            sizes[col]  = size
            dictTimestamps[col] = t

        Ts, lob, lobL3 = createLOB(dictTimestamps, sizes, Pi_Q0)
    else:
        s = 0
        Ts,lob,lobL3 = [],[],[]
        _, lob0, lob0_l3 = createLOB({}, {}, Pi_Q0, priceMid0 = price0, spread0 = spread0, ticksize = 0.01, numOrdersPerLevel = 5, lob0 = {}, lob0_l3 = {})
        Ts.append(0)
        lob.append(lob0[-1])
        lobL3.append(lob0_l3[-1])
        spread = lob0[0]['Ask_touch'][0] - lob0[0]['Bid_touch'][0]
        n = None
        timestamps = None
        lob0 = lob0[0]
        lob0_l3 = lob0_l3[0]
        lamb = None
        while s <= T:
            s, n, timestamps, timestamps_this, tau, lamb = thinningOgataIS(T, paramsPath, todPath, maxJumps = 1, s = s, n = n, Ts = timestamps, spread=spread, beta = beta, lamb = lamb)
            sizes = {}
            dictTimestamps = {}
            for t, col in zip(timestamps_this, cols):
                if len(t) == 0: continue
                if "co" in col: # handle size of cancel order in createLOB
                    size = 0
                else:
                    pi = Pis[col] #geometric + dirac deltas; pi = (p, diracdeltas(i,p_i))
                    p = pi[0]
                    dd = pi[1]
                    pi = np.array([p*(1-p)**k for k in range(1,10000)])
                    pi = pi*(1-sum([d[1] for d in dd]))/sum(pi)
                    for i, p_i in dd:
                        pi[i-1] = p_i + pi[i-1]
                    pi = pi/sum(pi)
                    cdf = np.cumsum(pi)
                    a = np.random.uniform(0, 1, size = len(t))
                    if type(a) != float:
                        size =[]
                        for i in a:
                            size.append(np.argmax(cdf>=i)+1)
                    else:
                        size = np.argmax(cdf>=a)+1
                sizes[col]  = size
                dictTimestamps[col] = t

            TsTmp, lobTmp, lobL3Tmp = createLOB(dictTimestamps, sizes, Pi_Q0, lob0 = lob0, lob0_l3 = lob0_l3)
            spread = lobTmp[-1]['Ask_touch'][0] - lobTmp[-1]['Bid_touch'][0]
            lob0 = lobTmp[-1]
            lob0_l3 = lobL3Tmp[-1]
            Ts.append([list(dictTimestamps.keys())[0], TsTmp[-1], tau])
            lob.append(lob0)
            lobL3.append(lob0_l3)
            with open("/SAN/fca/Konark_PhD_Experiments/simulated/AAPL.OQ_ResultsWCutoff_2019-01-02_2019-03-31_CLSLogLin_10_tmp" , "ab") as f: #"/home/konajain/params/"
                pickle.dump(([list(dictTimestamps.keys())[0], TsTmp[-1], tau], lob0, lob0_l3), f)
    return Ts, lob, lobL3



def createLOB(dictTimestamps, sizes, Pi_Q0, priceMid0 = 260, spread0 = 4, ticksize = 0.01, numOrdersPerLevel = 10, lob0 = {}, lob0_l3 = {}):
    lob = []
    lob_l3 = []
    T = []
    levels = ["Ask_deep", "Ask_touch", "Bid_touch", "Bid_deep"]
    cols = ["lo_deep_Ask", "co_deep_Ask", "lo_top_Ask","co_top_Ask", "mo_Ask", "lo_inspread_Ask" ,
            "lo_inspread_Bid" , "mo_Bid", "co_top_Bid", "lo_top_Bid", "co_deep_Bid","lo_deep_Bid" ]
    colsToLevels = {
        "lo_deep_Ask" : "Ask_deep",
        "lo_top_Ask" : "Ask_touch",
        "lo_top_Bid" : "Bid_touch",
        "lo_deep_Bid" : "Bid_deep"
    }
    if len(lob0) == 0:
        lob0['Ask_touch'] = (priceMid0 + np.floor(spread0/2)*ticksize, 0)
        lob0['Bid_touch'] = (priceMid0 - np.ceil(spread0/2)*ticksize, 0)
        lob0['Ask_deep'] = (priceMid0 + np.floor(spread0/2)*ticksize + ticksize, 0)
        lob0['Bid_deep'] = (priceMid0 - np.ceil(spread0/2)*ticksize - ticksize, 0)
        for k, pi in Pi_Q0.items():
            #geometric + dirac deltas; pi = (p, diracdeltas(i,p_i))
            p = pi[0]
            dd = pi[1]
            pi = np.array([p*(1-p)**k for k in range(1,100000)])
            pi = pi*(1-sum([d[1] for d in dd]))/sum(pi)
            for i, p_i in dd:
                pi[i-1] = p_i + pi[i-1]
            pi = pi/sum(pi)
            cdf = np.cumsum(pi)
            a = np.random.uniform(0, 1)
            qSize = np.argmax(cdf>=a) + 1
            lob0[k] = (lob0[k][0], qSize)
        for l in levels:
            tmp = (numOrdersPerLevel - 1)*[np.floor(lob0[l][1]/numOrdersPerLevel)]
            if tmp !=0:
                lob0_l3[l] = [lob0[l][1] - sum(tmp)] + tmp
            else:
                lob0_l3[l] = [lob0[l][1]]
    if len(dictTimestamps) == 0:
        return T, [lob0], [lob0_l3]

    dfs = []
    for event in dictTimestamps.keys():
        sizes_e = sizes[event]
        timestamps_e = dictTimestamps[event]
        dfs += [pd.DataFrame({"event" : len(timestamps_e)*[event], "time": timestamps_e, "size" : sizes_e})]
    dfs = pd.concat(dfs)
    dfs = dfs.sort_values("time")
    print(dfs.head())
    lob.append(lob0.copy())
    T.append(0)
    lob_l3.append(lob0_l3.copy())
    for i in range(len(dfs)):
            r = dfs.iloc[i]
            lobNew = lob[i].copy()
            lob_l3New = lob_l3[i].copy()
            T.append(r.time)
            if "Ask" in r.event :
                side = "Ask"
            else:
                side = "Bid"

            if "lo" in r.event:
                if "deep" in r.event:
                    if np.abs(lobNew[side + "_touch"][0] - lobNew[side + "_deep"][0]) <  2*ticksize:
                        direction = 1
                        if side == "Ask": direction = -1
                        lobNew[side + "_deep"] = (np.round(lobNew[side + "_touch"][0] - direction*ticksize, decimals=2), r['size'])
                        lob_l3New[side + "_deep"] = [r['size']]
                    else:
                        lobNew[side + "_deep"] = (lobNew[side + "_deep"][0], lobNew[side + "_deep"][1] + r['size'])
                        lob_l3New[side + "_deep"] += [r['size']]
                elif "top" in r.event:
                    lobNew[side + "_touch"] = (lobNew[side + "_touch"][0], lobNew[side + "_touch"][1] + r['size'])
                    lob_l3New[side + "_touch"] += [r['size']]
                else: #inspread
                    direction = 1
                    if side == "Ask": direction = -1
                    lobNew[side + "_deep"] = lobNew[side + "_touch"]
                    lob_l3New[side + "_deep"] = lob_l3New[side + "_touch"].copy()
                    lobNew[side + "_touch"] = (np.round(lobNew[side + "_touch"][0] + direction*ticksize, decimals=2), r['size'])
                    lob_l3New[side + "_touch"] = [r['size']]

            if "mo" in r.event:
                lobNew[side + "_touch"] = (lobNew[side + "_touch"][0], lobNew[side + "_touch"][1] - r['size'])
                if lobNew[side + "_touch"][1] > 0:
                    cumsum = np.cumsum(lob_l3New[side + "_touch"])
                    idx = np.argmax(cumsum >= r['size'])
                    tmp = lob_l3New[side + "_touch"][idx:]
                    offset = 0
                    if idx > 0: offset = cumsum[idx - 1]
                    tmp[0] = tmp[0] + offset - r['size']
                    lob_l3New[side + "_touch"] = tmp.copy()
                while lobNew[side + "_touch"][1] <= 0: # queue depletion
                    extraVolume = -1*lobNew[side + "_touch"][1]
                    lobNew[side + "_touch"] = (lobNew[side + "_deep"][0], lobNew[side + "_deep"][1] - extraVolume)
                    lob_l3New[side + "_touch"] = lob_l3New[side + "_deep"].copy()
                    if lobNew[side + "_touch"][1] > 0:
                        if extraVolume > 0:
                            cumsum = np.cumsum(lob_l3New[side + "_touch"])
                            idx = np.argmax(cumsum >= extraVolume)
                            tmp = np.array(lob_l3New[side + "_touch"][idx:])
                            tmp[0] = cumsum[idx] - extraVolume
                            tmp = tmp[tmp>0]
                            lob_l3New[side + "_touch"] = list(tmp).copy()
                    direction = 1
                    if side == "Bid": direction = -1
                    #geometric + dirac deltas; pi = (p, diracdeltas(i,p_i))
                    pi = Pi_Q0[side+"_deep"]
                    p = pi[0]
                    dd = pi[1]
                    pi = np.array([p*(1-p)**k for k in range(1,100000)])
                    pi = pi*(1-sum([d[1] for d in dd]))/sum(pi)
                    for i, p_i in dd:
                        pi[i-1] = p_i + pi[i-1]
                    pi = pi/sum(pi)
                    cdf = np.cumsum(pi)
                    a = np.random.uniform(0, 1)
                    qSize = np.argmax(cdf>=a)+1
                    lobNew[side + "_deep"] = (np.round(lobNew[side + "_deep"][0] + direction*ticksize, decimals=2), qSize)
                    tmp = (numOrdersPerLevel - 1)*[np.floor(lobNew[side + "_deep"][1]/numOrdersPerLevel)]
                    lob_l3New[side + "_deep"] = [lobNew[side + "_deep"][1] - sum(tmp)] + tmp

            if "co" in r.event:

                if "deep" in r.event:
                    size = np.random.choice(lob_l3New[side + "_deep"])
                    lobNew[side + "_deep"] = (lobNew[side + "_deep"][0], lobNew[side + "_deep"][1] - size)
                    lob_l3New[side + "_deep"].remove(size)
                elif "top" in r.event:
                    size = np.random.choice(lob_l3New[side + "_touch"])
                    lobNew[side + "_touch"] = (lobNew[side + "_touch"][0], lobNew[side + "_touch"][1] - size)
                    lob_l3New[side + "_touch"].remove(size)
                if lobNew[side + "_touch"][1] <= 0: # queue depletion
                    lobNew[side + "_touch"] = (lobNew[side + "_deep"][0], lobNew[side + "_deep"][1])
                    lob_l3New[side + "_touch"] = lob_l3New[side + "_deep"].copy()
                    direction = 1
                    if side == "Bid": direction = -1
                    pi = Pi_Q0[side+"_deep"]
                    p = pi[0]
                    dd = pi[1]
                    pi = np.array([p*(1-p)**k for k in range(1,100000)])
                    pi = pi*(1-sum([d[1] for d in dd]))/sum(pi)
                    for i, p_i in dd:
                        pi[i-1] = p_i + pi[i-1]
                    pi = pi/sum(pi)
                    cdf = np.cumsum(pi)
                    a = np.random.uniform(0, 1)
                    qSize = np.argmax(cdf>=a)-1
                    lobNew[side + "_deep"] = (np.round(lobNew[side + "_deep"][0] + direction*ticksize, decimals=2), qSize)
                    tmp = (numOrdersPerLevel - 1)*[np.floor(lobNew[side + "_deep"][1]/numOrdersPerLevel)]
                    lob_l3New[side + "_deep"] = [lobNew[side + "_deep"][1] - sum(tmp)] + tmp

                if lobNew[side + "_deep"][1] <= 0: # queue depletion
                    direction = 1
                    if side == "Bid": direction = -1
                    #geometric + dirac deltas; pi = (p, diracdeltas(i,p_i))
                    pi = Pi_Q0[side+"_deep"]
                    p = pi[0]
                    dd = pi[1]
                    pi = np.array([p*(1-p)**k for k in range(1,100000)])
                    pi = pi*(1-sum([d[1] for d in dd]))/sum(pi)
                    for i, p_i in dd:
                        pi[i-1] = p_i + pi[i-1]
                    pi = pi/sum(pi)
                    cdf = np.cumsum(pi)
                    a = np.random.uniform(0, 1)
                    qSize = np.argmax(cdf>=a)+1
                    lobNew[side + "_deep"] = (np.round(lobNew[side + "_deep"][0] + direction*ticksize, decimals=2), qSize)
                    tmp = ((2*numOrdersPerLevel) - 1)*[np.floor(lobNew[side + "_deep"][1]/(2*numOrdersPerLevel))]
                    lob_l3New[side + "_deep"] = [lobNew[side + "_deep"][1] - sum(tmp)] + tmp
            lob.append(lobNew.copy())
            lob_l3.append(lob_l3New.copy())
    return T, lob, lob_l3

