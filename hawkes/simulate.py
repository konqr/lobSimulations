# from tick.plot import plot_point_process
# from tick.hawkes import SimuHawkes, HawkesKernelPowerLaw
# import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd

import numpy as np

def powerLawKernel(x, alpha = 1., t0 = -1., beta = -2.):
    if x + t0 <= 0: return 0
    return alpha*((x)**beta)

def thinningOgata(T, paramsPath, num_nodes = 12):

    with open(paramsPath, "rb") as f:
        params = pickle.load(f)
    cols = ["lo_deep_Ask", "co_deep_Ask", "lo_top_Ask","co_top_Ask", "mo_Ask", "lo_inspread_Ask" ,
            "lo_inspread_Bid" , "mo_Bid", "co_top_Bid", "lo_top_Bid", "co_deep_Bid","lo_deep_Bid" ]
    baselines = num_nodes*[0]
    mat = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            kernelParams = params[cols[i] + "->" + cols[j]]
            print(cols[i] + "->" + cols[j])
            print((kernelParams[0]*np.exp(kernelParams[1][0]) , kernelParams[1][1]))
            mat[i][j]  = kernelParams[0]*np.exp(kernelParams[1][0])/((-1 - kernelParams[1][1])*(1e-4)**(-1 - kernelParams[1][1]))
        baselines[i] = params[cols[i]]
    print("spectral radius = ", np.max(np.linalg.eig(mat)[0]))
    s = 0
    n = num_nodes*[0]
    Ts = num_nodes*[()]
    lamb = sum(baselines)
    while s <= T:
        lambBar = lamb
        u = np.random.uniform(0,1)
        w = -1*np.log(u)/lambBar
        s += np.max([1e-6,w])
        decays = baselines.copy()
        for i in range(len(Ts)):
            taus = Ts[i]
            for tau in taus:
                if s - tau >= 500: continue
                if s - tau < 1e-4: continue
                for j in range(len(Ts)):
                    kernelParams = params[cols[i] + "->" + cols[j]]
                    decay = powerLawKernel(s - tau, alpha = kernelParams[0]*np.exp(kernelParams[1][0]), t0 = -1e-4, beta = kernelParams[1][1])
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
    return n, Ts


def simulate(T , paramsPath , Pis = None, Pi_Q0 = None):
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
            "Ask_deep" : [0.0012, [(1, 0.002), (10, 0.004), (50, 0.001), (100, 0.042), (200, 0.046), (500, 0.057)]]
        }
        Pi_Q0["Bid_touch"] = Pi_Q0["Ask_touch"]
        Pi_Q0["Bid_deep"] = Pi_Q0["Ask_deep"]
        # hawkes = SimuHawkes(n_nodes=12, end_time=T, verbose=True)
    # with open(paramsPath, "rb") as f:
    #     params = pickle.load(f)

    # norms = {}
    # for i in range(12):
    #     for j in range(12):
    #         kernelParams = params[cols[i] + "->" + cols[j]]
    #         kernel = HawkesKernelPowerLaw(kernelParams[0]*np.exp(kernelParams[1][0]), 8.1*1e-4, -1*kernelParams[1][1], support = 1000)
    #         print(cols[i] + "->" + cols[j])
    #         print(kernel.get_norm())
    #         norms[cols[i] + "->" + cols[j] ] = kernel.get_norm()
    #         #print(kernelParams[0]*np.exp(kernelParams[1][0]))
    #         #print(-1*kernelParams[1][1])
    #         if abs(kernel.get_norm()) >= 0.1 :
    #             hawkes.set_kernel(j,i, kernel)
    #     hawkes.set_baseline(i, params[cols[i]])
    # fig = plot_hawkes_kernel_norms(12, np.array(list(norms.values())).reshape((12,12)).T,
    #                          node_names=["LO_{ask_{+1}}", "CO_{ask_{+1}}",
    #                                      "LO_{ask_{0}}", "CO_{ask_{0}}", "MO_{ask_{0}}",
    #                                      "LO_{ask_{-1}}",
    #                                      "LO_{bid_{+1}}",
    #                                      "LO_{bid_{0}}", "CO_{bid_{0}}", "MO_{bid_{0}}",
    #                                      "LO_{bid_{-1}}", "CO_{bid_{-1}}"])
    # fig.savefig(paramsPath + "_kernels.png")
    # dt = 1e-4
    # hawkes.track_intensity(dt)
    # hawkes.simulate()
    # timestamps = hawkes.timestamps
    #
    # fig, ax = plt.subplots(12, 2, figsize=(16, 50))
    # plot_point_process(hawkes, n_points=50000, t_min=2, max_jumps=10, ax=ax[:,0])
    # plot_point_process(hawkes, n_points=50000, t_min=2, t_max=20, ax=ax[:, 1])
    # fig.savefig(paramsPath+".png")

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

    lob, lobL3 = createLOB(dictTimestamps, sizes, Pi_Q0)
    return lob, lobL3

def createLOB(dictTimestamps, sizes, Pi_Q0, priceMid0 = 100, spread0 = 10, ticksize = 0.01, numOrdersPerLevel = 10):
    lob = []
    lob_l3 = []
    T = []
    lob0 = {}
    lob0_l3 = {}
    levels = ["Ask_deep", "Ask_touch", "Bid_touch", "Bid_deep"]
    cols = ["lo_deep_Ask", "co_deep_Ask", "lo_top_Ask","co_top_Ask", "mo_Ask", "lo_inspread_Ask" ,
            "lo_inspread_Bid" , "mo_Bid", "co_top_Bid", "lo_top_Bid", "co_deep_Bid","lo_deep_Bid" ]
    colsToLevels = {
        "lo_deep_Ask" : "Ask_deep",
        "lo_top_Ask" : "Ask_touch",
        "lo_top_Bid" : "Bid_touch",
        "lo_deep_Bid" : "Bid_deep"
    }
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
                lobNew[side + "_deep"] = (lobNew[side + "_deep"][0], lobNew[side + "_deep"][1] + r['size'])
                lob_l3New[side + "_deep"] += [r['size']]
            elif "top" in r.event:
                lobNew[side + "_touch"] = (lobNew[side + "_touch"][0], lobNew[side + "_touch"][1] + r['size'])
                lob_l3New[side + "_touch"] += [r['size']]
            else: #inspread
                direction = 1
                if side == "Ask": direction = -1
                lobNew[side + "_deep"] = lobNew[side + "_touch"]
                lob_l3New[side + "_deep"] = lob_l3New[side + "_touch"]
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
                lob_l3New[side + "_touch"] = tmp
            while lobNew[side + "_touch"][1] <= 0: # queue depletion
                extraVolume = -1*lobNew[side + "_touch"][1]
                lobNew[side + "_touch"] = (lobNew[side + "_deep"][0], lobNew[side + "_deep"][1] - extraVolume)
                lob_l3New[side + "_touch"] = lob_l3New[side + "_deep"]
                if lobNew[side + "_touch"][1] > 0:
                    cumsum = np.cumsum(lob_l3New[side + "_touch"])
                    idx = np.argmax(cumsum >= extraVolume)
                    tmp = lob_l3New[side + "_touch"][idx:]
                    tmp[0] = tmp[0] - cumsum[idx] + extraVolume
                    lob_l3New[side + "_touch"] = tmp
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
                lob_l3New[side + "_touch"] = lob_l3New[side + "_deep"]
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
                tmp = (numOrdersPerLevel - 1)*[np.floor(lobNew[side + "_deep"][1]/numOrdersPerLevel)]
                lob_l3New[side + "_deep"] = [lobNew[side + "_deep"][1] - sum(tmp)] + tmp

        lob.append(lobNew.copy())
        lob_l3.append(lob_l3New.copy())
    return lob, lob_l3

