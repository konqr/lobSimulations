from tick.plot import plot_point_process
from tick.hawkes import SimuHawkes, HawkesKernelPowerLaw
import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd

def simulate(T , paramsPath , Pis , Pi_Q0):
    """
    :param T: time limit of simulations
    :param paramsPath: path of fitted params
    :param Pis: distribution of order sizes
    :param Pi_Q0: depleted queue size distribution
    """
    hawkes = SimuHawkes(n_nodes=12, end_time=T, verbose=True)
    with open(paramsPath, "rb") as f:
        params = pickle.load(f)
    cols = ["lo_deep_Ask", "co_deep_Ask", "lo_top_Ask","co_top_Ask", "mo_Ask", "lo_inspread_Ask" ,
                   "lo_inspread_Bid" , "mo_Bid", "co_top_Bid", "lo_top_Bid", "co_deep_Bid","lo_deep_Bid" ]
    for i in range(12):
        for j in range(12):
            kernelParams = params[cols[i] + "->" + cols[j]]
            kernel = HawkesKernelPowerLaw(kernelParams[0]*np.exp(kernelParams[1][0]), 1e-5, -1*kernelParams[1][1])
            hawkes.set_kernel(i,j, kernel)
        hawkes.set_baseline(i, params[cols[i]])
    dt = 1e-4
    hawkes.track_intensity(dt)
    hawkes.simulate()
    timestamps = hawkes.timestamps

    fig, ax = plt.subplots(12, 2, figsize=(16, 50))
    plot_point_process(hawkes, n_points=50000, t_min=2, max_jumps=10, ax=ax[:,0])
    plot_point_process(hawkes, n_points=50000, t_min=2, t_max=20, ax=ax[:, 1])
    fig.savefig(paramsPath+".png")

    sizes = {}
    dictTimestamps = {}
    for t, col in zip(timestamps, cols):
        if "co" in col: # handle size of cancel order in createLOB
            size = 0
        else:
            pi = Pis[col]
            cdf = np.cumsum(pi)
            a = np.random.uniform(0, 1, size = len(t))
            size = np.argmax(cdf>=a)-1
        sizes[col]  = size
        dictTimestamps[col] = timestamps

    lob = createLOB(dictTimestamps, sizes, Pi_Q0)
    return lob

def createLOB(dictTimestamps, sizes, Pi_Q0, priceMid0 = 100, spread0 = 10, ticksize = 0.01, numOrdersPerLevel = 10):
    lob = []
    lob_l3 = []
    T = []
    lob0 = {}
    lob0_l3 = {}
    levels = ["Ask_deep", "Ask_touch", "Bid_touch", "Bid_deep"]
    cols = ["lo_deep_Ask", "co_deep_Ask", "lo_top_Ask","co_top_Ask", "mo_Ask", "lo_inspread_Ask" ,
            "lo_inspread_Bid" , "mo_Bid", "co_top_Bid", "lo_top_Bid", "co_deep_Bid","lo_deep_Bid" ]
    lob0['Ask_touch'] = (priceMid0 + np.floor(spread0/2)*ticksize, 0)
    lob0['Bid_touch'] = (priceMid0 - np.ceil(spread0/2)*ticksize, 0)
    lob0['Ask_deep'] = (priceMid0 + np.floor(spread0/2)*ticksize + ticksize, 0)
    lob0['Bid_deep'] = (priceMid0 - np.ceil(spread0/2)*ticksize - ticksize, 0)
    for k, Pi in Pi_Q0.iteritems():
        cdf = np.cumsum(Pi)
        a = np.random.uniform(0, 1)
        qSize = np.argmax(cdf>=a)-1
        lob0[k] = (lob0[k][0], qSize)
    for l in levels:
        tmp = (numOrdersPerLevel - 1)*[np.floor(lob0[l][1]/numOrdersPerLevel)]
        lob0_l3[l] = [lob0[l][1] - sum(tmp)] + tmp

    dfs = []
    for event in dictTimestamps.keys():
        sizes_e = sizes[event]
        timestamps_e = dictTimestamps[event]
        dfs += [pd.DataFrame({"event" : len(timestamps_e)*[event], "time": timestamps_e, "size" : sizes_e})]
    dfs = pd.concat(dfs)
    dfs = dfs.sort_values("time")
    lob.append(lob0)
    T.append(0)
    lob_l3.append(lob0_l3)
    for i in range(len(dfs)):
        r = dfs.iloc[i]
        lobNew = lob[i]
        lob_l3New = lob_l3[i]
        T.append(r.time)
        if "Ask" in r.event :
            side = "Ask"
        else:
            side = "Bid"

        if "lo" in r.event:
            if "deep" in r.event:
                lobNew[side + "_deep"] = (lobNew[side + "_deep"][0], lobNew[side + "_deep"][1] + r.size)
                lob_l3New[side + "_deep"] += [r.size]
            elif "top" in r.event:
                lobNew[side + "_touch"] = (lobNew[side + "_touch"][0], lobNew[side + "_touch"][1] + r.size)
                lob_l3New[side + "_touch"] += [r.size]
            else: #inspread
                direction = 1
                if side == "Ask": direction = -1
                lobNew[side + "_deep"] = lobNew[side + "_touch"]
                lob_l3New[side + "_deep"] = lob_l3New[side + "_touch"]
                lobNew[side + "_touch"] = (lobNew[side + "_touch"][0] + direction*ticksize, r.size)
                lob_l3New[side + "_touch"] = [r.size]

        if "mo" in r.event:
            lobNew[side + "_touch"] = (lobNew[side + "_touch"][0], lobNew[side + "_touch"][1] - r.size)
            if lobNew[side + "_touch"][1] > 0:
                cumsum = np.cumsum(lob_l3New[side + "_touch"])
                idx = np.argmax(cumsum >= r.size)
                tmp = lob_l3New[side + "_touch"][idx:]
                tmp[0] = tmp[0] - cumsum[idx] + r.size
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
                cdf = np.cumsum(Pi_Q0[side+"_deep"])
                a = np.random.uniform(0, 1)
                qSize = np.argmax(cdf>=a)-1
                lobNew[side + "_deep"] = (lobNew[side + "_deep"][0] + direction*ticksize, qSize)
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
                cdf = np.cumsum(Pi_Q0[side+"_deep"])
                a = np.random.uniform(0, 1)
                qSize = np.argmax(cdf>=a)-1
                lobNew[side + "_deep"] = (lobNew[side + "_deep"][0] + direction*ticksize, qSize)
                tmp = (numOrdersPerLevel - 1)*[np.floor(lobNew[side + "_deep"][1]/numOrdersPerLevel)]
                lob_l3New[side + "_deep"] = [lobNew[side + "_deep"][1] - sum(tmp)] + tmp

            if lobNew[side + "_deep"][1] <= 0: # queue depletion
                direction = 1
                if side == "Bid": direction = -1
                cdf = np.cumsum(Pi_Q0[side+"_deep"])
                a = np.random.uniform(0, 1)
                qSize = np.argmax(cdf>=a)-1
                lobNew[side + "_deep"] = (lobNew[side + "_deep"][0] + direction*ticksize, qSize)
                tmp = (numOrdersPerLevel - 1)*[np.floor(lobNew[side + "_deep"][1]/numOrdersPerLevel)]
                lob_l3New[side + "_deep"] = [lobNew[side + "_deep"][1] - sum(tmp)] + tmp

        lob.append(lobNew)

    return lob

