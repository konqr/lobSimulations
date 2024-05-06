import pandas as pd
import numpy as np
import os
import pickle
import time
import datetime as dt
from scipy import stats
import statsmodels.api as sm
from hawkes import dataLoader
import matplotlib.pyplot as plt
import statsmodels

# We plan to make use of inter-event durations' Q-Q plots, signature plots, distribution of spread and returns, average shape of the book,
# autocorrelation of returns and order flow, and sample price paths as our set of stylized facts.
# TOD check - flow, spread
def cleanMOs(data):
    # remove walking orders
    dataNoMO = data.loc[(data.event!="mo_Ask")&(data.event != "mo_Bid")]
    dataMOAsk = data.loc[data.event=="mo_Ask"]
    dataMOAsk = dataMOAsk.loc[dataMOAsk.Time.diff() != 0]
    dataMOBid = data.loc[data.event=="mo_Bid"]
    dataMOBid = dataMOBid.loc[dataMOBid.Time.diff() != 0]
    return pd.concat([dataNoMO, dataMOBid, dataMOAsk])

def runQQInterArrival(ric, sDate, eDate, resultsPath, delta = 1e-1, inputDataPath = "/SAN/fca/Konark_PhD_Experiments/extracted/", avgSpread = 0.0169, spreadBeta =0.7479):
    paramsPath = inputDataPath +ric+"_ParamsInferredWCutoff_2019-01-02_2019-03-31_CLSLogLin_10"
    todPath = inputDataPath +ric+"_Params_2019-01-02_2019-03-29_dictTOD"
    with open(paramsPath, "rb") as f:
        params = pickle.load(f)
    with open(todPath, "rb") as f:
        tod = pickle.load(f)
    cols = ["lo_deep_Ask", "co_deep_Ask", "lo_top_Ask","co_top_Ask", "mo_Ask", "lo_inspread_Ask" ,
            "lo_inspread_Bid" , "mo_Bid", "co_top_Bid", "lo_top_Bid", "co_deep_Bid","lo_deep_Bid" ]
    num_nodes = len(cols)
    baselines = {}
    paramsDict = {}
    for i in range(num_nodes):
        for j in range(num_nodes):
            kernelParams = params.get(cols[i] + "->" + cols[j], None)
            if kernelParams is None: continue
            paramsDict[cols[i] + "->" + cols[j]]  = (kernelParams[0]*kernelParams[1][0], kernelParams[1][1] , kernelParams[1][2])
        baselines[cols[i]] = params[cols[i]]
    datas = []
    # timesLinspace = np.linspace(0, 23400, int(23400/delta))
    rounder= -1*int(np.round(np.log(delta)/np.log(10)))
    for d in pd.date_range(sDate, eDate):

        inputCsvPath = ric + "_" + d.strftime("%Y-%m-%d") + "_12D.csv"
        if inputCsvPath not in os.listdir(inputDataPath): continue
        print(d)
        logger_time=time.time()
        print(logger_time)
        data = pd.read_csv(inputDataPath + "/" + inputCsvPath)
        data = cleanMOs(data)
        data["Tminus1"] = 0
        data['Tminus1'].iloc[1:] = data['Time'].iloc[:-1] #floor T - Tminus1 to zero
        data = data.sort_values(["Time", "OrderID"])
        data['prevSpread'] = data['Ask Price 1'] - data['Bid Price 1'] + data['BidDiff'] - data['AskDiff']
        tracked_intensity_integrals = [np.array(list(baselines.values()))*0]
        for r_i in data.iterrows():
            r_i = r_i[1]
            t = r_i.Time
            # print(t)
            mat = np.zeros((num_nodes, num_nodes))
            s = t
            hourIndex = np.min([12,int(np.floor(s/1800))])
            for i in range(num_nodes):
                for j in range(num_nodes):
                    kernelParams = params.get(cols[i] + "->" + cols[j], None)
                    if kernelParams is None: continue
                    todMult = tod[cols[j]][hourIndex]
                    mat[i][j]  = todMult*kernelParams[0]*kernelParams[1][0]/((-1 + kernelParams[1][1])*kernelParams[1][2]) # alpha/(beta -1)*gamma
            specRad = np.max(np.linalg.eig(mat)[0])
            # print("spectral radius = ", specRad)
            specRad = np.max(np.linalg.eig(mat)[0]).real
            if specRad < 1 : specRad = 0.99 # dont change actual specRad if already good
            df = data.loc[data.Time < t]
            hourIdx = np.min([12,int(np.floor(t/1800))])
            if len(df) == 0:
                tracked_intensity_integrals = tracked_intensity_integrals + [np.array(list(baselines.values()))*t]
                continue
            tracked_intensity_integral = []
            for col in cols:
                mult = 1
                if "inspread" in col:
                    currSpread = df.iloc[-1].prevSpread
                    mult = (currSpread/avgSpread)**spreadBeta
                intensity_integral = baselines[col]*t
                for r in df.iterrows():
                    r = r[1]
                    t_j = r.Time
                    if r.event + '->' + col in paramsDict.keys():
                        _params = paramsDict[r.event + '->' + col]
                        # if _params[0] < 0: continue
                        # print(intensity)
                        # print((t - r.Time))
                        # print(_params)
                        intensity_integral += _params[0]*(1 - (1 + _params[2]*(t - t_j))**(1 - _params[1]))/(_params[2]*(_params[1] -1))
                # print(intensity)
                tracked_intensity_integral = tracked_intensity_integral + [mult*tod[col][hourIdx]*intensity_integral*0.99/specRad]
            with open(resultsPath + "/"+ric + "_" + d.strftime("%Y-%m-%d") + "_" + d.strftime("%Y-%m-%d") + "_qq", "ab") as f:
                pickle.dump([r_i, tracked_intensity_integral], f)
            tracked_intensity_integrals = tracked_intensity_integrals + [tracked_intensity_integral]
        tracked_intensity_integrals = np.array(tracked_intensity_integrals)
        print(len(tracked_intensity_integrals))
        print(time.time() - logger_time)
        logger_time = time.time()
        data[[c + " integral" for c in cols]] = tracked_intensity_integrals
        data[["Time", "OrderID", "event"] + [c + " integral" for c in cols]].to_csv(resultsPath + "/"+ric + "_" + d.strftime("%Y-%m-%d") +"_QQdf.csv")
        # datas.append(data[["Time", "event", "lambdaIntegral"]])
        # dictLambdaIntegral = data[["event", "lambdaIntegral"]].groupby("event").apply(np.array).to_dict()
        # for k in dictLambdaIntegral.keys():
        #     fig = sm.qqplot(dictLambdaIntegral[k], stats.expon, line='45', ylabel=  k)
        #     fig.savefig(resultsPath + "/"+ric + "_" + d.strftime("%Y-%m-%d") + "_qq_" + k + ".png")
    return

def runSignaturePlots(paths, resultsPath, ric, sDate, eDate, inputDataPath = "/SAN/fca/Konark_PhD_Experiments/extracted"):
    rvs = {}
    count = 0
    if os.path.isfile(resultsPath + "/"+ric + "_" + sDate.strftime("%Y-%m-%d") + "_" + eDate.strftime("%Y-%m-%d") + "_signatureDictEmpirical"):
        with open(resultsPath + "/"+ric + "_" + sDate.strftime("%Y-%m-%d") + "_" + eDate.strftime("%Y-%m-%d") + "_signatureDictEmpirical", "rb") as f:
            rvs = pickle.load(f)
        count = len(rvs[1])//(23400 -2 -1)
    else:
        for d in pd.date_range(sDate,eDate):

            l = dataLoader.Loader(ric, d, d, nlevels = 2, dataPath = "/SAN/fca/Konark_PhD_Experiments/extracted/")
            data = l.load()
            if len(data): data = data[0]
            else: continue
            count +=1
            data['mid'] = 0.5*(data['Ask Price 1'] + data['Bid Price 1'])
            times = data.Time.values - 34200
            mid = data.mid.values
            # rv = []
            for t in range(1,2001):
                sample_x = np.linspace(0, 23400, int(23400/t))
                idxs = np.searchsorted(times, sample_x)[1:-1] - 1
                # print(idxs)
                sample_y = mid[idxs]
                rvs[t] =  np.hstack([rvs.get(t, np.array([])), np.square(np.exp(np.diff(np.log(sample_y))) - 1)])
        with open(resultsPath + "/"+ric + "_" + sDate.strftime("%Y-%m-%d") + "_" + eDate.strftime("%Y-%m-%d") + "_signatureDictEmpirical", "wb") as f:
            pickle.dump(rvs, f)
    fig = plt.figure()
    plt.title(ric + " signature plot")
    plt.xlabel("Sampling Frequency (seconds)")
    plt.ylabel("Realized Variance")
    plt.scatter(list(rvs.keys()), [np.sum(l)/(count*23400) for l in list(rvs.values())], s = 2)
    fig.savefig(resultsPath + "/"+ric + "_" + sDate.strftime("%Y-%m-%d") + "_" + eDate.strftime("%Y-%m-%d") + "_signatureScatterEmpirical.png")
    rvsSim = {}
    countSim = 0
    for path in paths:
        print(path)
        tryer= 0
        while tryer < 5: # retry on pickle clashes
            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)
                tryer = 6
            except:
                time.sleep(1)
                tryer +=1

        countSim += 1
        times = np.append([0], np.array(data[0][1:])[:,1])
        lob = data[1]
        mid = np.array([0.5*(r['Ask_touch'][0] + r['Bid_touch'][0]) for r in lob])
        for t in range(1,2001):
            sample_x = np.linspace(0, 23400, int(23400/t))
            idxs = np.searchsorted(times, sample_x)[1:-1] - 1
            # print(idxs)
            sample_y = mid[idxs]
            rvsSim[t] =  np.hstack([rvsSim.get(t, np.array([])), np.square(np.exp(np.diff(np.log(sample_y))) - 1)])
    with open(resultsPath + "/"+ric + "_" + sDate.strftime("%Y-%m-%d") + "_" + eDate.strftime("%Y-%m-%d") + "_signatureDictSimulated", "wb") as f:
        pickle.dump(rvsSim, f)
    fig = plt.figure()
    plt.title(ric + " signature plot")
    plt.xlabel("Sampling Frequency (seconds)")
    plt.ylabel("Realized Variance")
    plt.scatter(list(rvs.keys()), [np.sum(l)/(count*23400) for l in list(rvs.values())], s = 2, label = "Empirical")
    plt.scatter(list(rvsSim.keys()), [np.sum(l)/(countSim*23400) for l in list(rvsSim.values())], s = 2, label = "Simulated")
    plt.legend()
    fig.savefig(resultsPath + "/"+ric + "_" + sDate.strftime("%Y-%m-%d") + "_" + eDate.strftime("%Y-%m-%d") + "_signatureScatterSimulated.png")
    return

def runDistribution(paths, resultsPath, sDate, eDate, ric):
    simRets = []
    simSpreads =[]
    simTimes = []
    t = .01

    for path in paths:
        print(path)
        tryer= 0
        while tryer < 5: # retry on pickle clashes
            try:
                with open(path, "rb") as f:
                    results = pickle.load(f)
                tryer = 6
            except:
                time.sleep(1)
                tryer +=1
        simDf = pd.DataFrame(results[1])
        simDf['Ask'] = simDf['Ask_touch'].apply(lambda x: x[0])
        simDf['Bid'] =  simDf['Bid_touch'].apply(lambda x: x[0])
        simDf['Mid'] = 0.5*(simDf['Ask'] + simDf['Bid'])
        simDf['Spread'] = simDf['Ask'] - simDf['Bid']
        mid = simDf.Mid.values

        times = np.append([0], np.array(results[0][1:])[:,1]).astype(float)
        sample_x = np.linspace(times[1], times[-1], int(23400/t))
        idxs = np.searchsorted(times, sample_x)[1:-1] - 1
        sample_y = mid[idxs]
        ret =  np.exp(np.diff(np.log(sample_y))) - 1
        simRets.append(ret)
        simSpreads.append(simDf.Spread.values)
        simTimes.append(times)

    empRets = []
    empSpreads = []
    for d in pd.date_range(sDate,eDate):
        l = dataLoader.Loader(ric, d, d, nlevels = 2, dataPath = "/SAN/fca/Konark_PhD_Experiments/extracted/")
        data = l.load()
        if len(data): data = data[0]
        else: continue
        data['Mid'] = 0.5*(data['Ask Price 1'] + data['Bid Price 1'])
        data['Spread'] = (data['Ask Price 1'] - data['Bid Price 1'])
        mid = data.Mid.values
        times = data.Time.values - 34200
        # sample_x = np.linspace(0, 23400, int(23400/t))
        idxs = np.searchsorted(times, sample_x)[1:-1] - 1
        sample_y = mid[idxs]
        ret =  np.exp(np.diff(np.log(sample_y))) - 1
        empRets.append(ret)
        empSpreads.append(data.Spread.values)
        # print(data.Spread.values)

    fig = plt.figure()
    plt.title(ric + " returns distribution")
    plt.xlabel("Returns")
    plt.ylabel("Frequency")
    plt.hist(np.hstack(empRets), bins = 100, label = "Empirical", alpha = 0.5, density = True)
    plt.hist(np.hstack(simRets), bins = 100, label = "Simulated", alpha = 0.5, density = True)
    plt.yscale("log")
    plt.legend()
    fig.savefig(resultsPath + "/"+ric + "_" + sDate.strftime("%Y-%m-%d") + "_" + eDate.strftime("%Y-%m-%d") + "_returnsDistribution.png")

    # Get histogram
    bins, freq = np.unique(np.round(100*np.hstack(empSpreads)), return_counts = True)
    histEmp = freq/sum(freq)
    simTimeIds = np.hstack([(i[1:] - 34200)//(1800) for i in simTimes])
    ids, counts = np.unique(simTimeIds, return_counts =True)
    counts = counts/sum(counts)
    dictNorm = {}
    for i, c in zip(ids, counts):
        dictNorm[i] = 1/(13*c)
    wts = [dictNorm[i] for i in simTimeIds]
    freq, binsSim = np.histogram(np.round(100*np.hstack([i[1:] for i in simSpreads])), bins = np.unique(np.round(100*np.hstack(simSpreads))), weights = wts, density=True)
    histSim = freq/sum(freq)
    # Threshold frequency
    freq = 1e-4

    # Zero out low values
    histEmp[np.where(histEmp <= freq)] = 0
    histSim[np.where(histSim <= freq)] = 0
    bins = bins[np.append(np.where(histEmp > 0)[0], [np.where(histEmp > 0)[0][-1]+1])]
    histEmp = histEmp[np.where(histEmp > 0)]
    binsSim = binsSim[np.append(np.where(histSim > 0)[0], [np.where(histSim > 0)[0][-1]+1])]
    histSim = histSim[np.where(histSim > 0)]
    # Plot
    width = 0.99 * (bins[1] - bins[0])
    center = bins[:-1]
    fig = plt.figure()
    plt.title(ric + " spreads distribution")
    plt.xlabel("Spread-in-ticks")
    plt.ylabel("Frequency")
    plt.bar(center, histEmp, align='center', width=width, label = "Empirical", alpha = 0.5)

    widthSim = 0.99 * (binsSim[1] - binsSim[0])
    centerSim = binsSim[:-1]
    plt.bar(centerSim, histSim, align='center', width=widthSim, label = "Simulated", alpha = 0.5)
    plt.yscale("log")
    plt.legend()
    fig.savefig(resultsPath + "/"+ric + "_" + sDate.strftime("%Y-%m-%d") + "_" + eDate.strftime("%Y-%m-%d") + "_spreadDistribution.png")

    return

def runACF(paths, resultsPath, sDate, eDate, ric):

    simRets = []
    t = .01
    sample_x = np.linspace(0, 23400, int(23400/t))
    for path in paths:
        print(path)
        tryer= 0
        while tryer < 5: # retry on pickle clashes
            try:
                with open(path, "rb") as f:
                    results = pickle.load(f)
                tryer = 6
            except:
                time.sleep(1)
                tryer +=1

        simDf = pd.DataFrame(results[1])
        simDf['Ask'] = simDf['Ask_touch'].apply(lambda x: x[0])
        simDf['Bid'] =  simDf['Bid_touch'].apply(lambda x: x[0])
        simDf['Mid'] = 0.5*(simDf['Ask'] + simDf['Bid'])
        mid = simDf.Mid.values
        times = np.append([0], np.array(results[0][1:])[:,1])

        idxs = np.searchsorted(times, sample_x)[1:-1] - 1
        sample_y = mid[idxs]
        ret =  np.exp(np.diff(np.log(sample_y))) - 1
        simRets.append(ret)
    empRets = []
    for d in pd.date_range(sDate,eDate):
        if d == dt.date(2019,1,9): continue
        l = dataLoader.Loader(ric, d, d, nlevels = 2, dataPath = "/SAN/fca/Konark_PhD_Experiments/extracted/")
        data = l.load()
        if len(data): data = data[0]
        else: continue
        data['Mid'] = 0.5*(data['Ask Price 1'] + data['Bid Price 1'])
        mid = data.Mid.values
        times = data.Time.values - 34200
        # sample_x = np.linspace(0, 23400, int(23400/t))
        idxs = np.searchsorted(times, sample_x)[1:-1] - 1
        sample_y = mid[idxs]
        ret =  np.exp(np.diff(np.log(sample_y))) - 1
        empRets.append(ret)

    fig = plt.figure()
    plt.title(ric + " absolute returns ACF")
    plt.xlabel("Lags")
    plt.ylabel("Autocorrelation")
    for r in empRets:
        emps = plt.plot(statsmodels.tsa.stattools.acf(np.abs(r), nlags = 10000)[1:], color = "blue", alpha=0.5)
    for r in simRets:
        sims = plt.plot(statsmodels.tsa.stattools.acf(np.abs(r), nlags = 10000)[1:], color = "orange", alpha=0.5)
    # plt.yscale("log")
    plt.legend([emps[0], sims[0]], ['Empirical', 'Simulated'])
    fig.savefig(resultsPath + "/"+ric + "_" + sDate.strftime("%Y-%m-%d") + "_" + eDate.strftime("%Y-%m-%d") + "_absReturnsACF.png")
    return

def runPricePaths(paths, resultsPath, sDate, eDate, ric):

    simMids = []
    simTimes = []
    for path in paths:
        print(path)
        tryer= 0
        while tryer < 5: # retry on pickle clashes
            try:
                with open(path, "rb") as f:
                    results = pickle.load(f)
                tryer =6
            except:
                time.sleep(1)
                tryer +=1

        simDf = pd.DataFrame(results[1])
        simDf['Ask'] = simDf['Ask_touch'].apply(lambda x: x[0])
        simDf['Bid'] =  simDf['Bid_touch'].apply(lambda x: x[0])
        simDf['Mid'] = 0.5*(simDf['Ask'] + simDf['Bid'])
        mid = simDf.Mid.values
        simMids.append(mid)
        simTimes.append(np.append([0], np.array(results[0][1:])[:,1]).astype(float) + 9.5*3600)
    empMids = []
    empTimes = []
    for d in pd.date_range(sDate,eDate):
        if d == dt.date(2019,1,9): continue
        l = dataLoader.Loader(ric, d, d, nlevels = 2, dataPath = "/SAN/fca/Konark_PhD_Experiments/extracted/")
        data = l.load()
        if len(data): data = data[0]
        else: continue
        data['Mid'] = 0.5*(data['Ask Price 1'] + data['Bid Price 1'])
        mid = data.Mid.values
        empMids.append(mid*160/mid[0])
        empTimes.append(data.Time.values.astype(float))

    fig = plt.figure()
    plt.title(ric + " Price Paths (Simulated)")
    plt.xlabel("Price")
    plt.ylabel("Time")
    alpha = 0.1
    count = 0
    for r, t in zip(simMids, simTimes):
        if (count < 10)&(np.random.uniform() < 0.01):
            alpha = 0.5
            count +=1
        plt.plot(t, r, color = "orange", alpha=alpha)
    count = 23400
    plt.xticks(ticks = 9.5*3600 + np.arange(0, count, 2340), labels = [time.strftime('%H:%M:%S', time.gmtime(x)) for x in 9.5*3600 + np.arange(0, count, 2340)], rotation = 20)
    fig.savefig(resultsPath + "/"+ric + "_" + sDate.strftime("%Y-%m-%d") + "_" + eDate.strftime("%Y-%m-%d") + "_simulatedMidPrices.png")
    fig = plt.figure()
    plt.title(ric + " Price Paths (Empirical)")
    plt.xlabel("Price")
    plt.ylabel("Time")
    alpha = 0.1
    count = 0
    for r, t in zip(empMids, empTimes):
        if (count < 10)&(np.random.uniform() < 0.01):
            alpha = 0.5
            count += 1
        plt.plot(t, r, color = "blue", alpha=alpha)
    count = 23400
    plt.xticks(ticks = 9.5*3600 + np.arange(0, count, 2340), labels = [time.strftime('%H:%M:%S', time.gmtime(x)) for x in 9.5*3600 + np.arange(0, count, 2340)], rotation = 20)
    fig.savefig(resultsPath + "/"+ric + "_" + sDate.strftime("%Y-%m-%d") + "_" + eDate.strftime("%Y-%m-%d") + "_empiricalMidPrices.png")
    return

def runTODCheck(paths, resultsPath, sDate, eDate,ric):
    count = len(paths)
    res = None
    for path in paths:
        print(path)
        tryer= 0
        while tryer < 5: # retry on pickle clashes
            try:
                with open(path, "rb") as f:
                    results = pickle.load(f)
                tryer  = 6
            except:
                time.sleep(1)
                tryer +=1

        df = pd.DataFrame(np.array(results[0][1:]), columns = ["event", "time", "x"])
        df['tod'] = df.time.astype(float).apply(lambda x: np.min([12,int(np.floor(x/1800))]))
        n = df.groupby(["event","tod"]).count()
        if res is None:
            res = n
        else:
            res += n
    res = res/count
    for c in df.event.unique():
        fig = plt.figure()
        plt.plot(res.loc[c].reset_index().tod, res.loc[c].reset_index().time)
        plt.ylabel("Avg num orders")
        plt.xlabel("Half-Hour Index")
        plt.title(c + " TOD (Simulated)")
        fig.savefig(resultsPath + "/"+ric + "_" + sDate.strftime("%Y-%m-%d") + "_" + eDate.strftime("%Y-%m-%d") + "_" +c +"tod.png")
    return

def runQQInterArrivalTrapezoid(ric, sDate, eDate, resultsPath, delta = 1e-1, inputDataPath = "/SAN/fca/Konark_PhD_Experiments/extracted/", avgSpread = 0.0169, spreadBeta =0.7479):
    paramsPath = inputDataPath +ric+"_ParamsInferredWCutoff_2019-01-02_2019-03-31_CLSLogLin_10"
    todPath = inputDataPath +ric+"_Params_2019-01-02_2019-03-29_dictTOD"
    with open(paramsPath, "rb") as f:
        params = pickle.load(f)
    with open(todPath, "rb") as f:
        tod = pickle.load(f)
    cols = ["lo_deep_Ask", "co_deep_Ask", "lo_top_Ask","co_top_Ask", "mo_Ask", "lo_inspread_Ask" ,
            "lo_inspread_Bid" , "mo_Bid", "co_top_Bid", "lo_top_Bid", "co_deep_Bid","lo_deep_Bid" ]
    num_nodes = len(cols)
    baselines = {}
    paramsDict = {}
    for i in range(num_nodes):
        for j in range(num_nodes):
            kernelParams = params.get(cols[i] + "->" + cols[j], None)
            if kernelParams is None: continue
            paramsDict[cols[i] + "->" + cols[j]]  = (kernelParams[0]*kernelParams[1][0], kernelParams[1][1] , kernelParams[1][2])
        baselines[cols[i]] = params[cols[i]]
    datas = []
    # timesLinspace = np.linspace(0, 23400, int(23400/delta))
    rounder= -1*int(np.round(np.log(delta)/np.log(10)))
    for d in pd.date_range(sDate, eDate):

        inputCsvPath = ric + "_" + d.strftime("%Y-%m-%d") + "_12D.csv"
        if inputCsvPath not in os.listdir(inputDataPath): continue
        print(d)
        logger_time=time.time()
        print(logger_time)
        data = pd.read_csv(inputDataPath + "/" + inputCsvPath)
        data = cleanMOs(data)
        data["Tminus1"] = 0
        data['Tminus1'].iloc[1:] = data['Time'].iloc[:-1] #floor T - Tminus1 to zero
        data = data.sort_values(["Time", "OrderID"])
        data['prevSpread'] = data['Ask Price 1'] - data['Bid Price 1'] + data['BidDiff'] - data['AskDiff']
        bigResultArr = np.zeros((len(data), 26))
        counter = 0
        for r_i in data.iterrows():

            r_i = r_i[1]
            t = r_i.Time
            # print(t)
            mat = np.zeros((num_nodes, num_nodes))
            s = t
            hourIndex = np.min([12,int(np.floor(s/1800))])
            for i in range(num_nodes):
                for j in range(num_nodes):
                    kernelParams = params.get(cols[i] + "->" + cols[j], None)
                    if kernelParams is None: continue
                    todMult = tod[cols[j]][hourIndex]
                    mat[i][j]  = todMult*kernelParams[0]*kernelParams[1][0]/((-1 + kernelParams[1][1])*kernelParams[1][2]) # alpha/(beta -1)*gamma
            specRad = np.max(np.linalg.eig(mat)[0])
            # print("spectral radius = ", specRad)
            specRad = np.max(np.linalg.eig(mat)[0]).real
            if specRad < 1 : specRad = 0.99 # dont change actual specRad if already good
            df = data.loc[data.Time < t]
            hourIdx = np.min([12,int(np.floor(t/1800))])
            bigResultArr[counter, 0] = np.argwhere(np.array(cols) == r_i.event)[0][0]
            bigResultArr[counter, 1] = t
            if len(df) == 0:
                for col in cols:
                    idx = np.argwhere(np.array(cols) == col)[0][0]
                    bigResultArr[counter, 2+2*idx] = baselines[col]
                    bigResultArr[counter, 2+2*idx + 1] = baselines[col]
                counter += 1
                continue
            for col in cols:
                idx = np.argwhere(np.array(cols) == col)[0][0]
                mult = 1
                if "inspread" in col:
                    currSpread = df.iloc[-1].prevSpread
                    mult = (currSpread/avgSpread)**spreadBeta
                intensity = baselines[col]
                for r in df.iterrows():
                    r = r[1]
                    t_j = r.Time
                    if r.event + '->' + col in paramsDict.keys():
                        _params = paramsDict[r.event + '->' + col]
                        intensity += _params[0]*(1 + _params[2]*(t - t_j))**( -1*_params[1])
                intensity = mult*tod[col][hourIdx]*intensity*(0.99/specRad)
                bigResultArr[counter, 2+2*idx] = np.max([0,intensity])
                if r_i.event + '->' + col in paramsDict.keys():
                    _params = paramsDict[r_i.event + '->' + col]
                    intensity += _params[0]
                intensity = mult*tod[col][hourIdx]*intensity*(0.99/specRad)
                bigResultArr[counter, 2+2*idx+1] = np.max([0,intensity])
            counter += 1
            with open(resultsPath + "/"+ric + "_" + d.strftime("%Y-%m-%d") + "_" + d.strftime("%Y-%m-%d") + "_qqTrapezoid", "wb") as f:
                pickle.dump(bigResultArr[:counter+1,:], f)

    return

def runInterArrivalTimes(paths, resultsPath, sDate, eDate, ric):
    cols = ["lo_deep_Ask", "co_deep_Ask", "lo_top_Ask","co_top_Ask", "mo_Ask", "lo_inspread_Ask" ,
            "lo_inspread_Bid" , "mo_Bid", "co_top_Bid", "lo_top_Bid", "co_deep_Bid","lo_deep_Bid" ]
    simPriceChangeTimes = []
    simTimes = []
    for path in paths:
        print(path)
        tryer= 0
        while tryer < 5: # retry on pickle clashes
            try:
                with open(path, "rb") as f:
                    results = pickle.load(f)
                tryer = 6
            except Exception as e:
                print("problem w " + path + " " + str(e))
                time.sleep(1)
                tryer +=1
                if tryer ==6:
                    continue

        simDf = pd.DataFrame(results[1])
        simDf['Ask'] = simDf['Ask_touch'].apply(lambda x: x[0])
        simDf['Bid'] =  simDf['Bid_touch'].apply(lambda x: x[0])
        simDf['Mid'] = 0.5*(simDf['Ask'] + simDf['Bid'])
        mid = simDf.Mid.values
        times = np.append([0], np.array(results[0][1:])[:,1]).astype(float)
        simPriceChangeTimes += [np.log(np.diff(times[np.append([0], np.diff(mid)) !=0].astype(float)))/np.log(10)]
        simTimes += [times[np.append([0], np.diff(mid)) !=0].astype(float)]
    simTimeIds = np.hstack([(i[1:] - 34200)//(1800) for i in simTimes])
    ids, counts = np.unique(simTimeIds, return_counts =True)
    counts = counts/sum(counts)
    dictNorm = {}
    for i, c in zip(ids, counts):
        dictNorm[i] = 1/(13*c)
    wts = [dictNorm[i] for i in simTimeIds]

    empPriceChangeTimes = []
    for d in pd.date_range(sDate,eDate):
        l = dataLoader.Loader(ric, d, d, nlevels = 2, dataPath = "/SAN/fca/Konark_PhD_Experiments/extracted/")
        data = l.load()
        if len(data): data = data[0]
        else: continue
        data['Mid'] = 0.5*(data['Ask Price 1'] + data['Bid Price 1'])
        mid = data.Mid.values
        times = data.Time.values - 34200
        ts = np.diff(times[np.append([0], np.diff(mid)) !=0].astype(float))
        empPriceChangeTimes += [np.log(ts[ts>0])/np.log(10)]
        # print(data.Spread.values)

    fig = plt.figure()
    plt.title(ric + " PriceChangeTime distribution")
    plt.xlabel("Times (log10)")
    plt.ylabel("Frequency")
    plt.hist(np.hstack(empPriceChangeTimes), bins = 100, label = "Empirical", alpha = 0.5, density = True)
    plt.hist(np.hstack(simPriceChangeTimes), bins = 100, label = "Simulated", alpha = 0.5, density = True, weights = wts)
    plt.legend()
    fig.savefig(resultsPath + "/"+ric + "_" + sDate.strftime("%Y-%m-%d") + "_" + eDate.strftime("%Y-%m-%d") + "_priceChangeTimeDistribution.png")

    times = {}
    for d in pd.date_range(sDate, eDate):
        try:
            data = pd.read_csv("/SAN/fca/Konark_PhD_Experiments/extracted/"+ric+"_"+ d.strftime("%Y-%m-%d") +"_12D.csv")
        except:
            continue
        # data = data.loc[(data.Time < 3*3600)&(data.Time > 2.5*3600)]
        data_times = data.groupby("event").Time.apply(lambda x: np.array(x)).to_dict()
        for k in data_times:
            tmp = times.get(k, [])
            times[k]=np.append(tmp, np.diff(data_times[k]) )

    times_Sim = {}
    times_Sim_wts = {}
    for path in paths:
        print(path)
        tryer= 0
        while tryer < 5: # retry on pickle clashes
            try:
                with open(path, "rb") as f:
                    results = pickle.load(f)
                tryer = 6
            except:
                time.sleep(1)
                tryer +=1
        data = pd.DataFrame(results[0][1:], columns = ["event", "Time", "tmp"])
        # data.loc[(data.Time < 3*3600)&(data.Time > 2.5*3600)]
        data_times = data.groupby("event").Time.apply(lambda x: np.array(x)[1:]).to_dict()
        for k in data_times:
            tmp = times_Sim.get(k, [])
            times_Sim[k]=np.append(tmp, np.diff(data_times[k]) )
            tmp = times_Sim_wts.get(k, [])
            times_Sim_wts[k]=np.append(tmp, data_times[k][1:] )
    for c in cols:
        simTimes = times_Sim_wts[c]
        simTimeIds = np.hstack([(i - 34200)//(1800) for i in simTimes])
        ids, counts = np.unique(simTimeIds, return_counts =True)
        counts = counts/sum(counts)
        dictNorm = {}
        for i, j in zip(ids, counts):
            dictNorm[i] = 1/(13*j)
        wts = [dictNorm[i] for i in simTimeIds]
        fig = plt.figure()
        plt.hist(np.log(times[c][times[c]>0])/np.log(10), bins = 100,alpha=.5, density = True, label = "Empirical")
        plt.hist(np.log(times_Sim[c])/np.log(10), bins = 100, density = True, alpha=.5, label = "Simulated", weights = wts)
        plt.legend()
        plt.title(c + " : Inter-arrival time")
        plt.xlabel("$log_{10}(Time)$")
        plt.ylabel("Probability Density")
        fig.savefig(resultsPath + "/"+ric + "_" + sDate.strftime("%Y-%m-%d") + "_" + eDate.strftime("%Y-%m-%d") + "_"+c+"Distribution.png")
    return

def run(ric = "AAPL.OQ", sDate = dt.date(2019,1,2), eDate = dt.date(2019,3,31), suffix = "_CLSLogLin_10", dataPath = "/SAN/fca/Konark_PhD_Experiments/simulated", resultsPath = "/SAN/fca/Konark_PhD_Experiments/results"):
    paths = [dataPath + "/" + i for i in os.listdir(dataPath) if (ric in i)&(suffix in i)&(~("tmp" in i))]
    if "Symm2" in ric: ric = ric.replace("Symm2", "")
    if "Symm" in ric: ric = ric.replace("Symm", "")
    # runQQInterArrival(ric, sDate, eDate, resultsPath)
    runSignaturePlots(paths, resultsPath, ric, sDate, eDate)
    runDistribution(paths, resultsPath , sDate, eDate, ric)
    runACF(paths, resultsPath, sDate, eDate, ric)
    runPricePaths(paths, resultsPath, sDate, eDate, ric)
    runTODCheck(paths, resultsPath, sDate, eDate,ric)
    runInterArrivalTimes(paths, resultsPath, sDate, eDate, ric)
    # runQQInterArrivalTrapezoid(ric, sDate, eDate, resultsPath)

    # TODO: qq for interevent time, priceChangeTime - q reactive HP uses it
    return