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

# We plan to make use of inter-event durations' Q-Q plots, signature plots, distribution of spread and returns, average shape of the book,
# autocorrelation of returns and order flow, and sample price paths as our set of stylized facts.
# TOD check - flow, spread

def runQQInterArrival(ric, sDate, eDate, resultsPath, delta = 1e-1, inputDataPath = "/SAN/fca/Konark_PhD_Experiments/extracted", avgSpread = 0.169, spreadBeta =0.7479):
    paramsPath = inputDataPath + "/"+ric+"_ParamsInferredWCutoff_2019-01-02_2019-03-31_CLSLogLin_10"
    todPath = inputDataPath + "/"+ric+"_Params_2019-01-02_2019-03-29_dictTOD"
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
    timesLinspace = np.linspace(0, 23400, int(23400/delta))
    rounder= -1*int(np.round(np.log(delta)/np.log(10)))
    for d in pd.date_range(sDate, eDate):

        inputCsvPath = ric + "_" + d.strftime("%Y-%m-%d") + "_12D.csv"
        if inputCsvPath not in os.listdir(inputDataPath): continue
        print(d)
        logger_time=time.time()
        print(logger_time)
        data = pd.read_csv(inputDataPath + "/" + inputCsvPath)
        data["Tminus1"] = 0
        data['Tminus1'].iloc[1:] = data['Time'].iloc[:-1] #floor T - Tminus1 to zero
        data = data.sort_values(["Time", "OrderID"])
        data['prevSpread'] = data['Ask Price 1'] - data['Bid Price 1'] + data['BidDiff'] - data['AskDiff']
        tracked_intensities = [list(baselines.values())]
        for t in timesLinspace:
            print(t)
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
            print("spectral radius = ", specRad)
            specRad = np.max(np.linalg.eig(mat)[0]).real
            if specRad < 1 : specRad = 0.99 # dont change actual specRad if already good
            df = data.loc[(data.Time < t)&(data.Time >= t-10)]
            hourIdx = np.min([12,int(np.floor(t/1800))])
            if len(df) == 0:
                tracked_intensities = tracked_intensities + [12*[0]]
                continue
            tracked_intensity = []
            for col in cols:
                mult = 1
                if "inspread" in col:
                    currSpread = df.iloc[-1].prevSpread
                    mult = (currSpread/avgSpread)**spreadBeta
                intensity = baselines[col]
                for r in df.iterrows():
                    r = r[1]
                    if r.event + '->' + col in paramsDict.keys():
                        _params = paramsDict[r.event + '->' + col]
                        if np.isnan(_params[2]): continue
                        # print(intensity)
                        print((t - r.Time))
                        # print(_params)
                        intensity += _params[0]/((1 + (t - r.Time)*_params[2])**_params[1])
                print(intensity)
                tracked_intensity = tracked_intensity + [np.max([0,mult*tod[col][hourIdx]*intensity*0.99/specRad])]
            tracked_intensities = tracked_intensities + [tracked_intensity]
        tracked_intensities = np.array(tracked_intensities + [12*[0]])
        print(len(tracked_intensities))
        print(len(timesLinspace))
        print(time.time() - logger_time)
        logger_time = time.time()
        def calc(r):
            idx = int(r.eventID)
            Time = r.Time
            Tminus1 = r.Tminus1
            i_end = np.argmax(timesLinspace > Time) - 1
            i_start = np.argmax(timesLinspace > Tminus1) - 1
            # idx = np.argmax(cols == event)
            if Time - Tminus1 < delta:
                # print(i_start)
                # print(idx)
                # print(tracked_intensities[i_start:i_end, idx])
                return tracked_intensities[i_start, idx]*np.max([0.5*1e-9,(Time-Tminus1)])

            integral = delta*np.sum(tracked_intensities[i_start:i_end, idx]) + (Time - np.round(Time, rounder))*tracked_intensities[i_end, idx]  - (Tminus1 - np.round(Tminus1, rounder))*tracked_intensities[i_start, idx]
            return integral
        data['eventID'] = data.event.apply(lambda x: np.argmax(np.array(cols) == x))
        data["lambdaIntegral"] = data[['eventID','Time', 'Tminus1']].apply(calc, axis=1) #np.apply_along_axis(calc, 1, data[['eventID','Time', 'Tminus1']].values)
        print(time.time() - logger_time)
        data[["Time", "OrderID", "event", "lambdaIntegral"]].to_csv(resultsPath + "/"+ric + "_" + d.strftime("%Y-%m-%d") +"_QQdf.csv")
        # datas.append(data[["Time", "event", "lambdaIntegral"]])
        dictLambdaIntegral = data[["event", "lambdaIntegral"]].groupby("event").apply(np.array).to_dict()
        for k in dictLambdaIntegral.keys():
            fig = sm.qqplot(dictLambdaIntegral[k], stats.expon, line='45', ylabel=  k)
            fig.savefig(resultsPath + "/"+ric + "_" + d.strftime("%Y-%m-%d") + "_qq_" + k + ".png")
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
        with open(path, "rb") as f:
            data = pickle.load(f)
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
    t = .01
    sample_x = np.linspace(0, 23400, int(23400/t))
    for path in paths:
        with open(path, "rb") as f:
            results = pickle.load(f)
        simDf = pd.DataFrame(results[1])
        simDf['Ask'] = simDf['Ask_touch'].apply(lambda x: x[0])
        simDf['Bid'] =  simDf['Bid_touch'].apply(lambda x: x[0])
        simDf['Mid'] = 0.5*(simDf['Ask'] + simDf['Bid'])
        simDf['Spread'] = simDf['Ask'] - simDf['Bid']
        mid = simDf.Mid.values
        times = np.append([0], np.array(results[0][1:])[:,1])

        idxs = np.searchsorted(times, sample_x)[1:-1] - 1
        sample_y = mid[idxs]
        ret =  np.exp(np.diff(np.log(sample_y))) - 1
        simRets.append(ret)
        simSpreads.append(simDf.Spread.values)
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
    plt.hist(np.hstack(empRets), bins = 100, label = "Empirical", alpha = 0.1, density = True)
    plt.hist(np.hstack(simRets), bins = 100, label = "Simulated", alpha = 0.1, density = True)
    plt.yscale("log")
    plt.legend()
    fig.savefig(resultsPath + "/"+ric + "_" + sDate.strftime("%Y-%m-%d") + "_" + eDate.strftime("%Y-%m-%d") + "_returnsDistribution.png")

    fig = plt.figure()
    plt.title(ric + " spreads distribution")
    plt.xlabel("Spread-in-ticks")
    plt.ylabel("Frequency")
    plt.hist(100*np.hstack(empSpreads), bins = np.arange(20), label = "Empirical", alpha = 0.1, density = True)
    plt.hist(100*np.hstack(simSpreads), bins = np.arange(20), label = "Simulated", alpha = 0.1, density = True)
    plt.yscale("log")
    plt.legend()
    fig.savefig(resultsPath + "/"+ric + "_" + sDate.strftime("%Y-%m-%d") + "_" + eDate.strftime("%Y-%m-%d") + "_spreadDistribution.png")

    return

def run(ric = "AAPL.OQ", sDate = dt.date(2019,1,2), eDate = dt.date(2019,3,31), suffix = "_CLSLogLin_10", dataPath = "/SAN/fca/Konark_PhD_Experiments/simulated", resultsPath = "/SAN/fca/Konark_PhD_Experiments/results"):
    paths = [dataPath + "/" + i for i in os.listdir(dataPath) if (ric in i)&(suffix in i)&(~("tmp" in i))]
    # runQQInterArrival(ric, sDate, eDate, resultsPath)
    # runSignaturePlots(paths, resultsPath, ric, sDate, eDate)
    # runDistribution(paths, resultsPath , sDate, eDate, ric)
    # runAverageShapeOfBook(paths, resultsPath)
    # runACF(paths, resultsPath, sDate, eDate, ric)
    # runPricePaths(paths, resultsPath)
    # runTODCheck(paths, resultsPath, param = "orderflow")
    # runTODCheck(paths, resultsPath, param = "spread")
    return
