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

def runQQInterArrival(ric, sDate, eDate, resultsPath, inputDataPath = "/SAN/fca/Konark_PhD_Experiments/extracted", avgSpread = 0.169, spreadBeta =0.7479):
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
            paramsDict[cols[i] + "->" + cols[j]]  = (kernelParams[0]*np.exp(kernelParams[1][0]), kernelParams[1][1] , kernelParams[1][2])
        baselines[cols[i]] = params[cols[i]]
    datas = []
    for d in pd.date_range(sDate, eDate):

        inputCsvPath = ric + "_" + d.strftime("%Y-%m-%d") + "_12D.csv"
        if inputCsvPath not in os.listdir(inputDataPath): continue
        print(d)
        print(time.time())
        data = pd.read_csv(inputDataPath + "/" + inputCsvPath)
        data["Tminus1"] = 0
        data['Tminus1'].iloc[1:] = data['Time'].iloc[:-1] #floor T - Tminus1 to zero
        data = data.sort_values(["Time", "OrderID"])
        data['prevSpread'] = data['Ask Price 1'] - data['Bid Price 1'] + data['BidDiff'] - data['AskDiff']
        tracked_intensities = [list(baselines.values())]
        for t in np.linspace(0, 23400, int(23400/1e-1)):
            df = data.loc[(data.Time < t)&(data.Time >= t-10)]
            hourIdx = np.min([12,int(np.floor(t/1800))])
            if len(df) == 0: continue
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
                        intensity += _params[0]*(t - r.Time + _params[2])**_params[1]
                tracked_intensity = tracked_intensity + [np.max([0,mult*tod[col][hourIdx]*intensity])]
            tracked_intensities = tracked_intensities + [tracked_intensity]
        tracked_intensities = np.array(tracked_intensities)
        def calc(r):
            event =r.event
            Time = r.Time
            Tminus1 = r.Tminus1
            i_end = np.argmax(np.linspace(0, 23400, int(23400/1e-1)) > Time)
            i_start = np.argmax(np.linspace(0, 23400, int(23400/1e-1)) > Tminus1)
            idx = np.argmax(cols == event)
            return 1e-1*np.sum(tracked_intensities[i_start:i_end, idx])
        data["lambdaIntegral"] = data[['event','Time', 'Tminus1']].apply(calc, axis=1)
        data[["Time", "event", "lambdaIntegral"]].to_csv(resultsPath + "/"+ric + "_" + d.strftime("%Y-%m-%d") +"_QQdf.csv")
        # datas.append(data[["Time", "event", "lambdaIntegral"]])
        dictLambdaIntegral = data[["event", "lambdaIntegral"]].groupby("event").apply(np.array).to_dict()
        for k in dictLambdaIntegral.keys():
            fig = sm.qqplot(dictLambdaIntegral[k], stats.expon, line='45', ylabel=  k)
            fig.savefig(resultsPath + "/"+ric + "_" + d.strftime("%Y-%m-%d") + "_qq_" + k + ".png")
    return

def runSignaturePlots(paths, resultsPath, ric, sDate, eDate, inputDataPath = "/SAN/fca/Konark_PhD_Experiments/extracted"):
    rvs = {}
    for d in pd.date_range(sDate,eDate):

        l = dataLoader.Loader(ric, d, d, nlevels = 2, dataPath = "/SAN/fca/Konark_PhD_Experiments/extracted/")
        data = l.load()
        if len(data): data = data[0]
        else: continue
        data['mid'] = 0.5*(data['Ask Price 1'] + data['Bid Price 1'])
        times = data.Time.values
        mid = data.mid.values
        rv = []
        for t in range(1,51):
            sample_x = np.linspace(0, 23400, int(23400/t))
            idxs = np.searchsorted(times, sample_x)[1:-1]
            print(idxs)
            sample_y = mid[idxs]
            rv.append([t, np.sum(np.square(np.diff(sample_y)))/23400])
        rvs[d.strftime("%Y-%m-%d")] = np.array(rv)
    with open(resultsPath + "/"+ric + "_" + d.strftime("%Y-%m-%d") + "_signatureDict", "wb") as f:
        pickle.dump(rvs, f)
    for k in rvs.keys():
        fig = plt.scatter(rvs[k][:,0], rvs[k][:,1])
        fig.savefig(resultsPath + "/"+ric + "_" + d.strftime("%Y-%m-%d") + "_signatureScatter_" + k + ".png")
    return



def run(ric = "AAPL.OQ", sDate = dt.date(2019,1,2), eDate = dt.date(2019,3,31), suffix = "_CLSLogLin_10", dataPath = "/SAN/fca/Konark_PhD_Experiments/simulated", resultsPath = "/SAN/fca/Konark_PhD_Experiments/results"):
    paths = [i for i in os.listdir(dataPath) if (ric in i)&(suffix in i)]
    # runQQInterArrival(ric, sDate, eDate, resultsPath)
    runSignaturePlots(paths, resultsPath, ric, sDate, eDate)
    # runDistribution(paths, resultsPath, dist = "spread" )
    # runDistribution(paths, resultsPath, dist = "returns")
    # runAverageShapeOfBook(paths, resultsPath)
    # runACF(paths, resultsPath, param = "returns")
    # runACF(paths, resultsPath, param = "orderflow")
    # runPricePaths(paths, resultsPath)
    # runTODCheck(paths, resultsPath, param = "orderflow")
    # runTODCheck(paths, resultsPath, param = "spread")
    return
