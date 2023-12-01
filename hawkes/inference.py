import numpy as np
import pandas as pd
import statsmodels.api as sm
import pickle
import gc
import datetime as dt
from hawkes import dataLoader

class ParametricFit():

    def __init__(self, data):
        self.data = data # list of 2D vectors : time, value of kernels

    def fitPowerLaw(self):
        Xs = sm.add_constant(np.hstack( [np.log(d[0]) for d in self.data] ))
        Ys = np.hstack([np.log(d[1]) for d in self.data])
        model = sm.OLS(Ys, Xs)
        res = model.fit()
        print(res.summary())
        thetas = res.params
        return thetas, res

    def fitExponential(self):
        Xs = sm.add_constant(np.hstack( [d[0] for d in self.data]))
        Ys = np.hstack([np.log(d[1]) for d in self.data])
        model = sm.OLS(Ys, Xs)
        res = model.fit()
        print(res.summary())
        thetas = res.params
        return thetas, res

    def fitBoth(self):
        thetasPowerLaw, resPowerLaw = self.fitPowerLaw()
        thetasExponential, resExponential = self.fitExponential()
        if resPowerLaw.aic > resExponential.aic:
            print("Exponential selected")
            thetas = thetasExponential
        else:
            print("Power Law selected")
            thetas = thetasPowerLaw
        return thetas

def run(suffix  = "_todIS_sgd"):
    cols = ["lo_deep_Ask", "co_deep_Ask", "lo_top_Ask","co_top_Ask", "mo_Ask", "lo_inspread_Ask" ,
            "lo_inspread_Bid" , "mo_Bid", "co_top_Bid", "lo_top_Bid", "co_deep_Bid","lo_deep_Bid" ]
    ric = "AAPL.OQ"
    sDate = dt.date(2019,1,10)
    eDate = dt.date(2019,1,10)

    l = dataLoader.Loader(ric, sDate, eDate, nlevels = 2, dataPath = "D:\\Work\\PhD\\Expt 1\\params\\")
    thetas = {}
    for d in pd.date_range(sDate,eDate):
        with open(l.dataPath + ric + "_Params_" + str(d.strftime("%Y-%m-%d")) + "_" + str(d.strftime("%Y-%m-%d")) + "_CLSLogLin_20" + suffix , "rb") as f: #"/home/konajain/params/"
            thetas.update(pickle.load(f))

    # each theta in kernel = \delta * h(midpoint)

    # 1. plain
    res = {}
    params = {}
    if len(thetas[sDate.strftime('%Y-%m-%d')]) == 12:
        for d, theta in thetas.items():
            for i, col in zip(np.arange(12), cols):
                exo = theta[i][0][0]
                res[col] = res.get(col, []) + [exo]
                phi = theta[i][1][1:].reshape((len(theta[i][1][1:])//12,12))
                num_datapoints = (phi.shape[0] + 2)//2
                min_lag =  1e-3
                max_lag = 500
                timegridLin = np.linspace(0,min_lag, num_datapoints)
                timegridLog = np.exp(np.linspace(np.log(min_lag), np.log(max_lag), num_datapoints))
                timegrid = np.append(timegridLin[:-1], timegridLog)
                for j in range(len(cols)):
                    col2 = cols[j]
                    points = phi[:,j]
                    timegrid_len = np.diff(timegrid)/2
                    timegrid_mid = timegrid[:-1] + timegrid_len
                    points = points / timegrid_len

                    res[col2 + '->' + col] =  res.get(col2 + '->' + col, []) + [(t,p) for t,p in zip(timegrid_mid, points)]
        for k, v in res.items():
            if "->" not in k:
                params[k] = np.mean(v)
            else:
                pars, resTemp = ParametricFit(v).fitPowerLaw()
                params[k] = pars
        return params, res
    # 3. spread + TOD
    elif len(thetas[sDate.strftime('%Y-%m-%d')]) == 3:
        newThetas = {}
        for d, theta in thetas.iteritems():
            theta1, theta2, theta3 = theta
            newThetas[d] = np.vstack([theta[:5,:], theta1, theta2, theta[7:,:]])
        thetas = newThetas
    # 2. TOD
    for d, theta in thetas.iteritems():
        for i, col in zip(np.arange(12), cols):
            theta_i = theta[i,:]
            exo = theta_i[:13]
            res[col] = res.get(col, []) + [exo]
            phi = theta_i[14:].reshape((len(theta_i[14:])//12,12))
            num_datapoints = (phi.shape[0] + 2)//2
            min_lag =  1e-3
            max_lag = 500
            timegridLin = np.linspace(0,min_lag, num_datapoints)
            timegridLog = np.exp(np.linspace(np.log(min_lag), np.log(max_lag), num_datapoints))
            timegrid = np.append(timegridLin[:-1], timegridLog)
            for j in range(len(cols)):
                col2 = cols[j]
                points = phi[:,j]
                timegrid_len = np.diff(timegrid)/2
                timegrid_mid = timegrid[:-1] + timegrid_len
                points = points / timegrid_len

                res[col2 + '->' + col] =  res.get(col2 + '->' + col, []) + [(t,p) for t,p in zip(timegrid_mid, points)]
    for k, v in res.items():
        if "->" not in k:
            params[k] = np.mean(v)
        else:
            pars, resTemp = ParametricFit(v).fitPowerLaw()
            params[k] = pars

    with open(l.dataPath + ric + "_ParamsInferred_" + str(sDate.strftime("%Y-%m-%d")) + "_" + str(eDate.strftime("%Y-%m-%d")) + "_CLSLogLin_" + str(len(timegridLin)) , "wb") as f: #"/home/konajain/params/"
        pickle.dump(params, f)
    return params, res

