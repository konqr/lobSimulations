import numpy as np
import pandas as pd
import statsmodels.api as sm
import pickle
import os
import datetime as dt
from hawkes import dataLoader
from scipy.optimize import curve_fit

class ParametricFit():

    def __init__(self, data):
        self.data = data # list of 2D vectors : time, value of kernels

    def fitPowerLaw(self, norm):
        Xs = sm.add_constant(np.hstack([np.log(d[0]) for d in self.data]))
        Ys = np.hstack([np.log(d[1]) for d in self.data])
        model = sm.OLS(Ys, Xs)
        res = model.fit()
        print(res.summary())
        thetas = res.params
        t0 = np.exp(-1*np.log(norm*(-1*thetas[1] - 1)/np.exp(thetas[0]))/(-1*thetas[1] - 1))
        thetas = np.append(thetas, [t0])
        return thetas, res

    def fitPowerLawCutoff(self, norm):
        def powerLawCutoff(time, beta, gamma, t0 = (10/9)*1e-4, norm = norm):
            alpha = norm*beta*(gamma - 1)*(1+beta*t0)**(gamma - 1)
            funcEval = alpha/((1 + beta*time)**gamma)
            funcEval[time < t0] = 0
            return funcEval
        Xs = np.hstack( [d[0] for d in self.data] )
        Ys = np.hstack([d[1] for d in self.data])
        params, cov = curve_fit(powerLawCutoff, Xs, Ys,bounds=([0, -2], [np.inf, 0]), maxfev = 1e6)
        thetas = params
        return thetas, cov

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

def run(sDate, eDate, suffix  = "_todIS_sgd"):
    cols = ["lo_deep_Ask", "co_deep_Ask", "lo_top_Ask","co_top_Ask", "mo_Ask", "lo_inspread_Ask" ,
            "lo_inspread_Bid" , "mo_Bid", "co_top_Bid", "lo_top_Bid", "co_deep_Bid","lo_deep_Bid" ]
    ric = "AAPL.OQ"


    l = dataLoader.Loader(ric, sDate, eDate, nlevels = 2, dataPath = "D:\\Work\\PhD\\Expt 1\\params\\")
    thetas = {}
    for d in pd.date_range(sDate,eDate):
        if d.strftime("%Y-%m-%d") == "2019-01-09": continue
        if os.path.exists(l.dataPath + ric + "_Params_" + str(d.strftime("%Y-%m-%d")) + "_" + str(d.strftime("%Y-%m-%d")) + "_CLSLogLin_20" + suffix):
            with open(l.dataPath + ric + "_Params_" + str(d.strftime("%Y-%m-%d")) + "_" + str(d.strftime("%Y-%m-%d")) + "_CLSLogLin_20" + suffix , "rb") as f: #"/home/konajain/params/"
                theta = list(pickle.load(f).values())[0]
                if len(theta) == 3:
                    theta1, theta2, theta3 = theta
                    theta = np.hstack([theta1[:,:5], theta2, theta3, theta1[:,5:]])
                if theta.shape[0] == 217:
                    theta = (theta[0,:], theta[1:,:].transpose())
                if theta.shape[0] == 229:
                    theta = (theta[:13,:].transpose(), theta[13:,:].transpose())
                thetas.update({d.strftime("%Y-%m-%d") : theta })

    # each theta in kernel = \delta * h(midpoint)

    # 1. plain
    res = {}
    params = {}
    if len(thetas[sDate.strftime('%Y-%m-%d')][0].shape) == 1:
        for d, theta in thetas.items():
            if d=="2019-01-09": continue
            for i, col in zip(np.arange(12), cols):
                exo = theta[0][i]
                res[col] = res.get(col, []) + [exo]
                phi = theta[1][i,:].reshape((len(theta[1][i,:])//12,12))
                num_datapoints = (phi.shape[0] + 2)//2
                min_lag =  1e-3
                max_lag = 500
                timegridLin = np.linspace(0,min_lag, num_datapoints)
                timegridLog = np.exp(np.linspace(np.log(min_lag), np.log(max_lag), num_datapoints))
                timegrid = np.append(timegridLin[:-1], timegridLog)
                for j in range(len(cols)):
                    col2 = cols[j]
                    points = phi[:,j][1:]
                    timegrid_len = np.diff(timegrid)/2
                    timegrid_mid = timegrid[:-1] + timegrid_len
                    points = points / timegrid_len[1:]

                    res[col2 + '->' + col] =  res.get(col2 + '->' + col, []) + [(t,p) for t,p in zip(timegrid_mid[1:], points)]

                    # points = phi[:,j]
                    # timegrid_len = np.diff(timegrid)/2
                    # timegrid_mid = timegrid[:-1] + timegrid_len
                    # points = points / timegrid_len
                    #
                    # res[col2 + '->' + col] =  res.get(col2 + '->' + col, []) + [(t,p) for t,p in zip(timegrid_mid, points)]
        for k, v in res.items():
            if "->" not in k:
                params[k] = np.mean(v)
            else:
                numDays = len(v)//len(timegrid_len[1:])
                norm = np.sum(np.multiply(np.array(v)[:,1], np.array(list(timegrid_len[1:])*numDays)))/numDays
                side = np.sign(norm)
                if np.abs(norm) > 1: norm = 0.85
                pars, resTemp = ParametricFit(np.abs(v)).fitPowerLaw(norm= np.abs(norm))
                params[k] = (side, pars)
                print(k, params[k])
                # pars = np.average(np.array(v)[:,1].reshape((9,18)), axis=0)
                # params[k] = pars
        with open(l.dataPath + ric + "_ParamsInferredWCutoff_" + str(sDate.strftime("%Y-%m-%d")) + "_" + str(eDate.strftime("%Y-%m-%d")) + "_CLSLogLin_" + str(len(timegridLin)) , "wb") as f: #"/home/konajain/params/"
            pickle.dump(params, f)
        return params, res
    # 3. spread + TOD

    # 2. TOD
    for d, theta in thetas.items():
        if d=="2019-01-09": continue
        for i, col in zip(np.arange(12), cols):
            theta_i = theta[0][i,:]
            exo = theta_i
            res[col] = res.get(col, []) + [exo]
            phi = theta[1][i,:]
            phi = phi.reshape((len(phi)//12,12))
            num_datapoints = (phi.shape[0] + 2)//2
            min_lag =  1e-3
            max_lag = 500
            timegridLin = np.linspace(0,min_lag, num_datapoints)
            timegridLog = np.exp(np.linspace(np.log(min_lag), np.log(max_lag), num_datapoints))
            timegrid = np.append(timegridLin[:-1], timegridLog)
            for j in range(len(cols)):
                col2 = cols[j]
                points = phi[:,j][1:]
                timegrid_len = np.diff(timegrid)/2
                timegrid_mid = timegrid[:-1] + timegrid_len
                points = points / timegrid_len[1:]

                res[col2 + '->' + col] =  res.get(col2 + '->' + col, []) + [(t,p) for t,p in zip(timegrid_mid[1:], points)]
    for k, v in res.items():
        if "->" not in k:
            params[k] = np.array(v).mean(axis=0)
        else:
            numDays = len(v)//len(timegrid_len[1:])
            norm = np.sum(np.multiply(np.array(v)[:,1], np.array(list(timegrid_len[1:])*numDays)))/numDays
            side = np.sign(norm)
            if np.abs(norm) > 1: norm = side*0.85
            pars, resTemp = ParametricFit(np.abs(v)).fitPowerLaw(norm= np.abs(norm))
            params[k] = (side, pars)

    with open(l.dataPath + ric + "_ParamsInferredWCutoff_" + str(sDate.strftime("%Y-%m-%d")) + "_" + str(eDate.strftime("%Y-%m-%d")) + "_CLSLogLin_" +suffix + "_"+ str(len(timegridLin)) , "wb") as f: #"/home/konajain/params/"
        pickle.dump(params, f)
    return params, res

