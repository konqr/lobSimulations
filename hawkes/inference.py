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

    def fitPowerLaw(self, norm): # integral infinity - how to control norm? - actually if t0 < min(t), then its ok
        # print(self.data)
        # beta*t**(-alpha)*1(t >= t0)
        Xs = np.hstack([d[0] for d in self.data])
        Ys = np.hstack([np.log(d[1]) for d in self.data])
        Xs = np.log(Xs)

        Xs = sm.add_constant(Xs)

        model = sm.OLS(Ys, Xs)
        res = model.fit()
        print(res.summary())
        thetas = res.params
        t0 = np.exp(-1*np.log(norm*(-1*thetas[1] - 1)/np.exp(thetas[0]))/(-1*thetas[1] - 1))
        thetas = np.append(thetas, [t0])
        return thetas, res

    def fitPowerLawCutoff(self, norm): # a not the same as norm in calibration
        def powerLawCutoff(time, alpha, beta, gamma):
            # alpha = a*beta*(gamma - 1)
            funcEval = alpha/((1 + gamma*time)**beta)
            # funcEval[time < t0] = 0
            return funcEval
        def jac(time, alpha, beta, gamma):
            f = powerLawCutoff(time, alpha, beta, gamma)
            return np.array([f/alpha, f*(-1*beta)*gamma/(1+gamma*time), f*(-1*np.log(1+gamma*time))]).T
        Xs = np.hstack( [d[0] for d in self.data] )
        Ys = np.hstack([d[1] for d in self.data])
        params, cov = curve_fit(powerLawCutoff, Xs, Ys, maxfev = int(1e6), jac = jac, p0 = [1000*norm*0.7, 1.7, 1000], bounds = ([0,0,0], [np.inf, 2, np.inf])) #bounds=([0, 0], [1, 2]),
        print(params[0]/(params[2]*(params[1] - 1)))
        print(norm)
        thetas = params
        return thetas, cov

    def fitPowerLawCutoffNormConstrained(self, norm): # a not the same as norm in calibration
        def powerLawCutoff(time, beta, gamma):
            alpha = norm*(gamma*(beta - 1))
            funcEval = alpha/((1 + gamma*time)**beta)
            # funcEval[time < t0] = 0
            return funcEval
        def jac(time, beta, gamma):
            f = powerLawCutoff(time, beta, gamma)
            return np.array([f*(-1*np.log(1+gamma*time)), f*(-1*beta)*gamma/(1+gamma*time)]).T
        Xs = np.hstack( [d[0] for d in self.data] )
        Ys = np.hstack([d[1] for d in self.data])
        alphaInit = np.median(Ys.reshape((len(Ys)//17, 17))[:,0])
        # print(Ys.reshape((len(Ys)//17, 17))[:,0])
        params, cov = curve_fit(powerLawCutoff, Xs, Ys, maxfev = int(1e6), jac = jac, p0 = [ 1.7, alphaInit/(norm*0.7)], bounds = ([0,0], [2, np.inf]), method="dogbox") #bounds=([0, 0], [1, 2]),
        # print(norm*(gamma*(beta - 1)/(params[]*(params[0] - 1)))
        # print(norm)
        thetas = np.append([norm*(params[1]*(params[0] - 1))], params)
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

def run(sDate, eDate, ric = "AAPL.OQ" , suffix  = "_IS_scs", avgSpread = 0.0169, spreadBeta = 0.7479):
    cols = ["lo_deep_Ask", "co_deep_Ask", "lo_top_Ask","co_top_Ask", "mo_Ask", "lo_inspread_Ask" ,
            "lo_inspread_Bid" , "mo_Bid", "co_top_Bid", "lo_top_Bid", "co_deep_Bid","lo_deep_Bid" ]

    l = dataLoader.Loader(ric, sDate, eDate, nlevels = 2, dataPath = "/SAN/fca/Konark_PhD_Experiments/extracted/")
    # thetas = {}
    # for d in pd.date_range(sDate,eDate):
    #     if d.strftime("%Y-%m-%d") == "2019-01-09": continue
    #     if os.path.exists(l.dataPath + ric + "_Params_" + str(d.strftime("%Y-%m-%d")) + "_" + str(d.strftime("%Y-%m-%d")) + "_CLSLogLin_20" + suffix):
    #         with open(l.dataPath + ric + "_Params_" + str(d.strftime("%Y-%m-%d")) + "_" + str(d.strftime("%Y-%m-%d")) + "_CLSLogLin_20" + suffix , "rb") as f: #"/home/konajain/params/"
    #             theta = list(pickle.load(f).values())[0]
    #             if len(theta) == 3:
    #                 theta1, theta2, theta3 = theta
    #                 theta = np.hstack([theta1[:,:5], theta2, theta3, theta1[:,5:]])
    #             if theta.shape[0] == 217:
    #                 theta = (theta[0,:], theta[1:,:].transpose())
    #             if theta.shape[0] == 229:
    #                 theta = (theta[:13,:].transpose(), theta[13:,:].transpose())
    #             thetas.update({d.strftime("%Y-%m-%d") : theta })
    with open(l.dataPath + ric + "_Params_" + str(sDate.strftime("%Y-%m-%d")) + "_" + str(eDate.strftime("%Y-%m-%d")) + suffix , "rb") as f: #"/home/konajain/params/"
        thetas = pickle.load(f)
        for k, v in thetas.items():
            theta1, theta2, theta3 = v
            # thetas[k] = np.hstack([theta1[:,:5], theta2, theta3, theta1[:,5:]])
            thetas[k] = np.hstack([theta2.transpose()[:,:5], theta1.transpose()*(avgSpread)**spreadBeta, theta3.transpose()*(avgSpread)**spreadBeta, theta2.transpose()[:,5:]])
    with open(l.dataPath + ric + "_Params_2019-01-02_2019-03-29_dictTOD", "rb") as f:
        tod = pickle.load(f)
    # each theta in kernel = \delta * h(midpoint)
    # Special Date:
    specDate = {
        "MEXP" : [dt.date(2019,1,18), dt.date(2019,2,15), dt.date(2019,4,18), dt.date(2019,5,17), dt.date(2019,7,19), dt.date(2019,8,16), dt.date(2019,10,18), dt.date(2019,11,15), dt.date(2020,1,17), dt.date(2020,2,21), dt.date(2020,4,17), dt.date(2020,5,15)],
        "FOMC3" : [dt.date(2019,1,30), dt.date(2019,3,20), dt.date(2019,5,1), dt.date(2019,6,19), dt.date(2019,7,31), dt.date(2019,9,18), dt.date(2019,10,30), dt.date(2019,12,11), dt.date(2020,1,29), dt.date(2020,3,18), dt.date(2020,4,29), dt.date(2020,6,10)],
        "MEND" : [dt.date(2019,1,31), dt.date(2019,4,30), dt.date(2019,5,31), dt.date(2019,10,31), dt.date(2020,1,31), dt.date(2020,4, 30)],
        "MSCIQ" : [dt.date(2019,2,28), dt.date(2019,8,27), dt.date(2020,2,28)],
        "QEXP" : [dt.date(2019,3,15), dt.date(2019,6,21), dt.date(2019,9,20), dt.date(2019,12,20), dt.date(2020,3,20), dt.date(2020,6,19) ],
        "QEND" : [dt.date(2019,3,29), dt.date(2019,6,28), dt.date(2019,9,30), dt.date(2019,12,31), dt.date(2020,3,31), dt.date(2020,6,30)],
        "MSCIS" : [dt.date(2019,5,28), dt.date(2019,11,26), dt.date(2020,5,29)],
        "HALF" : [dt.date(2019,7,3), dt.date(2019,11,29), dt.date(2019,12,24)],
        "DISRUPTION" : [dt.date(2020,3,9), dt.date(2020,3,12), dt.date(2020,3,16)],
        "RSL" : [dt.date(2020,6,26)]
    }

    specDates = []
    for ds in specDate.values():
        specDates += [i.strftime("%Y-%m-%d") for i in ds]
    # 1. plain
    res = {}
    params = {}
    norms = {}
    if len(thetas[sDate.strftime('%Y-%m-%d')][0].shape) == 1:
        for d, theta in thetas.items():
            if d=="2019-01-09": continue
            if d in specDates: continue
            data = pd.read_csv(l.dataPath +ric+"_"+ d +"_12D.csv")
            data["Q"] = (data.Time//1800).astype(int)
            avgEventsByTOD = (data.groupby(["event", "Q"])["Time"].count()/1800).to_dict()
            avgEvents = {}
            for k, v in avgEventsByTOD.items():
                avgEvents[k[0]] = avgEvents.get(k[0], []) + [v/tod[k[0]][k[1]]]
            avgEventsArr = []
            for c in cols:
                avgEventsArr.append(np.average(avgEvents[c]))
            avgEventsArr = np.array(avgEventsArr)

            for i, col in zip(np.arange(12), cols):
                exo = theta[0,i]
                phi = theta[1:,i]
                res[col] = res.get(col, []) + [exo]
                phi = phi.reshape((len(phi)//12,12))
                num_datapoints = (phi.shape[0] + 2)//2
                min_lag =  1e-3
                max_lag = 500
                timegridLin = np.linspace(0,min_lag, num_datapoints)
                timegridLog = np.exp(np.linspace(np.log(min_lag), np.log(max_lag), num_datapoints))
                timegrid = np.append(timegridLin[:-1], timegridLog)
                for j in range(len(cols)):
                    col2 = cols[j]
                    points = phi[:,j]

                    timegrid_len = np.diff(timegrid)
                    timegrid_mid = timegrid[:-1] + timegrid_len/2
                    norms[col2 + '->' + col] = norms.get(col2 + '->' + col, []) + [points.sum()]
                    points = points / timegrid_len

                    res[col2 + '->' + col] =  res.get(col2 + '->' + col, []) + [(t,p) for t,p in zip(timegrid_mid[1:], points[1:])]

            mat = np.zeros((len(cols), len(cols)))
            for i in range(len(cols)):
                for j in range(len(cols)):
                    mat[i][j] = norms[cols[j] + "->" + cols[i]][-1]
                    if "inspread" in cols[i]:
                        mat[i][j] = mat[i][j]*((avgSpread)**spreadBeta)
            print("kernel norm ", d, mat)
            exos = np.dot(np.eye(len(cols)) - mat, avgEventsArr.transpose())
            print("exos ", d, exos)
            for i, col in zip(np.arange(12), cols):
                res[col] = res.get(col, []) + [exos[i]]
        for k, v in res.items():
            if "->" not in k:
                print("exo ", k, v)
                params[k] = np.median(v)
            else:
                numDays = len(v)//17
                points = np.array(v).reshape((numDays,17,2))
                # print(points)
                for j in range(17):
                    t = points[:,j,1]
                    med = np.median(t)
                    if np.abs(med) < 1e-18: med = np.mean(t[np.abs(t)>1e-18])
                    t[np.abs(med/t) > 1e6] = med
                    t[np.abs(med/t) < 1e-6] = med
                    points[:,j,1] = t
                v = points.reshape((numDays*17, 2))
                # print(v)
                norm = np.average(norms[k])
                if np.abs(norm) < 1e-3:
                    continue
                side = np.sign(norm)
                # if np.abs(norm) > 1: norm = 0.99
                pars, resTemp = ParametricFit(np.abs(v)).fitPowerLawCutoffNormConstrained(norm= np.abs(norm))
                params[k] = (side, pars)
                print(k, params[k])
                # pars = np.average(np.array(v)[:,1].reshape((9,18)), axis=0)
                # params[k] = pars
        for k in ["lo_top_", "lo_deep_", "co_top_", "co_deep_", "mo_", "lo_inspread_"]:
            params[k+"Ask"] = 0.5*(params[k+"Ask"]+params[k+"Bid"])
            params[k+"Bid"] = params[k+"Ask"]
        with open(l.dataPath + ric + "_ParamsInferredWCutoff_" + str(sDate.strftime("%Y-%m-%d")) + "_" + str(eDate.strftime("%Y-%m-%d")) + "_CLSLogLin_" + str(len(timegridLin)) , "wb") as f: #"/home/konajain/params/"
            pickle.dump(params, f)
        return params, res
    # 3. spread + TOD

    # 2. TOD
    for d, theta in thetas.items():
        if d=="2019-01-09": continue
        for i, col in zip(np.arange(12), cols):
            if suffix == "_IS_scs":
                exo = theta[i,0]
                phi = theta[i,1:]
            else:
                theta_i = theta[0][i,:]
                exo = theta_i
                phi = theta[1][i,:]
            res[col] = res.get(col, []) + [exo]
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
            print(k, norm)
            side = np.sign(norm)
            #if np.abs(norm) > 1: norm = side*0.95
            pars, resTemp = ParametricFit(np.abs(v)).fitPowerLaw(norm= np.abs(norm))
            params[k] = (side, pars)

    with open(l.dataPath + ric + "_ParamsInferredWCutoff_" + str(sDate.strftime("%Y-%m-%d")) + "_" + str(eDate.strftime("%Y-%m-%d")) + "_CLSLogLin_" +suffix + "_"+ str(len(timegridLin)) , "wb") as f: #"/home/konajain/params/"
        pickle.dump(params, f)
    return params, res

