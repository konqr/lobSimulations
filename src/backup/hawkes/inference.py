import numpy as np
import pandas as pd
import statsmodels.api as sm
import pickle
import datetime as dt
from scipy.optimize import curve_fit
import copy
import matplotlib.pyplot as plt

from data.dataLoader import dataLoader

class ParametricFit():

    def __init__(self):
        pass

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
            return np.array([f/alpha, f*(-1*np.log(1+gamma*time)), f*(-1*beta)*gamma/(1+gamma*time)]).T
        Xs = np.hstack( [d[0] for d in self.data] )
        Ys = np.hstack([d[1] for d in self.data])
        params, cov = curve_fit(powerLawCutoff, Xs, Ys, maxfev = int(1e6), jac = jac, p0 = [1000*norm*0.7, 1.7, 1000], bounds = ([0,0,0], [np.inf, 2, np.inf])) #bounds=([0, 0], [1, 2]),
        print(params[0]/(params[2]*(params[1] - 1)))
        print(norm)
        thetas = params
        return thetas, cov

    def fitPowerLawCutoffNormConstrained(self, norm, alphaInit): # a not the same as norm in calibration
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
        # alphaInit = np.median(Ys.reshape((len(Ys)//17, 17))[:,0])
        # print(Ys.reshape((len(Ys)//17, 17))[:,0])
        params, cov = curve_fit(powerLawCutoff, Xs, Ys, maxfev = int(1e6), jac = jac, p0 = [ 1.7, alphaInit/(norm*0.7)], bounds = ([0,0], [3, np.inf]), method="dogbox") #bounds=([0, 0], [1, 2]),
        # print(norm*(gamma*(beta - 1)/(params[]*(params[0] - 1)))
        # print(norm)
        thetas = np.append([norm*(params[1]*(params[0] - 1))], params)
        return thetas, cov

    def fitPowerLawCutoffIntegralNormConstrained(self, norm): # a not the same as norm in calibration
        def powerLawCutoffIntegral(time, beta, gamma):
            funcEval = norm - (norm/((1 + gamma*time)**(beta-1)))
            # funcEval[time < t0] = 0
            return funcEval
        def jac(time, beta, gamma):
            return np.array([norm*(np.log(1+gamma*time)/((1+ gamma*time)**(beta-1))), norm*(-1+beta)*time/((1+gamma*time)**beta)]).T
        Xs = np.hstack( [d[0] for d in self.data] )
        Ys = np.hstack([d[1] for d in self.data])
        alphaInit = np.median(Ys.reshape((len(Ys)//17, 17))[:,0])
        # print(Ys.reshape((len(Ys)//17, 17))[:,0])
        params, cov = curve_fit(powerLawCutoffIntegral, Xs, Ys, maxfev = int(1e6), jac = jac, p0 = [ 1.7, alphaInit/(norm*0.7)], bounds = ([0,0], [3, np.inf]), method="dogbox") #bounds=([0, 0], [1, 2]),
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

    def run(self, sDate, eDate, ric = "AAPL.OQ" , suffix  = "_IS_scs", avgSpread = 0.0169, spreadBeta = 0.7479, resID = ""):
        cols = ["lo_deep_Ask", "co_deep_Ask", "lo_top_Ask","co_top_Ask", "mo_Ask", "lo_inspread_Ask" ,
                "lo_inspread_Bid" , "mo_Bid", "co_top_Bid", "lo_top_Bid", "co_deep_Bid","lo_deep_Bid" ]

        l = dataLoader.Loader(ric, sDate, eDate, nlevels = 2, dataPath = "/SAN/fca/Konark_PhD_Experiments/extracted/")
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
        k = list(thetas.keys())[0]

        avgLambda = {}
        for d, theta in thetas.items():
            if d=="2019-01-09": continue
            if d in specDates: continue
            data = pd.read_csv(l.dataPath +ric+"_"+ d +"_12D.csv")
            data = data.loc[data.Time < 23400]
            avgLambda_l = (data.groupby(["event"])["Time"].count()/23400).to_dict()
            for k, v in avgLambda_l.items():
                avgLambda[k] = avgLambda.get(k, []) + [v]
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
                        mat[i][j] = mat[i][j] #*((avgSpread)**spreadBeta)
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
                pointsOrig = copy.deepcopy(points)
                # denoising tricks #
                for j in range(numDays):
                    arr = points[j,:,1]
                    # 1 to len -1
                    arrTmp = copy.deepcopy(np.abs(arr))
                    nanidxs = []
                    exit= False
                    while exit is False:
                        exit = True
                        for i in range(1, len(arrTmp) - 1):
                            if (arrTmp[i-1]/arrTmp[i])/(arrTmp[i]/arrTmp[i+1]) > 1e4:
                                nanidxs.append(i)
                                arrTmp[i] = (arrTmp[i-1]+arrTmp[i+1])/2
                                exit = False
                    # edges : 1 and len:
                    if arrTmp[0]/arrTmp[1] < 1e-3:
                        nanidxs.append(0)
                    if arrTmp[-1]/arrTmp[-2] < 1e-3:
                        nanidxs.append(len(arrTmp)-1)
                    # anything near zero to nan
                    points[j,np.where(arrTmp < 1e-10),1]= np.nan
                    # finally
                    points[j,nanidxs,1] = np.nan
                alphaInit = np.abs(np.nanmedian(points[:,0,1]))
                i = 1
                skip =False
                while np.isnan(alphaInit):
                    print("AlphaInit is nan, moving to next time index")
                    alphaInit = np.abs(np.nanmedian(points[:,i,1]))
                    i +=1
                    if i == 17:
                        print("skipping "+ k + " norm " + str(np.average(norms[k])))
                        skip = True
                        break
                v = points.reshape((numDays*17, 2))
                v = v[~np.isnan(v[:,1]),:]
                # denoising complete #
                norm = np.average(norms[k])
                if skip or (np.abs(norm) < 5e-2):
                    continue
                side = np.sign(norm)
                # if np.abs(norm) > 1: norm = 0.99
                print(alphaInit)
                self.data = np.abs(v)
                pars, resTemp = self.fitPowerLawCutoffNormConstrained(norm=np.abs(norm), alphaInit=alphaInit )
                params[k] = (side, pars)

                plt.figure()
                plt.title(k + " (no denoising) Signum:" + str(int(side)))
                for p in pointsOrig:
                    plt.plot(p[:,0], np.abs(p[:,1]), alpha = 0.1, color='b')
                plt.xscale("log")
                plt.yscale("log")
                plt.xlabel("Lags (seconds, log scale)")
                plt.ylabel("Kernel Value (Abs, log scale)")
                plt.savefig("/SAN/fca/Konark_PhD_Experiments/results/"+ ric + "_PlotOrig_ParamsInferredWCutoff_" + str(sDate.strftime("%Y-%m-%d")) + "_" + str(eDate.strftime("%Y-%m-%d")) + "_" + k + ".png")

                plt.figure()
                plt.title(k + " Signum:" + str(int(side)))
                for p in points:
                    plt.plot(p[:,0], np.abs(p[:,1]), alpha = 0.1, color='b')
                alpha = pars[0]
                beta = pars[1]
                gamma = pars[2]
                plt.plot(p[:,0], np.abs(alpha/((1 + gamma*p[:,0])**beta)), color = "r")
                plt.xscale("log")
                plt.yscale("log")
                plt.xlabel("Lags (seconds, log scale)")
                plt.ylabel("Kernel Value (Abs, log scale)")
                plt.savefig("/SAN/fca/Konark_PhD_Experiments/results/"+ ric + "_PlotDenoised_ParamsInferredWCutoff_" + str(sDate.strftime("%Y-%m-%d")) + "_" + str(eDate.strftime("%Y-%m-%d")) + "_" + k + ".png")

                print(k, params[k])
        mat = np.zeros((12,12))
        for i in range(12):
            for j in range(12):
                kernelParams = params.get(cols[i] + "->" + cols[j], None)
                if kernelParams is None: continue
                mat[i][j]  = kernelParams[0]*kernelParams[1][0]/((-1 + kernelParams[1][1])*kernelParams[1][2])
        avgLambdaArr = []
        for c in cols:
            avgLambdaArr.append(np.average(avgLambda[c]))
        print(avgLambdaArr)
        todMult = []
        for c in cols:
            todMult.append(np.mean([1/i for i in list(tod[c].values())]))
        print(todMult)
        todMult = np.diag(todMult)
        print(todMult-mat)
        exos = np.dot(np.eye(len(cols)) - mat.transpose(), np.array(avgLambdaArr).transpose())
        print(exos)
        for i, col in zip(np.arange(12), cols):
            params[col] = exos[i]
        for k in ["lo_top_", "lo_deep_", "co_top_", "co_deep_", "mo_", "lo_inspread_"]:
            params[k+"Ask"] = 0.5*(params[k+"Ask"]+params[k+"Bid"])
            params[k+"Bid"] = params[k+"Ask"]
        with open(l.dataPath + ric + "_ParamsInferredWCutoffEyeMu_" + resID + "_" + str(sDate.strftime("%Y-%m-%d")) + "_" + str(eDate.strftime("%Y-%m-%d")) + "_CLSLogLin_" + str(len(timegridLin)) , "wb") as f: #"/home/konajain/params/"
            pickle.dump(params, f)
        return params, res