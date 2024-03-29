import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
import statsmodels.api as sm
from tick.hawkes import HawkesConditionalLaw, HawkesExpKern, HawkesEM
import pickle
import gc
from sklearn.linear_model import LinearRegression, ElasticNet, SGDRegressor, Ridge
import time
import cvxpy as cp
import datetime as dt
from scipy.optimize import lsq_linear
import osqp
import scipy as sp
from scipy import sparse
import sys
sys.path.append("/home/konajain/code/nphc2")
from nphc.main import NPHC, starting_point
from scipy.linalg import inv
import sys
# sys.path.append("/home/konajain/code")
# from aslsd.functionals.kernels.basis_kernels. \
#     basis_kernel_exponential import ExponentialKernel
# from aslsd.functionals.kernels.kernel import KernelModel
# from aslsd.models.hawkes.linear.mhp import MHP
# from aslsd.stats.events.process_path import ProcessPath


class ConditionalLeastSquares():
    # Kirchner 2015: An estimation procedure for the Hawkes Process
    def __init__(self, dictBinnedData, p, tau=1, **kwargs):
        self.dictBinnedData = dictBinnedData
        self.dates = list(self.dictBinnedData.keys())
        self.dims = list(self.dictBinnedData[self.dates[0]].keys())
        self.p = p  # 300
        self.tau = tau  # 0.01
        self.T = kwargs.get("T", np.nan)  # 30 min window size by default
        self.n = int(np.floor(self.T / self.tau))
        self.col = kwargs.get("col", "count")  # one of "count" or "size"
        self.data = {}

    def convertData(self):
        dictData = {}
        for d in self.dates:
            for j in self.dims:
                if j == self.dims[0]:
                    df = self.dictBinnedData[d][j][self.col].apply(lambda x: [x])
                else:
                    df = df.add(self.dictBinnedData[d][j][self.col].apply(lambda x: [x]))
            dictData[d] = df
        return dictData

    def getWindowedData(self):
        dictWindowedData = {}
        for d in self.dates:
            df = self.data[d]
            for i in range(0, len(df), self.n):
                dictWindowedData[d + "_" + str(i)] = df.iloc[i:i + self.n]
        return dictWindowedData

    def constructOneDesignMatrix(self, df):
        Z = np.ones([(len(self.dims) * self.p + 1), self.n - self.p])
        for i in range(self.n - self.p):
            Z[:, i] = np.append(np.vstack(np.flip(df.iloc[i:i + self.p].values).flatten()), [1])
        return Z

    def constructDesignMatrices(self):
        # Kirchner uses 30 min windows of tau = 0.01 sec width samples per day and then avgs over all the samples' estimated params to get the final one
        # This usage is for futures data though and comes from a rigorous model selection process using AIC
        # however the gap is "avging over all days" - that does not seem to be the best idea to me
        # TODO : look at alternate ways of utilizing multiday data - rolling windows not disjoint
        # this implementation follows Kirchner for baseline purposes - creates *disjoint* windows of 30 min each
        Zs = {}
        for k, v in self.windowedData.items():
            if len(v) != self.n: continue
            Zs[k] = self.constructOneDesignMatrix(v)
        return Zs

    def constructYs(self):
        Ys = {}
        for k, v in self.windowedData.items():
            if len(v) != self.n: continue
            Ys[k] = np.vstack(v.iloc[self.p:].values).T
        return Ys

    def fitThetas(self):
        thetas = {}
        i = 0
        for k, v in self.windowedData.items():
            if i == 1: break
            if len(v) != self.n: continue
            Z = self.constructOneDesignMatrix(v)
            Y = np.vstack(v.iloc[self.p:].values).T
            theta = Y.dot((Z.T).dot(np.linalg.inv(Z.dot(Z.T))))

            thetas[k] = theta
            i += 1
        return thetas

    def fit_old(self):
        # sanity check
        if self.p > len(self.dictBinnedData[self.dates[0]][self.dims[0]]):
            print("ERROR: p cannot be greater than num samples")
            return 0
        # need to convert binnedData into vectorized form - right now we have {'limit_bid' : binnedData, ... },
        # need - {{x_j}_{j=1...d}_i}_{i=1...n}
        self.data = self.convertData()
        self.windowedData = self.getWindowedData()
        # construct Z := design matrix
        # Zs = self.constructDesignMatrices()
        # Ys = self.constructYs()
        # fit
        theta_cls = self.fitThetas()

        return theta_cls

    def fit(self):
        # sanity check
        if self.p > len(self.dictBinnedData[self.dates[0]][self.dims[0]]):
            print("ERROR: p cannot be greater than num samples")
            return 0
        bigDfs = {}
        for i in self.dates:
            dictPerDate = self.dictBinnedData[i]
            l_df = []
            for j in dictPerDate.keys():
                l_df += [dictPerDate[j].rename(columns={'count': j})[j]]
            bigDf = pd.concat(l_df, axis=1)
            bigDfs[i] = bigDf
        thetas = {}
        for d, df in bigDfs.items():
            if np.isnan(self.T): self.T = len(df)
            model = VAR(df.iloc[0:self.T])
            res = model.fit(self.p)
            thetas[d] = res.params
        return thetas

class ConditionalLaw():

    def __init__(self, data, **kwargs):
        self.data = data
        self.cfg = kwargs
        # df = pd.read_csv("/home/konajain/data/AAPL.OQ_2020-09-14_12D.csv")
        # eventOrder = np.append(df.event.unique()[6:], df.event.unique()[-7:-13:-1])
        # timestamps = [list(df.groupby('event')['Time'].apply(np.array)[eventOrder].values)]

    def fit(self):
        # log : sampling is semi-log. It uses linear sampling on [0, min_lag] with sampling period delta_lag and log
        # sampling on [min_lag, max_lag] using exp(delta_lag) sampling period.

        hawkes_learner = HawkesConditionalLaw(
            claw_method=self.cfg.get("claw_method","log"),
            delta_lag=self.cfg.get("delta_lag",1e-4),
            min_lag=self.cfg.get("min_lag",1e-3),
            max_lag=self.cfg.get("max_lag",500),
            quad_method=self.cfg.get("quad_method","log"),
            n_quad=self.cfg.get("n_quad",200),
            min_support=self.cfg.get("min_support",1e-1),
            max_support=self.cfg.get("max_support",500),
            n_threads=self.cfg.get("n_threads",4)
        )
        hawkes_learner.fit(self.data)
        baseline = hawkes_learner.baseline
        kernels = hawkes_learner.kernels
        kernel_norms = hawkes_learner.kernels_norms
        return (baseline, kernels, kernel_norms)

    def fitEM(self):
        num_datapoints = self.cfg.get("num_datapoints", 10)
        min_lag = self.cfg.get("min_lag", 1e-3)
        max_lag = self.cfg.get("max_lag" , 500)

        timegridLin =np.linspace(0,min_lag, num_datapoints)
        timegridLog = np.exp(np.linspace(np.log(min_lag), np.log(max_lag), num_datapoints))
        timegrid = np.append(timegridLin[:-1], timegridLog)
        em = HawkesEM(kernel_discretization = timegrid, verbose=True, tol=1e-6, max_iter = 100000)
        em.fit(self.data)
        baseline = em.baseline
        kernel_support = em.kernel_support
        kernel = em.kernel
        kernel_norms = em.get_kernel_norms()
        return (baseline, kernel_support, kernel, kernel_norms)

class ConditionalLeastSquaresLogLin():

    def __init__(self, dictBinnedData, **kwargs):
        self.dictBinnedData = dictBinnedData
        self.dates = list(self.dictBinnedData.keys())
        self.cfg = kwargs
        self.cols = ["lo_deep_Ask", "co_deep_Ask", "lo_top_Ask","co_top_Ask", "mo_Ask", "lo_inspread_Ask" ,
                     "lo_inspread_Bid" , "mo_Bid", "co_top_Bid", "lo_top_Bid", "co_deep_Bid","lo_deep_Bid" ]
        # df = pd.read_csv("/home/konajain/data/AAPL.OQ_2020-09-14_12D.csv")
        # eventOrder = np.append(df.event.unique()[6:], df.event.unique()[-7:-13:-1])
        # timestamps = [list(df.groupby('event')['Time'].apply(np.array)[eventOrder].values)]
        # list of list of 12 np arrays

    def transformData(self, timegrid, date, arrs):
        print(date)
        timegrid_new = np.floor(timegrid/(timegrid[1] - timegrid[0])).astype(np.longlong)
        ser = []
        # bins = np.arange(0, np.max([np.max(arr) for arr in arrs]) + 1e-9, (timegrid[1] - timegrid[0]))
        for arr, col in zip(arrs, self.cols):
            print(col)
            arr = arr[::-1]
            assignedBins = np.ceil(arr/(timegrid[1] - timegrid[0])).astype(np.longlong)
            binDf = np.unique(assignedBins, return_counts = True)
            binDf = pd.DataFrame({"bin" : binDf[0], col : binDf[1]})
            binDf = binDf.set_index("bin")
            #binDf = binDf.reset_index()
            ser += [binDf]
        print("done with binning")
        df = pd.concat(ser, axis = 1)
        df = df.fillna(0)
        df = df.sort_index(ascending=False)
        del arrs
        gc.collect()
        res = []
        try:
            with open(self.cfg.get("loader").dataPath + self.cfg.get("loader").ric + "_" + str(date) + "_" + str(date) + "_" + str(len(timegrid)) +  "_inputResNano" , "rb") as f: #"/home/konajain/params/"
                while True:
                    try:
                        res.append(len(pickle.load(f)))
                    except EOFError:
                        break
        except:
            print("no previous data cache found")
        restartIdx = int(np.sum(res))
        res = []

        for i in range(restartIdx + 1, len(df) - 1):

            idx = df.index[i]
            bin_index_new = np.searchsorted(timegrid_new, idx - df.index, side="right")
            last_idx = len(timegrid_new)

            df['binIndexNew'] = bin_index_new
            df_filtered = df[df['binIndexNew'] != last_idx] # remove past > last elt in timegrid

            # unique_bins, bin_counts = np.unique(df_filtered['binIndexNew'], return_counts=True)

            bin_df = np.zeros((len(timegrid_new) - 1, len(self.cols)))
            df_filtered = df_filtered.loc[df_filtered.index[i+1]:]
            # rCurr = df.iloc[i].values[:-1]
            for j, col in enumerate(self.cols):

                bin_df[:, j] = np.bincount(df_filtered['binIndexNew'], weights=df_filtered[col], minlength=len(timegrid_new))[1:]
                # tmp = rCurr.copy()

            lags = bin_df
            res.append([df.loc[idx, self.cols].values, lags])

            if i%5000 == 0 :
                print(i)
                with open(self.cfg.get("loader").dataPath + self.cfg.get("loader").ric + "_" + str(date) + "_" + str(date) + "_" + str(len(timegrid)) + "_inputResNano" , "ab") as f: #"/home/konajain/params/"
                    pickle.dump(res, f)
                res =[]
                gc.collect()
            elif i==len(df)-2:
                with open(self.cfg.get("loader").dataPath + self.cfg.get("loader").ric + "_" + str(date) + "_" + str(date) + "_" + str(len(timegrid)) + "_inputResNano" , "ab") as f: #"/home/konajain/params/"
                    pickle.dump(res, f)
                res =[]
                gc.collect()

        return res

    def runTransformDate(self):
        num_datapoints = self.cfg.get("num_datapoints", 10)
        min_lag = self.cfg.get("min_lag", 1e-3)
        max_lag = self.cfg.get("max_lag" , 500)

        timegridLin =np.linspace(0,min_lag, num_datapoints)
        timegridLog = np.exp(np.linspace(np.log(min_lag), np.log(max_lag), num_datapoints))
        timegrid = np.append(timegridLin[:-1], timegridLog)
        # can either use np.histogram with custom binsize for adaptive grid
        # or
        # bin data by delta_lag*min_lag and then add bins for exponential lags
        def zero_runs(a):
            # Create an array that is 1 where a is 0, and pad each end with an extra 0.
            iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
            absdiff = np.abs(np.diff(iszero))
            # Runs start and end where absdiff is 1.
            ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
            return ranges
        for i in self.dates:
            dictPerDate = self.dictBinnedData[i]
            for j in range(len(dictPerDate)):
                arr = dictPerDate[j]
                zeroRanges = zero_runs(np.diff(arr))
                for x in zeroRanges:
                    arr[x[0]:x[1]+1] += np.linspace(0,1e-9,2+x[1] - x[0])[:-1]
                dictPerDate[j]= arr

            self.transformData(timegrid, i, dictPerDate)
        return

    def fit(self):

        thetas = {}
        for i in self.dates:
            res_d = []
            with open(self.cfg.get("loader").dataPath + self.cfg.get("loader").ric + "_" + i+ "_" + i + "_19_inputRes" , "rb") as f: #"/home/konajain/params/"
                while True:
                    try:
                        r_d = pickle.load(f)
                        if len(r_d[0]) == 2:
                            r_d = sum(r_d, [])
                        res_d.append(r_d)
                        # if len(res_d) >= 5:
                        #     break
                    except EOFError:
                        break
            res_d = sum(res_d, [])
            Ys = [res_d[i] for i in range(0,len(res_d),2)]
            Xs = [res_d[i+1] for i in range(0,len(res_d),2)]
            Xs = [r.flatten() for r in Xs]
            print(len(Xs))

            Ys = np.array(Ys)
            Xs = np.array(Xs)
            # model = sm.OLS(Ys, Xs))
            # res = model.fit()
            # #res = model.fit_regularized(maxiter = 1000) # doesntwork for multidim
            # params = res.params

            # model = ElasticNet(alpha = 1e-6, max_iter=5000).fit(Xs, Ys)
            # params = (model.intercept_, model.coef_)

            if self.cfg.get("solver", "sgd") == "pinv":
                lr = LinearRegression(positive=True).fit(Xs, Ys)
                print(lr.score(Xs, Ys))
                params = (lr.intercept_, lr.coef_)
            elif self.cfg.get("solver", "sgd") == "ridge":
                models = [Ridge( solver="svd", alpha = 10).fit(Xs[~np.isnan(Ys[:,i]), : ], Ys[:,i][~np.isnan(Ys[:,i])]) for i in range(Ys.shape[1])]
                params = [(model.intercept_, model.coef_) for model in models]
                intercepts = [p[0] for p in params]
                coefs = np.vstack([p[1] for p in params])
                params = (intercepts, coefs)
            elif self.cfg.get("solver", "sgd") == "constrained":
                nTimesteps = Xs[0].shape[0]//Ys[0].shape[0]
                nDim = Ys[0].shape[0]
                I = np.eye(nDim)
                constrsX = []
                constrsY = []
                for i in range(nDim): # TODO: this is not perfect - need to add constraints and solve the problem then
                    r = I[:,i]
                    constrsX.append(np.array([0] + nDim*[0] + (nTimesteps-1)*list(r)))
                    constrsY.append(0.999*np.ones(nDim))
                    # Xs.append(np.array(nTimesteps*list(r)))
                    # Ys.append(-1*r)
                constrsX = np.array(constrsX)
                constrsY = np.array(constrsY)
                Xs = sm.add_constant(Xs)
                x = cp.Variable((Xs.shape[1], Ys.shape[1]))
                constraints = [constrsX@x <= constrsY, constrsX@x >= -1*constrsY ]
                objective = cp.Minimize(0.5 * cp.sum_squares(Xs@x-Ys))
                prob = cp.Problem(objective, constraints)
                result = prob.solve(solver=cp.SCS, verbose=True)
                print(result)
                params = x.value
            else:
                models = [SGDRegressor(penalty = 'l2', alpha = 1e-3, fit_intercept=False, max_iter=5000, verbose=11, learning_rate = "invscaling").fit(Xs, Ys[:,i]) for i in range(Ys.shape[1])]
                params = [(model.intercept_, model.coef_) for model in models]

            thetas[i] = params #, paramsUncertainty)
        return thetas

    def fitConditionalTimeOfDay(self):
        print("fitConditionalTimeOfDay")
        thetas = {}
        for i in self.dates:
            res_d = []
            with open(self.cfg.get("loader").dataPath + self.cfg.get("loader").ric + "_" + i+ "_" + i + "_19_inputRes" , "rb") as f: #"/home/konajain/params/"
                while True:
                    try:
                        r_d = pickle.load(f)
                        if len(r_d[0]) == 2:
                            r_d = sum(r_d, [])
                        res_d.append(r_d)
                    except EOFError:
                        break
            res_d = sum(res_d, [])
            Ys = [res_d[i] for i in range(0,len(res_d),2)]
            Xs = np.array([res_d[i+1] for i in range(0,len(res_d),2)])
            print(len(Xs))
            df = pd.read_csv(self.cfg.get("loader").dataPath + self.cfg.get("loader").ric + "_"+ i +"_12D.csv")
            eventOrder = np.append(df.event.unique()[6:], df.event.unique()[-7:-13:-1])
            arrs = list(df.groupby('event')['Time'].apply(np.array)[eventOrder].values)
            num_datapoints = 10
            min_lag =  1e-3
            max_lag = 500
            timegridLin = np.linspace(0,min_lag, num_datapoints)
            timegridLog = np.exp(np.linspace(np.log(min_lag), np.log(max_lag), num_datapoints))
            timegrid = np.append(timegridLin[:-1], timegridLog)
            ser = []
            bins = np.arange(0, np.max([np.max(arr) for arr in arrs]) + 1e-9, (timegrid[1] - timegrid[0]))
            cols = ["lo_deep_Ask", "co_deep_Ask", "lo_top_Ask","co_top_Ask", "mo_Ask", "lo_inspread_Ask" ,
                    "lo_inspread_Bid" , "mo_Bid", "co_top_Bid", "lo_top_Bid", "co_deep_Bid","lo_deep_Bid" ]
            for arr, col in zip(arrs, cols):
                print(col)
                arr = np.max(arr) - arr
                assignedBins = np.searchsorted(bins, arr, side="right")
                binDf = np.unique(assignedBins, return_counts = True)
                binDf = pd.DataFrame({"bin" : binDf[0], col : binDf[1]})
                binDf = binDf.set_index("bin")
                ser += [binDf]

            print("done with binning")
            df = pd.concat(ser, axis = 1)
            df = df.fillna(0)
            df = df.sort_index(ascending=False)
            df = df.iloc[len(df) - Xs.shape[0]:]
            timestamps = np.floor(df.index.values * (timegridLin[1] - timegridLin[0]) / 1800)
            dummies = pd.get_dummies(timestamps).values

            Xs = [r.flatten() for r in Xs]
            Xs = np.array(Xs)
            Xs = np.hstack([dummies, Xs])
            print(Xs.shape)
            Ys = np.array(Ys)
            # model = ElasticNet(alpha = 1e-6, fit_intercept=False, max_iter=5000).fit(Xs, Ys)
            # params = model.coef_
            if self.cfg.get("solver", "sgd") == "pinv":
                lr = LinearRegression(positive=True, fit_intercept=False).fit(Xs, Ys)
                print(lr.score(Xs, Ys))
                params = lr.coef_
            elif self.cfg.get("solver", "sgd") == "ridge":
                lr = Ridge( solver="svd", alpha = 1e-6, fit_intercept=False).fit(Xs, Ys)
                print(lr.score(Xs, Ys))
                params =  lr.coef_
            elif self.cfg.get("solver", "sgd") == "constrained":
                nTimesteps = (Xs[0].shape[0] - 13)//Ys[0].shape[0]
                nDim = Ys[0].shape[0]
                I = np.eye(nDim)
                constrsX = []
                constrsY = []
                for i in range(nDim): # TODO: this is not perfect - need to add constraints and solve the problem then
                    r = I[:,i]
                    constrsX.append(np.array(13*[0] + nDim*[0] + (nTimesteps-1)*list(r)))
                    constrsY.append(0.85*np.ones(nDim))
                    # Xs.append(np.array(nTimesteps*list(r)))
                    # Ys.append(-1*r)
                constrsX = np.array(constrsX)
                constrsY = np.array(constrsY)
                x = cp.Variable((Xs.shape[1], Ys.shape[1]))
                constraints = [constrsX@x <= constrsY - 1e-3, constrsX@x >= -1*constrsY + 1e-3]
                objective = cp.Minimize(0.5 * cp.sum_squares(Xs@x-Ys))
                prob = cp.Problem(objective, constraints)
                result = prob.solve(solver=cp.SCS, verbose=True)
                print(result)
                params = x.value
            else:

                models = [SGDRegressor(penalty = 'l2', alpha = 1e-6, fit_intercept=False, max_iter=5000, verbose=11, learning_rate = "adaptive", eta0 = 1e-6).fit(Xs, Ys[:,i]) for i in range(Ys.shape[1])]
                params = [model.coef_ for model in models]

            thetas[i] = params #, paramsUncertainty)
        return thetas

    def fitConditionalTimeOfDayInSpread(self, spreadBeta = 0.41):

        thetas = {}
        for i in self.dates:
            res_d = []
            with open(self.cfg.get("loader").dataPath + self.cfg.get("loader").ric + "_" + i+ "_" + i + "_19_inputRes" , "rb") as f: #"/home/konajain/params/"
                while True:
                    try:
                        r_d = pickle.load(f)
                        if len(r_d[0]) == 2:
                            r_d = sum(r_d, [])
                        res_d.append(r_d)
                        # if len(res_d) >= 2:
                        #     break
                    except EOFError:
                        break
            res_d = sum(res_d, [])

            Xs = np.array([res_d[i+1] for i in range(0,len(res_d),2)])

            df = pd.read_csv(self.cfg.get("loader").dataPath + self.cfg.get("loader").ric + "_"+ i+"_12D.csv")

            df['spread'] = df['Ask Price 1'] - df['Bid Price 1'] + df['BidDiff'] - df['AskDiff']

            eventOrder = np.append(df.event.unique()[6:], df.event.unique()[-7:-13:-1])
            arrs = list(df.groupby('event')['Time'].apply(np.array)[eventOrder].values)
            spreads = list(df.groupby('event')['spread'].apply(np.array)[eventOrder].values)
            num_datapoints = 10
            min_lag =  1e-3
            max_lag = 500
            timegridLin = np.linspace(0,min_lag, num_datapoints)
            timegridLog = np.exp(np.linspace(np.log(min_lag), np.log(max_lag), num_datapoints))
            timegrid = np.append(timegridLin[:-1], timegridLog)
            ser = []
            bins = np.arange(0, np.max([np.max(arr) for arr in arrs]) + 1e-9, (timegrid[1] - timegrid[0]))
            cols = ["lo_deep_Ask", "co_deep_Ask", "lo_top_Ask","co_top_Ask", "mo_Ask", "lo_inspread_Ask" ,
                    "lo_inspread_Bid" , "mo_Bid", "co_top_Bid", "lo_top_Bid", "co_deep_Bid","lo_deep_Bid" ]
            for arr, sp, col in zip(arrs, spreads, cols):
                print(col)
                arr = np.max(arr) - arr
                sp[sp==0] = 1e-6
                assignedBins = np.searchsorted(bins, arr, side="right")
                binDf = np.unique(assignedBins, return_counts = True)
                avgSp = np.bincount(assignedBins, weights=sp, minlength=len(binDf[1]))
                #print(avgSp.shape)
                avgSp = avgSp[avgSp > 0]
                print(avgSp.shape)
                if avgSp.shape[0] != binDf[1].shape[0]:
                    avgSp = np.append(avgSp, 1e-6+np.zeros((binDf[1].shape[0] -avgSp.shape[0], )))
                print(avgSp.shape)
                avgSp = avgSp / binDf[1]

                binDf = pd.DataFrame({"bin" : binDf[0], col : binDf[1], "spread" : avgSp})
                print(binDf.head())
                binDf = binDf.set_index("bin")
                ser += [binDf]


            # df.Time = np.max(df.Time) - df.Time
            # df['binId'] =pd.cut(df['Time'], bins = bins, labels = False, include_lowest=True)
            # print(df.loc[df.binId.isna()])
            # df['binId'] = df['binId'].astype(int)
            # df = df.groupby('binId')['spread'].mean()
            # binSpread = df
            # print(binSpread.iloc[:100].index)

            print("done with binning")
            df = pd.concat(ser, axis = 1)
            df = df.fillna(0)
            df = df.sort_index(ascending=False)
            df = df.iloc[len(df) - Xs.shape[0]:]
            timestamps = np.floor(df.index.values * (timegridLin[1] - timegridLin[0]) / 1800)
            dummies = pd.get_dummies(timestamps).values

            Xs = np.array([r.flatten() for r in Xs])

            Ys_inspreadBid = [res_d[i][5] for i in range(0,len(res_d),2)]
            dummiesIS = dummies / (100*df['spread'].values.sum(axis = 1)[:,np.newaxis])**spreadBeta
            #XsIS = Xs/(df['spread'].values.sum(axis = 1)[:,np.newaxis])**spreadBeta
            XsIS = np.hstack([dummiesIS, Xs])
            print("done editing dummies")
            # model = ElasticNet(alpha = 1e-6, fit_intercept=False, max_iter=5000).fit(XsBid, Ys_inspreadBid)
            # params2 = model.coef_
            Xs_oth = np.hstack([dummies, Xs])
            print(Xs_oth.shape)
            Ys_oth = [np.append(res_d[i][:5],res_d[i][7:]) for i in range(0,len(res_d),2)]
            Ys_inspreadAsk = [res_d[i][6] for i in range(0,len(res_d),2)]
            Ys_oth = np.array(Ys_oth)
            if self.cfg.get("solver", "sgd") == "pinv":
                lr = LinearRegression(positive=True, fit_intercept=False).fit(XsIS, Ys_inspreadBid)
                print(lr.score(XsIS, Ys_inspreadBid))
                params2 = (lr.intercept_, lr.coef_)

                lr = LinearRegression(positive=True, fit_intercept=False).fit(XsIS, Ys_inspreadAsk)
                print(lr.score(XsIS, Ys_inspreadAsk))
                params3 = (lr.intercept_, lr.coef_)

                lr = LinearRegression(positive=True, fit_intercept=False).fit(Xs_oth, Ys_oth)
                print(lr.score(Xs_oth, Ys_oth))
                params1 = (lr.intercept_, lr.coef_)

            elif self.cfg.get("solver", "sgd") == "ridge":
                lr = Ridge( solver="svd", alpha = 1e-6, fit_intercept=False).fit(XsIS, Ys_inspreadBid)
                print(lr.score(XsIS, Ys_inspreadBid))
                params2 =  lr.coef_

                lr = Ridge( solver="svd", alpha = 1e-6, fit_intercept=False).fit(XsIS, Ys_inspreadAsk)
                print(lr.score(XsIS, Ys_inspreadAsk))
                params3 =  lr.coef_

                lr = Ridge( solver="svd", alpha = 1e-6, fit_intercept=False).fit(Xs_oth, Ys_oth)
                print(lr.score(Xs_oth, Ys_oth))
                params1 =  lr.coef_
            elif self.cfg.get("solver", "sgd") == "constrained":
                params = ()
                for Xs, Ys in zip([XsIS, XsIS, Xs_oth], [np.array(Ys_inspreadBid), np.array(Ys_inspreadAsk), Ys_oth]):
                    if  len(Ys.shape) == 1: nDim = 1
                    else: nDim = Ys[0].shape[0]
                    nTimesteps = 18
                    I = np.eye(12)
                    constrsX = []
                    constrsY = []
                    for i in range(12): # TODO: this is not perfect - need to add constraints and solve the problem then
                        r = I[:,i]
                        constrsX.append(np.array(13*[0] + 12*[0] + (nTimesteps-1)*list(r)))
                        constrsY.append(0.999*np.ones(nDim))
                        # Xs.append(np.array(nTimesteps*list(r)))
                        # Ys.append(-1*r)
                    constrsX = np.array(constrsX)
                    constrsY = np.array(constrsY)
                    x = cp.Variable((Xs.shape[1], nDim))
                    constraints = [constrsX@x <= constrsY, constrsX@x >= -1*constrsY]
                    objective = cp.Minimize(0.5 * cp.sum_squares(Xs@x-Ys.reshape(len(Ys), nDim)))
                    prob = cp.Problem(objective, constraints)
                    result = prob.solve(solver=cp.SCS, verbose=True)
                    print(result)
                    params += (x.value,)
                params2, params3, params1 = params
            else:
                model = SGDRegressor(penalty = 'l2', alpha = 1e-6, fit_intercept=False, max_iter=5000, verbose=11, learning_rate = "adaptive", eta0 = 1e-6).fit(XsIS, Ys_inspreadBid)
                params2 = model.coef_


                model = SGDRegressor(penalty = 'l2', alpha = 1e-6, fit_intercept=False, max_iter=5000, verbose=11, learning_rate = "adaptive", eta0 = 1e-6).fit(XsIS, Ys_inspreadAsk)
                params3 = model.coef_


                models = [SGDRegressor(penalty = 'l2', alpha = 1e-6, fit_intercept=False, max_iter=5000, verbose=11, learning_rate = "adaptive", eta0 = 1e-6).fit(Xs_oth, Ys_oth[:,i]) for i in range(Ys_oth.shape[1])]
                params1 = [model.coef_ for model in models]

            thetas[i] = (params1, params2, params3) #, paramsUncertainty)
        return thetas

    def fitConditionalInSpread(self, spreadBeta = 0.41, avgSpread = 0.028):
        cols = ["lo_deep_Ask", "co_deep_Ask", "lo_top_Ask","co_top_Ask", "mo_Ask", "lo_inspread_Ask" ,
                "lo_inspread_Bid" , "mo_Bid", "co_top_Bid", "lo_top_Bid", "co_deep_Bid","lo_deep_Bid" ]
        with open(self.cfg.get("loader").dataPath + self.cfg.get("loader").ric + "_2019-01-02_2019-03-31_graphDict", "rb") as f:
            boundsDict = pickle.load(f)
        thetas = {}

        # TOD conditioning
        dfs = []
        for i in self.dates:
            try:
                df = pd.read_csv(self.cfg.get("loader").dataPath + self.cfg.get("loader").ric + "_" + i+"_12D.csv")
                dfs.append(df)
            except:
                continue
        df = pd.concat(dfs)
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
            "DISRUPTION" : [dt.date(2019,1,9),dt.date(2020,3,9), dt.date(2020,3,12), dt.date(2020,3,16)],
            "RSL" : [dt.date(2020,6,26)]
        }
        specDates = []
        for ds in specDate.values():
            specDates += [i.strftime("%Y-%m-%d") for i in ds]
        df['halfHourId'] = df['Time'].apply(lambda x: int(np.floor(x/1800)))
        df = df.loc[df.Date.apply(lambda x : x not in specDate.values())]
        dictTOD = {}
        for e in df.event.unique():
            df_e = df.loc[df.event == e].groupby(["Date","halfHourId"])['Time'].count().reset_index().groupby("halfHourId")['Time'].mean()
            df_e = df_e/df_e.mean()
            dictTOD[e] = df_e.to_dict()
        with open(self.cfg.get("loader").dataPath + self.cfg.get("loader").ric + "_Params_" + self.dates[0] + "_" + self.dates[-1] + "_dictTOD" , "wb") as f: #"/home/konajain/params/"
            pickle.dump(dictTOD, f)
        if self.cfg.get("path_dictTOD", 0):
            with open( self.cfg.get("path_dictTOD"), "rb") as f: #"/home/konajain/params/"
                dictTOD = pickle.load(f)
        # XsIS_list = []
        # Xs_oth_list = []
        # Ys_inspreadBid_list = []
        # Ys_inspreadAsk_list = []
        # Ys_oth_list = []
        for i in self.dates:
            date = i
            res_d = []
            with open(self.cfg.get("loader").dataPath + self.cfg.get("loader").ric + "_" + i+ "_" + i + "_19_inputRes" , "rb") as f: #"/home/konajain/params/"
                while True:
                    try:
                        r_d = pickle.load(f)
                        if len(r_d[0]) == 2:
                            r_d = sum(r_d, [])
                        res_d.append(r_d)
                        # if len(res_d) >= 2:
                        #     break
                    except EOFError:
                        break
            res_d = sum(res_d, [])

            Xs = np.array([res_d[i+1] for i in range(0,len(res_d),2)])

            df = pd.read_csv(self.cfg.get("loader").dataPath + self.cfg.get("loader").ric + "_"+ i+"_12D.csv")

            df['spread'] = df['Ask Price 1'] - df['Bid Price 1'] + df['BidDiff'] - df['AskDiff']

            eventOrder = np.append(df.event.unique()[6:], df.event.unique()[-7:-13:-1])
            arrs = list(df.groupby('event')['Time'].apply(np.array)[eventOrder].values)
            spreads = list(df.groupby('event')['spread'].apply(np.array)[eventOrder].values)
            num_datapoints = 10
            min_lag =  1e-3
            max_lag = 500
            timegridLin = np.linspace(0,min_lag, num_datapoints)
            timegridLog = np.exp(np.linspace(np.log(min_lag), np.log(max_lag), num_datapoints))
            timegrid = np.append(timegridLin[:-1], timegridLog)
            timegrid_len = np.diff(timegrid)
            ser = []
            bins = np.arange(0, np.max([np.max(arr) for arr in arrs]) + 1e-9, (timegrid[1] - timegrid[0]))
            cols = ["lo_deep_Ask", "co_deep_Ask", "lo_top_Ask","co_top_Ask", "mo_Ask", "lo_inspread_Ask" ,
                    "lo_inspread_Bid" , "mo_Bid", "co_top_Bid", "lo_top_Bid", "co_deep_Bid","lo_deep_Bid" ]
            for arr, sp, col in zip(arrs, spreads, cols):
                print(col)
                arr = np.max(arr) - arr
                sp[sp==0] = 1e-6
                assignedBins = np.searchsorted(bins, arr, side="right")
                binDf = np.unique(assignedBins, return_counts = True)
                avgSp = np.bincount(assignedBins, weights=sp, minlength=len(binDf[1]))
                #print(avgSp.shape)
                avgSp = avgSp[avgSp > 0]
                print(avgSp.shape)
                if avgSp.shape[0] != binDf[1].shape[0]:
                    avgSp = np.append(avgSp, 1e-6+np.zeros((binDf[1].shape[0] -avgSp.shape[0], )))
                print(avgSp.shape)
                avgSp = avgSp / binDf[1]
                binDf = pd.DataFrame({"bin" : binDf[0], col : binDf[1], "spread" : avgSp})
                print(binDf.head())
                binDf = binDf.set_index("bin")
                ser += [binDf]

            print("done with binning")
            df = pd.concat(ser, axis = 1)
            df = df.fillna(0)
            df = df.sort_index(ascending=False)
            df = df.iloc[len(df) - Xs.shape[0]:]
            timestamps = np.floor(df.index.values * (timegridLin[1] - timegridLin[0]) / 1800)

            Xs = np.array([r.flatten() for r in Xs])
            Xs = sm.add_constant(Xs)

            spr = (df['spread'].apply(lambda x: np.mean(x[x>0]), axis = 1))**spreadBeta
            spr = spr.values.reshape((len(spr),1))
            XsIS = Xs*spr
            Xs_oth = Xs
            print(Xs_oth.shape)
            Ys_inspreadBid = np.array([res_d[i][6] for i in range(0,len(res_d),2)])
            todMultiplier = np.array([dictTOD['lo_inspread_Bid'][x] for x in timestamps])
            Ys_inspreadBid = Ys_inspreadBid/todMultiplier
            Ys_oth = np.array([np.append(res_d[i][:5],res_d[i][7:]) for i in range(0,len(res_d),2)])
            todMultiplier = np.array([[dictTOD[k][x] for k in ['lo_deep_Ask', 'co_deep_Ask', 'lo_top_Ask', 'co_top_Ask', 'mo_Ask', 'mo_Bid', 'co_top_Bid',
                                                               'lo_top_Bid', 'co_deep_Bid', 'lo_deep_Bid']] for x in timestamps])
            Ys_oth = Ys_oth / todMultiplier
            Ys_inspreadAsk = np.array([res_d[i][5] for i in range(0,len(res_d),2)])
            todMultiplier = np.array([dictTOD['lo_inspread_Ask'][x] for x in timestamps])
            Ys_inspreadAsk = Ys_inspreadAsk/todMultiplier
            #Ys_oth = np.array(Ys_oth)
            # XsIS_list.append(XsIS)
            # Xs_oth_list.append(Xs_oth)
            # Ys_oth_list.append(Ys_oth)
            # Ys_inspreadAsk_list.append(Ys_inspreadAsk)
            # Ys_inspreadBid_list.append(Ys_inspreadBid)
            #
            # XsIS = np.vstack(XsIS_list)
            # Xs_oth = np.vstack(Xs_oth_list)
            # Ys_oth = np.vstack(Ys_oth_list)
            # Ys_inspreadBid = np.vstack(Ys_inspreadBid_list)
            # Ys_inspreadAsk = np.vstack(Ys_inspreadAsk_list)
            params = ()
            scaler = (timegrid_len[1:]/timegrid_len[0])
            for Xs, Ys, id in zip([Xs_oth, XsIS, XsIS], [Ys_oth, np.array(Ys_inspreadBid), np.array(Ys_inspreadAsk)], ["oth","inspreadBid", "inspreadAsk"]):
                if len(Ys.shape) == 1: nDim = 1
                else: nDim = Ys[0].shape[0]
                nTimesteps = 18
                I = np.eye(12)
                constrsX = []
                constrsY = []
                for i in range(12): # i
                    r = I[:,i]
                    # INTEGRAL FIT CODE BELOW - stability issues, above approx is much better
                    # arr = []
                    # for s in scaler:
                    #     arr += list(s*r)
                    # constrsX.append(np.array([0] + 12*[0] + arr))
                    #
                    constrsX.append(np.array([0] + 12*[0] + (nTimesteps-1)*list(r)))
                    constrsY.append(0.999*np.ones(nDim))
                    # Xs.append(np.array(nTimesteps*list(r)))
                    # Ys.append(-1*r)
                boundsX, boundsY_u, boundsY_l = [] ,[],[]
                for i in range(12): # i
                    r = I[:,i]

                    if id == "inspreadBid":
                        boundsX.append(np.array([0] + 12*[0] + (nTimesteps-1)*list(r)))
                        ub = np.max([0,boundsDict[cols[i] +"->" + "lo_inspread_Bid"]])
                        lb = np.min([0,boundsDict[cols[i] +"->" + "lo_inspread_Bid"]])
                        boundsY_u.append(np.append([1/12], ub*np.array(12*[1] + (nTimesteps-1)*list(r))))
                        boundsY_l.append(np.append([0], lb*np.array( 12*[0] + (nTimesteps-1)*list(r))))
                    if id == "inspreadAsk":
                        boundsX.append(np.array([0] + 12*[0] + (nTimesteps-1)*list(r)))
                        ub = np.max([0,boundsDict[cols[i] +"->" + "lo_inspread_Ask"]])
                        lb = np.min([0,boundsDict[cols[i] +"->" + "lo_inspread_Ask"]])
                        boundsY_u.append(np.append([1/12], ub*np.array(12*[1] + (nTimesteps-1)*list(r))))
                        boundsY_l.append(np.append([0], lb*np.array( 12*[0] + (nTimesteps-1)*list(r))))
                    if id == "oth":
                        boundsX.append(np.array([0] + 12*[0] + (nTimesteps-1)*list(r)))
                        ubL, lbL = np.ones((nDim, Xs.shape[1])), np.ones((nDim, Xs.shape[1]))
                        for j,col in zip(range(10),cols[:5]+cols[7:]):

                            ub = np.max([0,boundsDict[cols[i] +"->" + col]])
                            lb = np.min([0,boundsDict[cols[i] +"->" + col]])
                            ubL[j] = np.append([1/12], ub*np.array(12*[1] + (nTimesteps-1)*list(r)))
                            lbL[j] = np.append([0], lb*np.array( 12*[0] + (nTimesteps-1)*list(r)))
                        # print(ubL, lbL)
                        boundsY_u.append(ubL)
                        boundsY_l.append(lbL)
                constrsX = np.array(constrsX)
                constrsY = np.array(constrsY)
                boundsX = np.array(boundsX)
                # print(boundsY_u.shape)
                boundsY_u = np.array(boundsY_u).sum(axis=0).transpose().reshape((Xs.shape[1], nDim))
                print(boundsY_u)
                boundsY_l = np.array(boundsY_l).sum(axis=0).transpose().reshape((Xs.shape[1], nDim))
                print(boundsY_l)
                if self.cfg.get("solver", "sgd") == "scs":
                    p = []
                    for i in range(nDim):
                        x = cp.Variable((Xs.shape[1], 1))
                        constraints = [constrsX@x <= constrsY[:,i].reshape(constrsY.shape[0], 1), constrsX@x >= -1*constrsY[:,i].reshape(constrsY.shape[0], 1), x >= boundsY_l[:,i].reshape(boundsY_l.shape[0], 1), x <= boundsY_u[:,i].reshape(boundsY_u.shape[0], 1)]
                        objective = cp.Minimize(0.5 * cp.sum_squares(Xs@x-Ys.reshape(len(Ys), nDim)[:,i].reshape(len(Ys), 1)))
                        prob = cp.Problem(objective, constraints)
                        result = prob.solve(solver=cp.SCS, verbose=True, time_limit_secs=7200) #, x = thetas_old0
                        print(result)
                        p += [x.value]
                    params += (np.vstack(p),)
                elif self.cfg.get("solver", "sgd") == "scipy":
                    p = []
                    if nDim == 1:
                        Ys = Ys.reshape(len(Ys), 1)
                    for i in range(nDim):
                        A = np.vstack([Xs, constrsX])
                        B = np.hstack([Ys[:,i], constrsY[:,i]])
                        p += [lsq_linear(A, B, bounds=(boundsY_l[:,i], boundsY_u[:,i]), lsmr_tol='auto', verbose=2, max_iter = 10000).x]
                    params += (np.vstack(p),)
                elif self.cfg.get("solver", "sgd") == "osqp":
                    p = []
                    for i in range(nDim):
                        mult =1.
                        if nDim == 1:
                            mult = 1./avgSpread**spreadBeta
                            if id == "inspreadBid":
                                mult = mult/max(list(dictTOD["lo_inspread_Bid"].values()))
                                idxY = 6
                            if id == "inspreadAsk":
                                mult = mult/max(list(dictTOD["lo_inspread_Ask"].values()))
                                idxY = 5
                        else:
                            colsOth = cols[:5] + cols[7:]
                            col = colsOth[i]
                            if "Ask" in col: idxY = i
                            else: idxY = i + 2
                            mult = mult/max(list(dictTOD[col].values()))
                        R = sparse.csc_matrix(np.dot(Xs.transpose(), Xs))
                        # print(R)
                        q = -1*np.dot(Xs.transpose(), Ys.reshape(len(Ys), nDim)[:,i].reshape(len(Ys), 1))
                        # print(q)
                        G = sparse.csc_matrix(np.vstack([constrsX, np.eye(Xs.shape[1])]))

                        constrsY_alt = constrsY[:,i].reshape(constrsY.shape[0], 1)
                        l = np.vstack([-1*constrsY_alt, boundsY_l[:,i].reshape(boundsY_l.shape[0], 1)])*mult
                        u = np.vstack([constrsY_alt, boundsY_u[:,i].reshape(boundsY_u.shape[0], 1)])*mult
                        prob = osqp.OSQP()
                        prob.setup(R, q, G, l, u, eps_abs = 1e-4, eps_rel = 1e-4, eps_prim_inf=1e-6, eps_dual_inf=1e-6, polish=True, polish_refine_iter = 100, max_iter = 1000000)
                        res = prob.solve()
                        k  =1
                        while res.info.status_val in [3, 4]:
                            print("retrying with bigger tolerance")
                            mult = 10**k
                            prob = osqp.OSQP()
                            prob.setup(R, q, G, l, u, eps_abs = 1e-4, eps_rel = 1e-4, eps_prim_inf=mult*1e-6, eps_dual_inf=mult*1e-6, polish=True, polish_refine_iter = 100, max_iter = 1000000)
                            res = prob.solve()
                            k+=1
                        p += [res.x]
                    params += (np.vstack(p),)
            params2, params3, params1 = params
            thetas[date] = (params1, params2, params3) #, paramsUncertainty)
            with open(self.cfg.get("loader").dataPath + self.cfg.get("loader").ric + "_Params_" + date + "_" + date + "_IS_"+self.cfg.get("solver", "sgd")+"Sparse_bounds" , "wb") as f: #"/home/konajain/params/"
                pickle.dump(thetas[date], f)
        return thetas

    def fitTODParams(self):
        # TOD conditioning
        dfs = []
        for i in self.dates:
            try:
                df = pd.read_csv(self.cfg.get("loader").dataPath + self.cfg.get("loader").ric + "_" + i+"_12D.csv")
                dfs.append(df)
            except:
                continue
        df = pd.concat(dfs)
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
            "DISRUPTION" : [dt.date(2019,1,9),dt.date(2020,3,9), dt.date(2020,3,12), dt.date(2020,3,16)],
            "RSL" : [dt.date(2020,6,26)]
        }
        specDates = []
        for ds in specDate.values():
            specDates += [i.strftime("%Y-%m-%d") for i in ds]
        df['halfHourId'] = df['Time'].apply(lambda x: int(np.floor(x/1800)))
        df = df.loc[df.Date.apply(lambda x : x not in specDate.values())]
        dictTOD = {}
        for e in df.event.unique():
            df_e = df.loc[df.event == e].groupby(["Date","halfHourId"])['Time'].count().reset_index().groupby("halfHourId")['Time'].mean()
            df_e = df_e/df_e.mean()
            dictTOD[e] = df_e.to_dict()
        with open(self.cfg.get("loader").dataPath + self.cfg.get("loader").ric + "_Params_" + self.dates[0] + "_" + self.dates[-1] + "_dictTOD" , "wb") as f: #"/home/konajain/params/"
            pickle.dump(dictTOD, f)
        return

    def fitHawkesGraph(self):
        big_data = []
        cols = ["lo_deep_Ask", "co_deep_Ask", "lo_top_Ask","co_top_Ask", "mo_Ask", "lo_inspread_Ask" ,
                "lo_inspread_Bid" , "mo_Bid", "co_top_Bid", "lo_top_Bid", "co_deep_Bid","lo_deep_Bid" ]
        for i in self.dates:

            try:
                df = pd.read_csv(self.cfg.get("loader").dataPath + self.cfg.get("loader").ric + "_" + i+"_12D.csv")
            except:
                continue
            timestamps = list(df.groupby('event')['Time'].apply(np.array)[cols].values)
            big_data.append(timestamps)

        nphc = NPHC()
        nphc.fit(big_data,half_width=1.,filtr="rectangular",method="parallel_by_day")
        cumulants_list = [nphc.L, nphc.C, nphc.K_c]
        start_point = starting_point(cumulants_list, random=True)
        R_pred = nphc.solve(training_epochs=50000,display_step=500,learning_rate=1e-2,optimizer='adam')
        d = len(nphc.L[0])
        G_pred = np.eye(d) - inv(R_pred)

        boundsDict = {}
        for i in range(12):
            for j in range(12):
                boundsDict[cols[i]+"->"+cols[j]] = np.sign(G_pred[i][j])
        with open(self.cfg.get("loader").dataPath + self.cfg.get("loader").ric + "_Params_" + self.dates[0] + "_" + self.dates[-1] + "_graphDict", "wb") as f:
            pickle.dump(boundsDict, f)
        return

    def fitOrderSizeDistris(self):
        sizes = {}
        queues = {}
        for i in self.dates:

            try:
                data = pd.read_csv(self.cfg.get("loader").dataPath + self.cfg.get("loader").ric + "_" + i+"_12D.csv")
            except:
                continue
            df = data.groupby("event").Size.apply(list).to_dict()
            for k, v in df.items():
                sizes[k] = sizes.get(k, []) + v

            queues["Ask Size 1"] = np.append(queues.get("Ask Size 1", []) , data["Ask Size 1"].values)
            queues["Ask Size 2"] = np.append(queues.get("Ask Size 2", []) , data["Ask Size 2"].values)
            queues["Bid Size 1"] = np.append(queues.get("Bid Size 1", []) , data["Bid Size 1"].values)
            queues["Bid Size 2"] = np.append(queues.get("Bid Size 2", []) , data["Bid Size 2"].values)
        params = {}
        roundNums = [1,10,50,100,200,500]
        hists = {}
        for k, v in sizes.items():
            n, bins, _ = plt.hist(v, bins = "sqrt", density = True)
            hists[k] = (n, bins)
        for k,v in hists.items():
            print(k)
            edges = v[1]

            params_rN = {}
            for r in roundNums:
                idx = np.searchsorted(edges, r)
                params_rN[r] = v[0][idx]
            params[k] = params_rN
        for k, v in sizes.items():
            v = np.array(v)
            d = params[k]
            i = [x not in roundNums for x in v]
            d[0] = 1./np.average(v[i])
            params[k] = d
        paramsFinal = {}
        for c in ["lo_top", "lo_deep", "lo_inspread", "mo"]:
            b = params[c+"_Bid"]
            a = params[c+"_Ask"]
            paramsFinal[c+"_Bid"] = [(a[0]+b[0])/2, [(i, (a[i]+b[i])/2) for i in roundNums]]
        with open(self.cfg.get("loader").dataPath + self.cfg.get("loader").ric + "_Params_" + self.dates[0] + "_" + self.dates[-1] + "_orderSizesDict", "wb") as f:
            pickle.dump(paramsFinal, f)

        hists = {}
        for k, v in queues.items():
            n, bins, _ = plt.hist(v, bins = "sqrt", density = True)
            hists[k] = (n, bins)
        params = {}
        roundNums = [1,10,50,100,200,500,1000]
        for k,v in hists.items():
            print(k)
            edges = v[1]

            params_rN = {}
            for r in roundNums:
                idx = np.searchsorted(edges, r)
                params_rN[r] = v[0][idx]
            params[k] = params_rN
        for k, v in queues.items():
            v = np.array(v)
            d = params[k]
            i = [x not in roundNums for x in v]
            d[0] = 1./np.average(v[i])
            params[k] = d
        paramsFinal = {}
        for c in [" Size 1", " Size 2"]:
            b = params["Bid"+c]
            a = params["Ask" + c]
            id = "deep"
            if c == " Size 1": id = "touch"
            paramsFinal["Ask_" + id] = [(a[0]+b[0])/2, [(i, (a[i]+b[i])/2) for i in roundNums]]
        with open(self.cfg.get("loader").dataPath + self.cfg.get("loader").ric + "_Params_" + self.dates[0] + "_" + self.dates[-1] + "_queueSizesDict", "wb") as f:
            pickle.dump(paramsFinal, f)

        return

class ASLSD():
    def __init__(self, dates, **kwargs):
        self.dates = dates
        self.cfg = kwargs

    def fit(self):

        # LL = sum_i=1...d (sum_{t_i} ln(lambda_i (t)) - \int_0^T  lambda_i (s) ds)


        cols = ["lo_deep_Ask", "co_deep_Ask", "lo_top_Ask","co_top_Ask", "mo_Ask", "lo_inspread_Ask" ,
                "lo_inspread_Bid" , "mo_Bid", "co_top_Bid", "lo_top_Bid", "co_deep_Bid","lo_deep_Bid" ]

        thetas = {}

        # TOD conditioning
        dfs = []
        for i in self.dates:
            try:
                df = pd.read_csv(self.cfg.get("loader").dataPath + self.cfg.get("loader").ric + "_" + i+"_12D.csv")
                dfs.append(df)
            except:
                continue
        df = pd.concat(dfs)
        dictTimes = df.groupby("event").Time.apply(np.array).to_dict()
        list_times = [dictTimes[c] for c in cols]
        list_times = ProcessPath(list_times, 23400)
        #Define a model
        dims = len(cols)
        kernel_matrix = [[KernelModel([ExponentialKernel(), ExponentialKernel()]) for j in range(dims)]
                         for i in range(dims)]
        mhp = MHP(kernel_matrix)
        kwargs = {'is_log_param': True, 'is_log_grad': True}
        mhp.fit(list_times, 23400, n_iter=10000, verbose=True, **kwargs)
        fit_log = mhp.fit_log
        print(fit_log)
        fig = mhp.plot_solver_path( dpi=None, figsize=(150,150), save = True,  filename = "/SAN/fca/Konark_PhD_Experiments/results/aslsd_fit_path_2exp_"+self.dates[0] + "_" + self.dates[-1]+".png")
        fig = mhp.plot_kernels(dpi=None, figsize=(100,100), save = True,  filename = "/SAN/fca/Konark_PhD_Experiments/results/aslsd_fit_kernels_2exp_"+self.dates[0] + "_" + self.dates[-1]+".png")
        mhp.save("/SAN/fca/Konark_PhD_Experiments/extracted/aslsd_params_fit_2exp_"+self.dates[0] + "_" + self.dates[-1])
        return


