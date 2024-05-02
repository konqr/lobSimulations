import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
import statsmodels.api as sm
from tick.hawkes import HawkesConditionalLaw, HawkesExpKern, HawkesEM,  HawkesSumExpKern
import pickle
import gc
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge
import cvxpy as cp
import datetime as dt
from scipy.optimize import lsq_linear
import osqp
from scipy import sparse
# import sys
# sys.path.append("/home/konajain/code/nphc2")
# from nphc.main import NPHC, starting_point
from scipy.linalg import inv
# sys.path.append("/home/konajain/code")
# from aslsd.functionals.kernels.basis_kernels. \
#     basis_kernel_exponential import ExponentialKernel
# from aslsd.functionals.kernels.kernel import KernelModel
# from aslsd.models.hawkes.linear.mhp import MHP
# from aslsd.stats.events.process_path import ProcessPath
import copy

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
            with open(self.cfg.get("loader").dataPath + self.cfg.get("loader").ric + "_" + str(date) + "_" + str(date) + "_" + str(len(timegrid)) +  "_inputRes" , "rb") as f: #"/home/konajain/params/"
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
                with open(self.cfg.get("loader").dataPath + self.cfg.get("loader").ric + "_" + str(date) + "_" + str(date) + "_" + str(len(timegrid)) + "_inputRes" , "ab") as f: #"/home/konajain/params/"
                    pickle.dump(res, f)
                res =[]
                gc.collect()
            elif i==len(df)-2:
                with open(self.cfg.get("loader").dataPath + self.cfg.get("loader").ric + "_" + str(date) + "_" + str(date) + "_" + str(len(timegrid)) + "_inputRes" , "ab") as f: #"/home/konajain/params/"
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

    def fitConditionalInSpread(self, spreadBeta = 0.41, avgSpread = 0.028, suffix=""):
        cols = ["lo_deep_Ask", "co_deep_Ask", "lo_top_Ask","co_top_Ask", "mo_Ask", "lo_inspread_Ask" ,
                "lo_inspread_Bid" , "mo_Bid", "co_top_Bid", "lo_top_Bid", "co_deep_Bid","lo_deep_Bid" ]
        if self.cfg.get("path_dictGraph", 0):
            with open( self.cfg.get("path_dictGraph"), "rb") as f: #"/home/konajain/params/"
                boundsDict = pickle.load(f)
        else:
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

            spr = df['spread'].apply(lambda x: np.mean(x[x>0]), axis = 1)**spreadBeta
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
                        boundsY_u.append(np.append([1.], ub*np.array(12*[1] + (nTimesteps-1)*list(r))))
                        boundsY_l.append(np.append([0], lb*np.array( 12*[0] + (nTimesteps-1)*list(r))))
                    if id == "inspreadAsk":
                        boundsX.append(np.array([0] + 12*[0] + (nTimesteps-1)*list(r)))
                        ub = np.max([0,boundsDict[cols[i] +"->" + "lo_inspread_Ask"]])
                        lb = np.min([0,boundsDict[cols[i] +"->" + "lo_inspread_Ask"]])
                        boundsY_u.append(np.append([1.], ub*np.array(12*[1] + (nTimesteps-1)*list(r))))
                        boundsY_l.append(np.append([0], lb*np.array( 12*[0] + (nTimesteps-1)*list(r))))
                    if id == "oth":
                        boundsX.append(np.array([0] + 12*[0] + (nTimesteps-1)*list(r)))
                        ubL, lbL = np.ones((nDim, Xs.shape[1])), np.ones((nDim, Xs.shape[1]))
                        for j,col in zip(range(10),cols[:5]+cols[7:]):

                            ub = np.max([0,boundsDict[cols[i] +"->" + col]])
                            lb = np.min([0,boundsDict[cols[i] +"->" + col]])
                            ubL[j] = np.append([1.], ub*np.array(12*[1] + (nTimesteps-1)*list(r)))
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
                        prob.setup(R, q, G, l, u, eps_abs = 1e-6, eps_rel = 1e-6, eps_prim_inf=1e-7, eps_dual_inf=1e-7, polish=True, polish_refine_iter = 100, max_iter = 1000000)
                        res = prob.solve()
                        k  =1
                        while res.info.status_val in [3, 4]:
                            print("retrying with bigger tolerance")
                            mult = 10**k
                            prob = osqp.OSQP()
                            prob.setup(R, q, G, l, u, eps_abs = 1e-6, eps_rel = 1e-6, eps_prim_inf=mult*1e-7, eps_dual_inf=mult*1e-7, polish=True, polish_refine_iter = 100, max_iter = 1000000)
                            res = prob.solve()
                            k+=1
                        p += [res.x]
                    params += (np.vstack(p),)
            params2, params3, params1 = params
            thetas[date] = (params1, params2, params3) #, paramsUncertainty)
            with open(self.cfg.get("loader").dataPath + self.cfg.get("loader").ric + "_Params_" + date + "_" + date + "_IS_"+self.cfg.get("solver", "sgd")+"Sparse_bounds" + suffix , "wb") as f: #"/home/konajain/params/"
                pickle.dump(thetas[date], f)
        return thetas

    def fitTODParams(self):

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
        # TOD conditioning
        dfs = []
        for i in self.dates:
            if i not in specDates:
                try:
                    df = pd.read_csv(self.cfg.get("loader").dataPath + self.cfg.get("loader").ric + "_" + i+"_12D.csv")
                    dfs.append(df)
                except:
                    continue
        df = pd.concat(dfs)
        df['halfHourId'] = df['Time'].apply(lambda x: int(np.floor(x/1800)))
        # df = df.loc[df.Date.apply(lambda x : x not in specDate.values())]
        dictTOD = {}
        for e in df.event.unique():
            df_e = df.loc[df.event == e].groupby(["Date","halfHourId"])['Time'].count().reset_index().groupby("halfHourId")['Time'].mean()
            df_e = df_e/df_e.mean()
            dictTOD[e] = df_e.to_dict()
        for c in ["lo_deep_", "co_deep_", "lo_top_","co_top_", "mo_", "lo_inspread_" ]:
            c1 = c + "Ask"
            c2 = c+"Bid"
            for k in dictTOD[c1].keys():
                dictTOD[c1][k] = (dictTOD[c1][k]+dictTOD[c2][k])*0.5
                dictTOD[c2][k] = dictTOD[c1][k]
            for k,v in dictTOD[c1].items():
                dictTOD[c1][k] = v/np.average(list(dictTOD[c1].values()))
                dictTOD[c2][k] = v/np.average(list(dictTOD[c1].values()))
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

        resultsPath = (self.cfg.get("loader").dataPath).replace("extracted", "results")
        sizes = {}
        queues = {}
        for i in self.dates:

            try:
                data = pd.read_csv(self.cfg.get("loader").dataPath + self.cfg.get("loader").ric + "_" + i+"_12D.csv")
            except:
                continue
            df = data.groupby("event").Size.apply(list).to_dict()
            for k, v in df.items():
                if "Ask" in k: k = k.replace("Ask", "Bid")
                sizes[k] = sizes.get(k, []) + v

            queues["Ask_touch"] = np.append(queues.get("Ask_touch", []) , data["Ask Size 1"].values)
            queues["Ask_deep"] = np.append(queues.get("Ask_deep", []) , data["Ask Size 2"].values)
            queues["Ask_touch"] = np.append(queues.get("Ask_touch", []) , data["Bid Size 1"].values)
            queues["Ask_deep"] = np.append(queues.get("Ask_deep", []) , data["Bid Size 2"].values)
        params = {}

        #,200,500]
        for c in sizes.keys():
            roundNums = [1,10,50,100]
            if "mo" in c: roundNums += [200]
            diracs = {}
            size, freq = np.unique(sizes[c], return_counts = True)
            size_freq = np.vstack([size, freq])
            size_freq_copy = copy.deepcopy(size_freq)
            for num in roundNums:
                idx = np.where(size_freq[0,:] == num)[0][0]
                diracs[num] = size_freq_copy[1,idx]/np.sum(freq)
                size_freq_copy[1,idx] = size_freq_copy[1, idx+1]
            p = np.sum(size_freq_copy[1,:])/np.sum(np.multiply(size_freq_copy[0,:], size_freq_copy[1,:]))
            p_dirac = []
            totalProb = sum([p*(1-p)**(num-1) for num in roundNums])
            totalObsProb = sum(list(diracs.values()))
            for num in roundNums:
                prob = p*(1-p)**(num-1)
                p_dirac.append((num, np.max([0, (diracs[num]*(totalProb - 1)/(totalObsProb - 1)) - prob])))
            params[c] = [p, p_dirac]
        hists = {}
        for k, v in sizes.items():
            p , dd  = params[k]
            pi = np.array([p*(1-p)**k for k in range(1,10000)])
            # pi = pi*(1-sum([d[1] for d in dd]))/sum(pi)
            for i, p_i in dd:
                pi[i-1] = p_i + pi[i-1]
            pi = pi/sum(pi)
            fig = plt.figure()
            n, bins, _ = plt.hist(v, bins = np.arange(1,10000), density = True)
            plt.plot(np.arange(1,10000), pi, color = "red", alpha = 0.5)
            hists[k] = (n, bins)
            plt.title(k)
            plt.xscale("log")
            plt.yscale("log")
            plt.savefig(resultsPath + self.cfg.get("loader").ric + "_Plot_" + self.dates[0] + "_" + self.dates[-1] + "_orderSizes_"+k+".png")
        with open(self.cfg.get("loader").dataPath + self.cfg.get("loader").ric + "_Params_" + self.dates[0] + "_" + self.dates[-1] + "_orderSizesDict", "wb") as f:
            pickle.dump(params, f)

        params = {}
        sizes = copy.deepcopy(queues)
        #,200,500]
        for c in sizes.keys():
            roundNums = [1,10,100, 500, 1000]
            diracs = {}
            size, freq = np.unique(sizes[c], return_counts = True)
            size_freq = np.vstack([size, freq])
            size_freq_copy = copy.deepcopy(size_freq)
            for num in roundNums:
                idx = np.where(size_freq[0,:] == num)[0][0]
                diracs[num] = size_freq_copy[1,idx]/np.sum(freq)
                size_freq_copy[1,idx] = size_freq_copy[1, idx+1]
            p = np.sum(size_freq_copy[1,:])/np.sum(np.multiply(size_freq_copy[0,:], size_freq_copy[1,:]))
            p_dirac = []
            totalProb = sum([p*(1-p)**(num-1) for num in roundNums])
            totalObsProb = sum(list(diracs.values()))
            for num in roundNums:
                prob = p*(1-p)**(num-1)
                p_dirac.append((num, np.max([0, (diracs[num]*(totalProb - 1)/(totalObsProb - 1)) - prob])))
            params[c] = [p, p_dirac]
        hists = {}
        for k, v in sizes.items():
            p , dd  = params[k]
            pi = np.array([p*(1-p)**k for k in range(1,10000)])
            # pi = pi*(1-sum([d[1] for d in dd]))/sum(pi)
            for i, p_i in dd:
                pi[i-1] = p_i + pi[i-1]
            pi = pi/sum(pi)
            fig = plt.figure()
            n, bins, _ = plt.hist(v, bins = np.arange(1,10000), density = True)
            plt.plot(np.arange(1,10000), pi, color = "red", alpha = 0.5)
            hists[k] = (n, bins)
            plt.title(k)
            plt.xscale("log")
            plt.yscale("log")
            plt.savefig(resultsPath + self.cfg.get("loader").ric + "_Plot_" + self.dates[0] + "_" + self.dates[-1] + "_queueSizes_"+k+".png")
        with open(self.cfg.get("loader").dataPath + self.cfg.get("loader").ric + "_Params_" + self.dates[0] + "_" + self.dates[-1] + "_queueSizesDict", "wb") as f:
            pickle.dump(params, f)

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

class MLE():

    def __init__(self, data, **kwargs):
        self.data = data
        self.cfg = kwargs

    def fit(self, tol=1e-6, max_iter=100000, elastic_net_ratio = 0, penalty = "none", solver = "bfgs"):
        hawkes_learner = HawkesSumExpKern(decays = [1.7e3, 0.1*1.7e3, 0.01*1.7e3, 0.001*1.7e3],solver=solver, verbose = True, penalty = penalty, tol=tol, max_iter=max_iter, elastic_net_ratio =elastic_net_ratio)
        hawkes_learner.fit(self.data)
        baseline = hawkes_learner.baseline
        kernels = hawkes_learner.coeffs
        return (baseline, kernels)

class Optimizer:

    def __init__(self, dictInput):
        self.dictIP = dictInput

    def projectBounds(self, params, LB, UB):
        params[params < LB] = LB
        params[params > UB] = UB

    def ComputeWorkingSet(self, params, grad, LB, UB):
        mask = np.ones_like(grad, dtype=int)
        mask[(params < LB + self.optTol * 2) & (grad >= 0)] = 0
        mask[(params > UB - self.optTol * 2) & (grad <= 0)] = 0
        working = np.where(mask == 1)[0]
        return working

    def isLegal(self, x):
        return not np.isnan(x).any()

    def lbfgsUpdate(self, y, s, corrections, old_dirs, old_stps, Hdiag):
        ys = np.dot(y, s)
        if ys > 1e-10:
            numCorrections = old_dirs.shape[1]

            if numCorrections < corrections:
                old_dirs = np.hstack([old_dirs, np.expand_dims(s, axis=1)])
                old_stps = np.hstack([old_stps, np.expand_dims(y, axis=1)])
            else:
                old_dirs[:, :-1] = old_dirs[:, 1:]
                old_stps[:, :-1] = old_stps[:, 1:]
                old_dirs[:, -1] = s
                old_stps[:, -1] = y

            Hdiag = ys / np.dot(y, y)

    def lbfgs(self, g, s, y, Hdiag):
        k = s.shape[1]

        ro = np.zeros(k)
        for i in range(k):
            ro[i] = 1 / np.dot(y[:, i], s[:, i])

        q = np.zeros((len(g), k + 1))
        r = np.zeros((len(g), k + 1))
        al = np.zeros(k)
        be = np.zeros(k)

        q[:, -1] = g

        for i in range(k - 1, -1, -1):
            al[i] = ro[i] * np.dot(s[:, i], q[:, i + 1])
            q[:, i] = q[:, i + 1] - al[i] * y[:, i]

        r[:, 0] = Hdiag * q[:, 0]

        for i in range(k):
            be[i] = ro[i] * np.dot(y[:, i], r[:, i])
            r[:, i + 1] = r[:, i] + s[:, i] * (al[i] - be[i])

        return r[:, -1]

    def PLBFGS(self, LB, UB):
        self.maxIter_ = 10000

        print("{:10} {:10} {:10} {:10} {:10}".format("Iteration", "FunEvals", "Step Length", "Function Val", "Opt Cond"))
        nVars = len(self.process_.GetParameters())

        x = (np.random.randn(nVars) + 1) * 0.5
        self.projectBounds(x, LB, UB)

        f = 0
        g = np.zeros_like(x)
        self.process_.SetParameters(x)
        self.process_.NegLoglikelihood(f, g)

        working = self.ComputeWorkingSet(x, g, LB, UB)

        if len(working) == 0:
            print("All variables are at their bound and no further progress is possible at initial point")
            return
        elif np.linalg.norm(g[working]) <= self.optTol:
            print("All working variables satisfy optimality condition at initial point")
            return

        i = 1
        funEvals = 1
        maxIter = self.maxIter_

        corrections = 100
        old_dirs = np.zeros((nVars, 0))
        old_stps = np.zeros((nVars, 0))
        Hdiag = 0
        suffDec = 1e-4

        g_old = g.copy()
        x_old = x.copy()

        while funEvals < maxIter:
            d = np.zeros_like(x)

            if i == 1:
                d[working] = -g[working]
                Hdiag = 1
            else:
                self.lbfgsUpdate(g - g_old, x - x_old, corrections, old_dirs, old_stps, Hdiag)
                d[working] = self.lbfgs(-g[working], old_dirs[:, working], old_stps[:, working], Hdiag)

            g_old = g.copy()
            x_old = x.copy()

            f_old = f
            gtd = np.dot(g, d)
            if gtd > -self.optTol:
                print("Directional Derivative below optTol")
                break

            if i == 1:
                t = min(1 / np.sum(np.abs(g[working])), 1.0)
            else:
                t = 1.0

            x_new = x + t * d
            self.projectBounds(x_new, LB, UB)
            self.process_.SetParameters(x_new)
            self.process_.NegLoglikelihood(f_new, g_new)
            funEvals += 1

            lineSearchIters = 1
            while f_new > f + suffDec * np.dot(g, x_new - x) or np.isnan(f_new):
                temp = t
                t = 0.1 * t

                if t < temp * 1e-3:
                    t = temp * 1e-3
                elif t > temp * 0.6:
                    t = temp * 0.6

                if np.sum(np.abs(t * d)) < self.optTol:
                    print("Line Search failed")
                    t = 0
                    f_new = f
                    g_new = g
                    break

                x_new = x + t * d
                self.projectBounds(x_new, LB, UB)
                self.process_.SetParameters(x_new)
                self.process_.NegLoglikelihood(f_new, g_new)
                funEvals += 1
                lineSearchIters += 1

            x = x_new
            f = f_new
            g = g_new

            working = self.ComputeWorkingSet(x, g, LB, UB)

            if len(working) == 0:
                print("{:10} {:10} {:10.2f} {:10.2f} {:10}".format(i, funEvals, t, f, 0))
                print("All variables are at their bound and no further progress is possible")
                break
            else:
                print("{:10} {:10} {:10.2f} {:10.2f} {:10.2f}".format(i, funEvals, t, f, np.sum(np.abs(g[working]))))

                if np.linalg.norm(g[working]) <= self.optTol:
                    print("All working variables satisfy optimality condition")
                    break

            if np.sum(np.abs(t * d)) < self.optTol:
                print("Step size below optTol")
                break

            if np.abs(f - f_old) < self.optTol:
                print("Function value changing by less than optTol")
                break

            if funEvals > maxIter:
                print("Function Evaluations exceed maxIter")
                break

            i += 1

        print()

class PlainHawkes:

    def __init__(self):
        self.num_sequences_ = 0
        self.num_dims_ = 0
        self.all_exp_kernel_recursive_sum_ = []
        self.all_timestamp_per_dimension_ = []
        self.observation_window_T_ = np.array([])
        self.intensity_itegral_features_ = []
        self.parameters_ = np.array([])
        self.Beta_ = np.array([])
        self.options_ = None

    def InitializeDimension(self, data):
        num_sequences_ = len(data)
        self.all_timestamp_per_dimension_ = [[[[] for _ in range(self.num_dims_)] for _ in range(num_sequences_)]]

        for c in range(num_sequences_):
            seq = data[c].GetEvents()

            for event in seq:
                self.all_timestamp_per_dimension_[c][event.DimentionID].append(event.time)

    def Initialize(self, data):
        self.num_sequences_ = len(data)
        self.num_dims_ = data[0].num_dims()
        self.all_exp_kernel_recursive_sum_ = np.empty((self.num_sequences_, self.num_dims_, self.num_dims_), dtype=object)
        self.all_timestamp_per_dimension_ = []  # will be initialized in InitializeDimension function
        self.observation_window_T_ = np.zeros(self.num_sequences_)
        self.intensity_integral_features_ = np.zeros((self.num_sequences_, self.num_dims_, self.num_dims_))

        self.InitializeDimension(data)

        for k in range(self.num_sequences_):
            for m in range(self.num_dims_):
                for n in range(self.num_dims_):
                    if len(self.all_timestamp_per_dimension_[k][n]) > 0:
                        self.all_exp_kernel_recursive_sum_[k][m][n] = np.zeros(len(self.all_timestamp_per_dimension_[k][n]))

                        if m != n:
                            for j in range(len(self.all_timestamp_per_dimension_[k][m])):
                                if self.all_timestamp_per_dimension_[k][m][j] < self.all_timestamp_per_dimension_[k][n][0]:
                                    self.all_exp_kernel_recursive_sum_[k][m][n][0] += np.exp(-self.Beta_[m, n] * (self.all_timestamp_per_dimension_[k][n][0] - self.all_timestamp_per_dimension_[k][m][j]))

                            for i in range(1, len(self.all_timestamp_per_dimension_[k][n])):
                                value = np.exp(-self.Beta_[m, n] * (self.all_timestamp_per_dimension_[k][n][i] - self.all_timestamp_per_dimension_[k][n][i - 1])) * self.all_exp_kernel_recursive_sum_[k][m][n][i - 1]

                                for j in range(len(self.all_timestamp_per_dimension_[k][m])):
                                    if (self.all_timestamp_per_dimension_[k][n][i - 1] <= self.all_timestamp_per_dimension_[k][m][j] < self.all_timestamp_per_dimension_[k][n][i]):
                                        value += np.exp(-self.Beta_[m, n] * (self.all_timestamp_per_dimension_[k][n][i] - self.all_timestamp_per_dimension_[k][m][j]))

                                self.all_exp_kernel_recursive_sum_[k][m][n][i] = value
                        else:
                            for i in range(1, len(self.all_timestamp_per_dimension_[k][n])):
                                self.all_exp_kernel_recursive_sum_[k][m][n][i] = np.exp(-self.Beta_[m, n] * (self.all_timestamp_per_dimension_[k][n][i] - self.all_timestamp_per_dimension_[k][n][i - 1])) * (1 + self.all_exp_kernel_recursive_sum_[k][m][n][i - 1])

        for c in range(self.num_sequences_):
            self.observation_window_T_[c] = data[c].GetTimeWindow()

            for m in range(self.num_dims_):
                for n in range(self.num_dims_):
                    event_dim_m = np.array(self.all_timestamp_per_dimension_[c][m])
                    self.intensity_integral_features_[c, m, n] = (1 - np.exp(-self.Beta_[m, n] * (self.observation_window_T_[c] - event_dim_m))).sum()

    def Intensity(self, t, data):
        intensity_dim = np.zeros(self.num_dims_)

        Lambda0_ = self.parameters_[:self.num_dims_]
        Alpha_ = self.parameters_[self.num_dims_:].reshape(self.num_dims_, self.num_dims_)

        intensity_dim = Lambda0_

        seq = data.GetEvents()

        for event in seq:
            if event.time < t:
                for d in range(self.num_dims_):
                    intensity_dim[d] += Alpha_[event.DimensionID, d] * np.exp(-self.Beta_[event.DimensionID, d] * (t - event.time))
            else:
                break

        return np.sum(intensity_dim)

    def IntensityUpperBound(self, t, L, data):
        intensity_upper_dim = np.zeros(self.num_dims_)

        Lambda0_ = self.parameters_[:self.num_dims_]
        Alpha_ = self.parameters_[self.num_dims_:].reshape(self.num_dims_, self.num_dims_)

        intensity_upper_dim = Lambda0_

        seq = data.GetEvents()

        for event in seq:
            if event.time <= t:
                for d in range(self.num_dims_):
                    intensity_upper_dim[d] += Alpha_[event.DimensionID, d] * np.exp(-self.Beta_[event.DimensionID, d] * (t - event.time))
            else:
                break

        return np.sum(intensity_upper_dim)

    def NegLoglikelihood(self, objvalue, gradient):
        if not self.all_timestamp_per_dimension_:
            print("Process is uninitialized with any data.")
            return

        gradient[:] = np.zeros(self.num_dims_ * (1 + self.num_dims_))

        grad_lambda0_vector = gradient[:self.num_dims_].reshape(-1, 1)
        grad_alpha_matrix = gradient[self.num_dims_:].reshape(self.num_dims_, self.num_dims_)

        Lambda0_ = self.parameters_[:self.num_dims_].reshape(-1, 1)
        Alpha_ = self.parameters_[self.num_dims_:].reshape(self.num_dims_, self.num_dims_)

        objvalue[0] = 0

        for k in range(self.num_sequences_):
            timestamp_per_dimension = self.all_timestamp_per_dimension_[k]
            exp_kernel_recursive_sum = self.all_exp_kernel_recursive_sum_[k]

            for n in range(self.num_dims_):
                obj_n = 0

                for i in range(len(timestamp_per_dimension[n])):
                    local_sum = Lambda0_[n] + 1e-4

                    for m in range(self.num_dims_):
                        local_sum += Alpha_[m, n] * exp_kernel_recursive_sum[m][n][i]

                    obj_n += np.log(local_sum)

                    grad_lambda0_vector[n] += 1 / local_sum

                    for m in range(self.num_dims_):
                        grad_alpha_matrix[m, n] += exp_kernel_recursive_sum[m][n][i] / local_sum

                obj_n -= ((Alpha_[:, n] / self.Beta_[:, n]) * self.intensity_integral_features_[k][:, n]).sum()

                grad_alpha_matrix[:, n] -= self.intensity_integral_features_[k][:, n] / self.Beta_[:, n]

                obj_n -= self.observation_window_T_[k] * Lambda0_[n]

                grad_lambda0_vector[n] -= self.observation_window_T_[k]

                objvalue[0] += obj_n

        gradient /= -self.num_sequences_
        objvalue /= -self.num_sequences_

        # Regularization for base intensity
        if self.options_.base_intensity_regularizer == 'L22':
            grad_lambda0_vector += self.options_.coefficients["LAMBDA"] * Lambda0_
            objvalue += 0.5 * self.options_.coefficients["LAMBDA"] * np.sum(Lambda0_ ** 2)

        elif self.options_.base_intensity_regularizer == 'L1':
            grad_lambda0_vector += self.options_.coefficients["LAMBDA"]
            objvalue += self.options_.coefficients["LAMBDA"] * np.sum(np.abs(Lambda0_))

        # Regularization for excitation matrix
        grad_alpha_vector = gradient[self.num_dims_:].reshape(-1, 1)
        alpha_vector = self.parameters_[self.num_dims_:].reshape(-1, 1)

        if self.options_.excitation_regularizer == 'L22':
            grad_alpha_vector += self.options_.coefficients["BETA"] * alpha_vector
            objvalue += 0.5 * self.options_.coefficients["BETA"] * np.sum(alpha_vector ** 2)

        elif self.options_.excitation_regularizer == 'L1':
            grad_alpha_vector += self.options_.coefficients["BETA"]
            objvalue += self.options_.coefficients["BETA"] * np.sum(np.abs(alpha_vector))

        return

    def Gradient(self, k, gradient):
        if not self.all_timestamp_per_dimension_:
            print("Process is uninitialized with any data.")
            return

        gradient[:] = np.zeros(self.num_dims_ * (1 + self.num_dims_))

        grad_lambda0_vector = gradient[:self.num_dims_].reshape(-1, 1)
        grad_alpha_matrix = gradient[self.num_dims_:].reshape(self.num_dims_, self.num_dims_)

        Lambda0_ = self.parameters_[:self.num_dims_].reshape(-1, 1)
        Alpha_ = self.parameters_[self.num_dims_:].reshape(self.num_dims_, self.num_dims_)

        timestamp_per_dimension = self.all_timestamp_per_dimension_[k]
        exp_kernel_recursive_sum = self.all_exp_kernel_recursive_sum_[k]

        for n in range(self.num_dims_):
            for i in range(len(timestamp_per_dimension[n])):
                local_sum = Lambda0_[n]

                for m in range(self.num_dims_):
                    local_sum += Alpha_[m, n] * exp_kernel_recursive_sum[m][n][i]

                grad_lambda0_vector[n] += 1 / local_sum

                for m in range(self.num_dims_):
                    grad_alpha_matrix[m, n] += exp_kernel_recursive_sum[m][n][i] / local_sum

            grad_alpha_matrix[:, n] -= self.intensity_integral_features_[k][:, n] / self.Beta_[:, n]

            grad_lambda0_vector[n] -= self.observation_window_T_[k]

        gradient /= -self.num_sequences_


    def fit(self, data, options):
        self.Initialize(data)

        self.options_ = options

        opt = Optimizer(self)

        opt.PLBFGS(0, 1e10)

        self.RestoreOptionToDefault()


    def PredictNextEventTime(self, data, num_simulations):
        pass  # Implementation not provided

    def IntensityIntegral(self, lower, upper, data):
        sequences = [data]
        self.InitializeDimension(sequences)

        Lambda0_ = np.array(self.parameters_[:self.num_dims_])
        Alpha_ = np.array(self.parameters_[self.num_dims_:].reshape(self.num_dims_, self.num_dims_))

        timestamp_per_dimension = self.all_timestamp_per_dimension_[0]

        integral_value = 0

        for n in range(self.num_dims_):
            integral_value += Lambda0_[n] * (upper - lower)

            for m in range(self.num_dims_):
                event_dim_m = np.array(timestamp_per_dimension[m])

                mask = (event_dim_m < lower).astype(float)
                a = (mask * (((-self.Beta_[m, n] * (lower - event_dim_m)) * mask).exp() - ((-self.Beta_[m, n] * (upper - event_dim_m)) * mask).exp())).sum()

                mask = ((event_dim_m >= lower) & (event_dim_m < upper)).astype(float)
                b = (mask * (1 - ((-self.Beta_[m, n] * (upper - event_dim_m)) * mask).exp())).sum()

                integral_value += (Alpha_[m, n] / self.Beta_[m, n]) * (a + b)

        return integral_value

    def RestoreOptionToDefault(self):
        pass  # Implementation not provided

    def AssignDim(self, intensity_dim):
        pass  # Implementation not provided

    def UpdateExpSum(self, t, last_event_per_dim, expsum):
        pass  # Implementation not provided

    def Simulate(self, vec_T, sequences):
        pass  # Implementation not provided

    def Simulate(self, n, num_sequences, sequences):
        pass  # Implementation not provided


