import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
import statsmodels.api as sm
#from tick.hawkes import HawkesConditionalLaw
import pickle
import gc
from sklearn.linear_model import LinearRegression, ElasticNet, SGDRegressor, Ridge
import time

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
            delta_lag=self.cfg.get("delta_lag",1e-1),
            min_lag=self.cfg.get("min_lag",1),
            max_lag=self.cfg.get("max_lag",500),
            quad_method=self.cfg.get("quad_method","log"),
            n_quad=self.cfg.get("n_quad",200),
            min_support=self.cfg.get("min_support",1e-4),
            max_support=self.cfg.get("max_support",10),
            n_threads=self.cfg.get("n_threads",4)
        )
        hawkes_learner.fit(self.data)
        baseline = hawkes_learner.baseline
        kernels = hawkes_learner.kernels
        kernel_norms = hawkes_learner.kernels_norms
        return (baseline, kernels, kernel_norms)

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
        timegrid_new = np.floor(timegrid/(timegrid[1] - timegrid[0])).astype(int)
        ser = []
        bins = np.arange(0, np.max([np.max(arr) for arr in arrs]) + 1e-9, (timegrid[1] - timegrid[0]))
        for arr, col in zip(arrs, self.cols):
            print(col)
            arr = np.max(arr) - arr
            assignedBins = np.searchsorted(bins, arr, side="right")
            binDf = np.unique(assignedBins, return_counts = True)
            binDf = pd.DataFrame({"bin" : binDf[0], col : binDf[1]})
            binDf = binDf.set_index("bin")
            #binDf = binDf.reset_index()
            ser += [binDf]
        print("done with binning")
        df = pd.concat(ser, axis = 1)
        df = df.fillna(0)
        df = df.sort_index()
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
        # for i in range(restartIdx+1,len(df)-1):
        #     print(i)
        #     idx = df.index[i]
        #     df['binIndexNew'] = np.searchsorted(timegrid_new, df.index - idx, side="right")
        #     lastIdx = len(timegrid_new)
        #     dfFiltered = df.loc[df['binIndexNew'] != lastIdx]
        #     binDf = dfFiltered.loc[dfFiltered.index[i+1]:].groupby("binIndexNew")[self.cols].sum()
        #
        #     binDf['const'] = 1.
        #     if len(binDf) < len(timegrid_new):
        #         missing = np.setdiff1d(np.arange(1,len(timegrid_new)+1), binDf.index, assume_unique=True)
        #         empty = pd.DataFrame(index=missing)
        #         binDf = pd.concat([empty, binDf], axis=1).fillna(0.)
        #         binDf = binDf.sort_index()
        #     lags = binDf.values
        #     res.append([df[self.cols].loc[idx].values, lags])

        for i in range(restartIdx + 1, len(df) - 1):

            idx = df.index[i]
            bin_index_new = np.searchsorted(timegrid_new, df.index - idx, side="right")
            last_idx = len(timegrid_new)

            df['binIndexNew'] = bin_index_new
            df_filtered = df[df['binIndexNew'] != last_idx]

            unique_bins, bin_counts = np.unique(df_filtered['binIndexNew'], return_counts=True)

            bin_df = np.zeros((len(timegrid_new) - 1, len(self.cols)))
            df_filtered = df_filtered.loc[df_filtered.index[i+1]:]
            for j, col in enumerate(self.cols):
                bin_df[:, j] = np.bincount(df_filtered['binIndexNew'], weights=df_filtered[col], minlength=len(timegrid_new))[1:]

            #bin_df = np.hstack((bin_df, np.ones((len(unique_bins), 1))))  # Adding 'const' column

            # if len(unique_bins) < len(timegrid_new):
            #     missing = np.setdiff1d(np.arange(1, len(timegrid_new) + 1), unique_bins, assume_unique=True)
            #     empty = np.zeros((len(missing), len(self.cols) + 1))
            #     bin_df = np.vstack((empty, bin_df))
            #     bin_df = bin_df[np.argsort(bin_df[:, 0])]

            lags = bin_df
            res.append([df.loc[idx, self.cols].values, lags])

            if i%5000 == 0 :
                print(i)
                with open(self.cfg.get("loader").dataPath + self.cfg.get("loader").ric + "_" + date + "_" + date + "_" + str(len(timegrid)) + "_inputRes" , "ab") as f: #"/home/konajain/params/"
                    pickle.dump(res, f)
                res =[]
                gc.collect()
            elif i==len(df)-2:
                with open(self.cfg.get("loader").dataPath + self.cfg.get("loader").ric + "_" + date + "_" + date + "_" + str(len(timegrid)) + "_inputRes" , "ab") as f: #"/home/konajain/params/"
                    pickle.dump(res, f)
                res =[]
                gc.collect()

        return res

    def runTransformDate(self):
        num_datapoints = self.cfg.get("num_datapoints", 10)
        min_lag = self.cfg.get("min_lag", 1e-3)
        max_lag = self.cfg.get("max_lag" , 500)
        timegridLin = np.linspace(0,min_lag, num_datapoints)
        timegridLog = np.exp(np.linspace(np.log(min_lag), np.log(max_lag), num_datapoints))
        timegrid = np.append(timegridLin[:-1], timegridLog)
        # can either use np.histogram with custom binsize for adaptive grid
        # or
        # bin data by delta_lag*min_lag and then add bins for exponential lags

        for i in self.dates:
            dictPerDate = self.dictBinnedData[i]
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
                    except EOFError:
                        break
            res_d = sum(res_d, [])
            Ys = np.array([res_d[i] for i in range(0,len(res_d),2)])
            Xs = [res_d[i+1] for i in range(0,len(res_d),2)]
            Xs = [np.append([1],r.flatten()) for r in Xs]
            print(len(Xs))

            # model = sm.OLS(Ys, Xs)
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
                lr = Ridge( solver="svd", alpha = 1e-6).fit(Xs, Ys)
                print(lr.score(Xs, Ys))
                params = (lr.intercept_, lr.coef_)

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

            Xs = np.array([r.flatten() for r in Xs])
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
            dummiesIS = dummies / (df['spread'].values.sum(axis = 1)[:,np.newaxis])**spreadBeta
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
            else:
                model = SGDRegressor(penalty = 'l2', alpha = 1e-6, fit_intercept=False, max_iter=5000, verbose=11, learning_rate = "adaptive", eta0 = 1e-6).fit(XsIS, Ys_inspreadBid)
                params2 = model.coef_


                model = SGDRegressor(penalty = 'l2', alpha = 1e-6, fit_intercept=False, max_iter=5000, verbose=11, learning_rate = "adaptive", eta0 = 1e-6).fit(XsIS, Ys_inspreadAsk)
                params3 = model.coef_


                models = [SGDRegressor(penalty = 'l2', alpha = 1e-6, fit_intercept=False, max_iter=5000, verbose=11, learning_rate = "adaptive", eta0 = 1e-6).fit(Xs_oth, Ys_oth[:,i]) for i in range(Ys_oth.shape[1])]
                params1 = [model.coef_ for model in models]

            thetas[i] = (params1, params2, params3) #, paramsUncertainty)
        return thetas


