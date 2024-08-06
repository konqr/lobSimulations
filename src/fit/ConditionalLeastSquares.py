import numpy as np
from statsmodels.tsa.api import VAR

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