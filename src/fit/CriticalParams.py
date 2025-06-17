import sys
sys.path.append("/home/konajain/code/lobSimulations")
from src.data import dataLoader
import pandas as pd
import datetime as dt
import numpy as np

def fit_eta_hat():
    stocks = ['SIRI','BAC', 'INTC','CSCO','ORCL','MSFT','AAPL','ABBV', 'PM','IBM','TSLA','CHTR','AMZN', 'GOOG', 'BKNG']
    results = {}
    for ric in stocks:
        res = {}
        for j in pd.date_range(dt.date(2019,1,2), dt.date(2019,2,4)):
            if j == dt.date(2019,1,9): continue
            l = dataLoader.dataLoader(ric, j, j, nlevels = 10, dataPath = "/SAN/fca/Konark_PhD_Experiments/extracted/GOOG/")
            data = l.load()
            master_dict = {}
            if len(data):
                data = data[0]
            else:
                continue
            data['timeDiff'] = data['Time'].diff()
            for i in range(1,11):
                data['diff'] = data['Ask Price ' + str(i)].shift(-1) - data['Ask Price ' + str(i)]
                res[i] = np.append(res.get(i, []), data['diff'].loc[data['diff'] > 0].values)
                data['diff'] = data['Bid Price ' + str(i)] - data['Bid Price ' + str(i)].shift(-1)
                res[i] = np.append(res.get(i, []), data['diff'].loc[data['diff'] > 0].values)
        eta_hat = 1./np.average(np.average(100*res[1], axis = 0)) # geometric distribution prior
        results[ric] = eta_hat
    return results

def main():
    results = fit_eta_hat()
    print(results)

main()