##Extracting data:
# import py7zr
#
# with py7zr.SevenZipFile('/cs/academic/phd3/konajain/data/AAPL_2019-01-01_2020-09-27_10.7z', mode='r') as z:
#     z.extractall()
import datetime as dt
import pickle
from hawkes import dataLoader, fit
import pandas as pd
import numpy as np

def main():
    ric = "AAPL.OQ"
    sDate = dt.date(2020,9,14)
    eDate = dt.date(2020,9,14)
    binLength = 1e-4
    p = 30
    for d in pd.date_range(sDate, eDate):
        l = dataLoader.Loader(ric, d, d, nlevels = 2) #, dataPath = "/home/konajain/data/"
        #data = l.load12DTimestamps()
        df = pd.read_csv("D:\\Work\\PhD\\Data\\AAPL.OQ_2020-09-14_12D.csv")
        df = df.loc[df.Time < 100]
        eventOrder = np.append(df.event.unique()[6:], df.event.unique()[-7:-13:-1])
        data = {'2020-09-14' : list(df.groupby('event')['Time'].apply(np.array)[eventOrder].values)}
        cls = fit.ConditionalLeastSquaresLogLin(data, loader = l)
        thetas = cls.fit()
        # with open("/home/konajain/params/" + ric + "_" + str(sDate) + "_" + str(eDate) + "_" + str(binLength) + "_" + str(p) + "_" + str(T) , "wb") as f: #"/home/konajain/params/"
        #     pickle.dump(thetas, f)
    return thetas
    # ric = "AAPL.OQ"
    # d = dt.date(2020,9,14)
    # l = dataLoader.Loader(ric, d, d, nlevels = 2, dataPath = "/home/konajain/data/")
    # #a = l.load12DTimestamps()
    # df = pd.read_csv("/home/konajain/data/AAPL.OQ_2020-09-14_12D.csv")
    # eventOrder = np.append(df.event.unique()[6:], df.event.unique()[-7:-13:-1])
    # timestamps = [list(df.groupby('event')['Time'].apply(np.array)[eventOrder].values)]
    # cls = fit.ConditionalLaw(timestamps)
    # params = cls.fit()
    # with open("/home/konajain/params/" + ric + "_" + str(d) + "_" + str(d) + "_condLaw" , "wb") as f: #"/home/konajain/params/"
    #     pickle.dump(params, f)
    # return params

main()
