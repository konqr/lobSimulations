##Extracting data:
# import py7zr
#
# with py7zr.SevenZipFile('/cs/academic/phd3/konajain/data/AAPL_2019-01-01_2020-09-27_10.7z', mode='r') as z:
#     z.extractall()
import datetime as dt
import pickle
from hawkes import dataLoader, fit

def main():
    ric = "AAPL.OQ"
    sDate = dt.date(2019,1,7)
    eDate = dt.date(2019,1,7)
    binLength = 0.1
    p = 900

    l = dataLoader.Loader(ric, sDate, eDate, nlevels = 2) #, dataPath = "/home/konajain/data/"
    data = l.loadBinned(binLength = binLength, filterTop = True)
    T = len(data[sDate.strftime("%Y-%m-%d")]['limit_bid'])
    cls = fit.ConditionalLeastSquares(data, p, 1, T=T)
    thetas = cls.fit()
    with open("D:\\Work\\PhD\\Data\\" + ric + "_" + str(sDate) + "_" + str(eDate) + "_" + str(binLength) + "_" + str(p) + "_" + str(T) , "wb") as f: #"/home/konajain/params/"
        pickle.dump(thetas, f)
    return thetas

main()
