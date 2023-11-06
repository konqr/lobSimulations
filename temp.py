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
    sDate = dt.date(2019,1,2)
    eDate = dt.date(2019,1,2)
    binLength = 1
    p = 300
    T = 1000
    l = dataLoader.Loader(ric, sDate, eDate, nlevels = 2, dataPath = "/home/konajain/data/")
    data = l.loadRollingWindows(binLength = binLength, filterTop = True)
    cls = fit.ConditionalLeastSquares(data, p, 1, T=T)
    thetas = cls.fit()
    with open("/home/konajain/params/" + ric + "_" + str(sDate) + "_" + str(eDate) + "_" + str(binLength) + "_" + str(p) + "_" + str(T)) as f:
        pickle.dump(thetas, f)
    return thetas

main()
