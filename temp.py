##Extracting data:
# import py7zr
#
# with py7zr.SevenZipFile('/cs/academic/phd3/konajain/data/AAPL_2019-01-01_2020-09-27_10.7z', mode='r') as z:
#     z.extractall()
import datetime as dt

from hawkes import dataLoader, fit

def main():
    l = dataLoader.Loader("AAPL.OQ", dt.date(2019,1,2), dt.date(2019,1,3), nlevels = 2)
    data = l.loadBinned(binLength = 0.5)
    cls = fit.ConditionalLeastSquares(data, 300, 0.5)
    thetas = cls.fit()
    return thetas

main()
