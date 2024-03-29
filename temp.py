##Extracting data:
# import py7zr
#
# with py7zr.SevenZipFile('/cs/academic/phd3/konajain/data/AAPL_2019-01-01_2020-09-27_10.7z', mode='r') as z:
#     z.extractall()
import datetime as dt
import pickle
# from hawkes import dataLoader, fit, inference, \
from hawkes import simulate
# from hawkes.qof import *
import pandas as pd
import numpy as np
import os

def main():
    # ric = "fake"
    # sDate = dt.date(2019,1,2)
    # eDate = dt.date(2019,1,2)
    # for d in range(1,101):
    #     l = dataLoader.Loader(ric, d, d, nlevels = 2)#, dataPath = "/SAN/fca/Konark_PhD_Experiments/simulated/fakeData/")
    #     if os.path.exists(l.dataPath+ric +"_"+str(d)+ "_12D.csv"):
    #         df = pd.read_csv(l.dataPath+ric +"_"+str(d)+ "_12D.csv")
    #         eventOrder = ["lo_deep_Ask", "co_deep_Ask", "lo_top_Ask","co_top_Ask", "mo_Ask", "lo_inspread_Ask" , "lo_inspread_Bid" , "mo_Bid", "co_top_Bid", "lo_top_Bid", "co_deep_Bid","lo_deep_Bid" ]
    #         data = { str(d) : list(df.groupby('event')['Time'].apply(np.array)[eventOrder].values)}
    #     else:
    #         data = l.load12DTimestamps()
    #     #df = pd.read_csv(l.dataPath+"AAPL.OQ_2020-09-14_12D.csv")
    #     #df = df.loc[df.Time < 100]
    #
    #     cls = fit.ConditionalLeastSquaresLogLin(data, loader = l, solver = "osqp") #, numDataPoints = 100, min_lag = 1e-2)
    #     cls.fitConditionalInSpread()
    #     # with open(l.dataPath + ric + "_" + str(sDate) + "_" + str(eDate) + "_CLSLogLin" , "wb") as f: #"/home/konajain/params/"
    #     #     pickle.dump(thetas, f)
    # return 0
    # ric = "AAPL.OQ"
    # d = dt.date(2020,9,14)
    # l = dataLoader.Loader(ric, d, d, nlevels = 2, dataPath = "/home/konajain/data/")
    # #a = l.load12DTimestamps()
    # df = pd.read_csv("/home/konajain/data/AAPL.OQ_2020-09-14_12D.csv")
    #         eventOrder = np.append(df.event.unique()[6:], df.event.unique()[-7:-13:-1])
    #         timestamps = [list(df.groupby('event')['Time'].apply(np.array)[eventOrder].values)]
    #         cls = fit.ConditionalLaw(timestamps)
    #         params = cls.fit()
    #         with open("/home/konajain/params/" + ric + "_" + str(d) + "_" + str(d) + "_condLaw" , "wb") as f: #"/home/konajain/params/"
    #             pickle.dump(params, f)
    # return params

    # with open('D:\\Work\\PhD\\Expt 1\\params\\AAPL.OQ_2020-09-14_2020-09-14_CLSLogLin', 'rb') as f:
    #     params = pickle.load(f)
    # res = params['2020-09-14 00:00:00'].T
    # num_datapoints = 10
    # min_lag =  1e-3
    # max_lag = 500
    # timegridLin = np.linspace(0,min_lag, num_datapoints)
    # timegridLog = np.exp(np.linspace(np.log(min_lag), np.log(max_lag), num_datapoints))
    # timegrid = np.append(timegridLin[:-1], timegridLog)
    # from hawkes import inference
    # cls = inference.ParametricFit([(timegrid[2:-1], np.abs(res[4,:].reshape((18,12))[:,4][1:-1])/timegrid[2:-1]), (timegrid[2:-1], np.abs(res[7,:].reshape((18,12))[:,7][1:-1])/timegrid[2:-1])])
    # cls.fitBoth()
    # return 0
    #
    # ric = "AAPL.OQ"
    # sDate = dt.date(2019,1,2)
    # eDate = dt.date(2019,1,2)
    # dictIp = {}
    # for d in pd.date_range(sDate, eDate):
    #     l =  dataLoader.Loader(ric, d, d, nlevels = 2) #, dataPath = "/SAN/fca/DRL_HFT_Investigations/LOBSimulations/extracted/")
    #     if os.path.exists(l.dataPath+"AAPL.OQ_"+ d.strftime("%Y-%m-%d") + "_" + d.strftime("%Y-%m-%d") + "_19_inputRes"):
    #         dictIp.update({ d.strftime("%Y-%m-%d") : []})
    #     else:
    #         continue
    #     #df = pd.read_csv(l.dataPath+"AAPL.OQ_2020-09-14_12D.csv")
    #     #df = df.loc[df.Time < 100]
    #
    # cls = fit.ConditionalLeastSquaresLogLin(dictIp, loader = l, solver="osqp") #, numDataPoints = 100, min_lag = 1e-2)
    # thetas = cls.fitConditionalInSpread()
    # # with open(l.dataPath + ric + "_Params_" + str(sDate.strftime("%Y-%m-%d")) + "_" + str(eDate.strftime("%Y-%m-%d")) + "_CLSLogLin_20" , "wb") as f: #"/home/konajain/params/"
    # #     pickle.dump(thetas, f)
    # return 0

    # inference.run(dt.date(2019,1,2),dt.date(2019,1,14), suffix = "_cvx") #suffix = "_sgd"
    # inference.run(dt.date(2019,1,2),dt.date(2019,1,14), suffix = "_tod_cvx")
    # inference.run(dt.date(2019,1,2),dt.date(2019,1,31), suffix = "_todIS_cvx3")
    # return 0
    T, lob, lobL3=simulate.simulateMarketImpactStudy(6.5*3600, "D:\\Work\\PhD\\Expt 1\\params\\AAPL.OQ_ParamsInferredWCutoff_2019-01-02_2019-03-31_CLSLogLin_10", "D:\\Work\\PhD\\Expt 1\\params\\AAPL.OQ_Params_2019-01-02_2019-03-29_dictTOD", orderInitTime = 1)
    # for l in lob:
    #     print(l)
    # with open("D:\\Work\\PhD\\Expt 1\\results\\AAPL.OQ_ResultsWCutoff_2019-01-02_2019-01-31_CLSLogLin__todIS_cvx3_10" , "wb") as f: #"/home/konajain/params/"
    #     pickle.dump((T, lob, lobL3), f)
    # return T, lob, lobL3
    # T, lob, lobL3=simulate.simulate(6.5*3600, "D:\\Work\\PhD\\Expt 1\\params\\AAPL.OQ_ParamsInferredWCutoff_2019-01-02_2019-03-31_CLSLogLin_10", "D:\\Work\\PhD\\Expt 1\\params\\AAPL.OQ_Params_2019-01-02_2019-03-29_dictTOD")
    # with open("D:\\Work\\PhD\\Expt 1\\results\\AAPL.OQ_ResultsWCutoff_2019-01-02_2019-03-31_CLSLogLin_10_"+str(i) , "wb") as f: #"/home/konajain/params/"
    #     pickle.dump((T, lob, lobL3), f)
    # runQQInterArrivalTrapezoid("AAPL.OQ", dt.date(2019,1,2), dt.date(2019,1,2), "D:\Work\PhD\Data\\", delta = 1e-1, inputDataPath = "D:\Work\PhD\Data\\")
    return



main()

# df = pd.read_csv("/SAN/fca/DRL_HFT_Investigations/LOBSimulations/extracted/AAPL.OQ_2019-01-10_12D.csv")
# eventOrder = np.append(df.event.unique()[6:], df.event.unique()[-7:-13:-1])
# arrs = list(df.groupby('event')['Time'].apply(np.array)[eventOrder].values)
# num_datapoints = 10
# min_lag =  1e-3
# max_lag = 500
# timegridLin = np.linspace(0,min_lag, num_datapoints)
# timegridLog = np.exp(np.linspace(np.log(min_lag), np.log(max_lag), num_datapoints))
# timegrid = np.append(timegridLin[:-1], timegridLog)
# timegrid_new = np.floor(timegrid/(timegrid[1] - timegrid[0])).astype(int)
# ser = []
# bins = np.arange(0, np.max([np.max(arr) for arr in arrs]) + 1e-9, (timegrid[1] - timegrid[0]))
# cols = ["lo_deep_Ask", "co_deep_Ask", "lo_top_Ask","co_top_Ask", "mo_Ask", "lo_inspread_Ask" ,
#         "lo_inspread_Bid" , "mo_Bid", "co_top_Bid", "lo_top_Bid", "co_deep_Bid","lo_deep_Bid" ]
# for arr, col in zip(arrs, cols):
#     print(col)
#     arr = np.max(arr) - arr
#     assignedBins = np.searchsorted(bins, arr, side="right")
#     binDf = np.unique(assignedBins, return_counts = True)
#     binDf = pd.DataFrame({"bin" : binDf[0], col : binDf[1]})
#     binDf = binDf.set_index("bin")
#     #binDf = binDf.reset_index()
#     ser += [binDf]
#
# print("done with binning")
# df = pd.concat(ser, axis = 1)
# df = df.fillna(0)
# df = df.sort_index()


# res_d = []
# with open("/SAN/fca/DRL_HFT_Investigations/LOBSimulations/extracted/AAPL.OQ_2019-01-02_2019-01-02_inputRes" , "rb") as f: #"/home/konajain/params/"
#     while True:
#         try:
#             r_d = pickle.load(f)
#             if len(r_d[0]) == 2:
#                 r_d = sum(r_d, [])
#             res_d.append(r_d)
#         except EOFError:
#             break

# col = ['Ask Price ', 'Ask Size ', 'Bid Price ', 'Bid Size ']
#
# theNames = []
# cols = []
# for i in range(1, 11):
#     for j in col:
#         cols.append(str(j)+str(i))
#         theNames.append(str(j) + str(i))
#
# for d in pd.date_range(dt.date(2019,1,2), dt.date(2019,12,31)):
#     try:
#         df = pd.read_csv("/SAN/fca/Konark_PhD_Experiments/extracted/INTC_"+d.strftime("%Y-%m-%d")+"_34200000_57600000_orderbook_10.csv",names=theNames)
#         df['spread'] = df['Ask Price 1'] - df['Bid Price 1']
#         print(len(df.loc[df.spread == 100])/len(df))
#     except:
#         continue


