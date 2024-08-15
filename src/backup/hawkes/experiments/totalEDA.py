import datetime as dt
import pandas as pd
from matplotlib import pyplot as plt
import os
import sys
sys.path.append("/home/konajain/code/lobSimulations")
from src.backup.hawkes import dataLoader
import numpy as np
import time
from IPython import get_ipython
import itertools
import pickle

def nanmed(data):
    d = {}
    d['q_LO'] = np.nanmedian(data['q_LO'])
    d['q_MO'] = np.nanmedian(data['q_MO'])
    d['eta_is'] = np.nanmedian(data['eta_is'])
    return pd.Series(d)

def main(ric, edaspread = False, edashape = False, edasparse = False, edarest = False, edaqd = False, edashapemaxima = False, edashapesparsity = False, edaleverage = False, edaleverage_top = False,edaleverageIS=False, assumptions = False):


    # # Spread

    # In[64]:

    ric = ric
    samplingTime = 60
    if edaspread:

        # In[65]:


        spreads = []
        for j in pd.date_range(dt.date(2019,1,2), dt.date(2019,12,31)):
            l = dataLoader.Loader(ric, j, j, nlevels = 1, dataPath = "/SAN/fca/Konark_PhD_Experiments/extracted/GOOG/")
            data = l.load()
            if len(data):
                data = data[0]
            else:
                continue

            data['spread'] = data['Ask Price 1'] - data['Bid Price 1']
            data['timeDiff'] = data['Time'].diff()
            data['spreadTwa'] = data['spread']*data['timeDiff']
            data['id'] = data['Time'].apply(lambda x: int((x - 34200)//samplingTime))
            twaspread = (data[['spreadTwa', 'id']].groupby('id').sum().values)/samplingTimeUW
            spreads.append(twaspread)


        # In[137]:


        #get_ipython().run_line_magic('matplotlib', 'inline')
        plt.figure()
        for s in spreads:
            plt.plot(100*s, alpha = 0.01, color = "steelblue")
        avg = np.average(100*np.array([s for s in spreads if len(s) == 390]), axis = 0)
        plt.plot(avg, color="r", label = "average")
        plt.xlabel("Time")
        plt.ylabel("Spread in ticks")
        plt.legend()
        plt.title("Spread by TOD - " + ric)
        plt.xticks(ticks = np.arange(0, 23400//samplingTime, 1800//samplingTime), labels = [time.strftime('%H:%M:%S', time.gmtime(x)) for x in 9.5*3600 + samplingTime*np.arange(0, 23400//samplingTime, 1800//samplingTime)], rotation = 20)
        plt.savefig("/SAN/fca/Konark_PhD_Experiments/smallTick/"+ric+"_EDA_SpreadTOD.png")
        plt.show()
        with open("/SAN/fca/Konark_PhD_Experiments/smallTick/"+ric+"_EDA_SpreadTOD", "wb") as f:
            pickle.dump(avg, f)

        # In[138]:


        np.average(np.average(100*np.array([s for s in spreads if len(s) == 390]), axis = 0))


        # In[157]:


        plt.figure()
        a = plt.hist(np.ravel(100*np.array([s for s in spreads if len(s) == 390])), bins = 1000, density = True, color="mediumpurple")
        plt.xlabel("Ratio")
        plt.xlabel("Spread in ticks")
        plt.yscale("log")
        plt.text(a[1][-1]*0.75, np.max(a[0]), "Mean = " + str(np.round(np.average(np.average(100*np.array([s for s in spreads if len(s) == 390]), axis = 0)),2)), bbox=dict(alpha=0.5))

        plt.title("Spread density (log) : " + ric)
        plt.savefig("/SAN/fca/Konark_PhD_Experiments/smallTick/"+ric+"_EDA_SpreadDistriLog.png")
        with open("/SAN/fca/Konark_PhD_Experiments/smallTick/"+ric+"_EDA_SpreadDistri", "wb") as f:
            pickle.dump(avg, f)

        # In[156]:


        plt.figure()
        a =plt.hist(np.ravel(100*np.array([s for s in spreads if len(s) == 390])), bins = 1000, density = True, color="lightcoral")
        plt.xlabel("Ratio")
        plt.xlabel("Spread in ticks")
        plt.text(a[1][-1]*0.75, np.max(a[0]), "Mean = " + str(np.round(np.average(np.average(100*np.array([s for s in spreads if len(s) == 390]), axis = 0)),2)), bbox=dict(alpha=0.5))
        plt.title("Spread density : " + ric)
        plt.savefig("/SAN/fca/Konark_PhD_Experiments/smallTick/"+ric+"_EDA_SpreadDistri.png")



    # In[4]:


    import gc
    gc.collect()




    if edashape:
        # # Shape of the book

        # In[80]:


        # prices = np.arange(0,100)
        final_res = {}
        final_res_far = {}
        for j in pd.date_range(dt.date(2019,1,2), dt.date(2019,12,31)):
            if j == dt.date(2019,1,9): continue
            l = dataLoader.Loader(ric, j, j, nlevels = 10, dataPath = "/SAN/fca/Konark_PhD_Experiments/extracted/GOOG/")
            data = l.load()
            master_dict, master_dict_far = {}, {}
            if len(data):
                data = data[0]
            else:
                continue
            data['timeDiff'] = data['Time'].diff()
            data['total v_a'] = data[["Ask Size " + str(i) for i in range(1,11) ]].sum(axis=1)
            data['total v_b'] = data[["Bid Size " + str(i) for i in range(1,11) ]].sum(axis=1)
            data['mid'] = (data['Ask Price 1'] + data['Bid Price 1'])*0.5
            for i in range(10,0, -1):
                data["Ask Size " + str(i)] = data["Ask Size " + str(i)]/data['total v_a']
                data["Bid Size " + str(i)] = data["Ask Size " + str(i)]/data['total v_b']
                data["Ask Price1 " + str(i)] = np.round(200*(data["Ask Price " + str(i)] - data['mid'])).astype(int)
                data["Bid Price1 " + str(i)] = np.round(200*(data['mid'] - data["Bid Price " + str(i)])).astype(int)
                data["Ask Price2 " + str(i)] = np.round(200*(data["Ask Price " + str(i)] - data["Bid Price 1"])).astype(int)
                data["Bid Price2 " + str(i)] = np.round(200*(data['Ask Price 1'] - data["Bid Price " + str(i)])).astype(int)
                data['tmp'] = data['Ask Size ' + str(i)]*data['timeDiff']
                data_dict = (data.groupby("Ask Price1 " + str(i))['tmp'].sum()/23400).to_dict()
                data_dict_far = (data.groupby("Ask Price2 " + str(i))['tmp'].sum()/23400).to_dict()
                # print(sum(data_dict.values()))
                for k, v in data_dict.items():
                    master_dict[k] = master_dict.get(k, [])+ [v]
                for k, v in data_dict_far.items():
                    master_dict_far[k] = master_dict_far.get(k, [])+ [v]
                data['tmp'] = data['Bid Size ' + str(i)]*data['timeDiff']
                data_dict = (data.groupby("Bid Price1 " + str(i))['tmp'].sum()/23400).to_dict()
                data_dict_far = (data.groupby("Bid Price2 " + str(i))['tmp'].sum()/23400).to_dict()
                for k, v in data_dict.items():
                    master_dict[k] = master_dict.get(k, [])+ [v]
                for k, v in data_dict_far.items():
                    master_dict_far[k] = master_dict_far.get(k, [])+ [v]
            res_dict = {}
            std_dict = {}
            for k,v in master_dict.items():
                res_dict[k] = np.average(v)
                std_dict[k] = np.std(v)
                final_res[k] = final_res.get(k, [])+ [res_dict[k]]
            res_dict_far = {}
            std_dict_far = {}
            for k,v in master_dict_far.items():
                res_dict_far[k] = np.average(v)
                std_dict_far[k] = np.std(v)
                final_res_far[k] = final_res_far.get(k, [])+ [res_dict_far[k]]
        # In[81]:


        #get_ipython().run_line_magic('matplotlib', 'inline')
        plt.figure()
        # plt.plot(np.arange(0,100,2)/2, [(res_dict[k] + res_dict[k+1])/2 for k in range(0,100,2)])
        key0 = list(final_res.keys())[0]
        for i in range(len(final_res[key0])):
            x, y = [], []
            for k in range(1,250):
                if (final_res.get(k, None) is not None)and(len(final_res.get(k, [])) == len(final_res[key0])):
                    x+= [k]
                    y+= [final_res[k][i]]
            x = np.array(x)
            if ric in ['SIRI','BAC', 'INTC','CSCO','ORCL','MSFT']:
                x = x[0:-1:2]
                x = [i for i in x if i <= 20]
                y= [final_res[k][i] for k in x]
            elif ric in ['AAPL','ABBV', 'PM','IBM']:
                x = x[0:-1:2]
                x = [i for i in x if i <= 20]
                y= [final_res[k][i] for k in x]
            x = np.array(x)
            plt.plot(x/2, y, alpha = 0.1, color = "steelblue")
        plt.plot(x/2, [np.average(final_res[k]) for k in x], color="r")
        plt.xlabel("Depth from mid (in ticks)")
        plt.ylabel("Volume ratio")
        plt.legend()
        plt.title("Shape of LOB - " + ric)
        plt.savefig("/SAN/fca/Konark_PhD_Experiments/smallTick/"+ric+"_EDA_Shape.png")
        with open("/SAN/fca/Konark_PhD_Experiments/smallTick/"+ric+"_EDA_Shape", "wb") as f:
            pickle.dump(final_res, f)
        with open("/SAN/fca/Konark_PhD_Experiments/smallTick/"+ric+"_EDA_ShapeFT", "wb") as f:
            pickle.dump(final_res_far, f)


        # In[ ]:


        # plt.figure()
        # # plt.plot(np.arange(0,100,2)/2, [(res_dict[k] + res_dict[k+1])/2 for k in range(0,100,2)])
        # for i in range(len(final_res_far[50])-1):
        #     plt.plot(np.arange(2,250,2)/2, [final_res_far[k][i] for k in range(2,250,2)], alpha = 0.1, color = "steelblue")
        # plt.plot(np.arange(2,250,2)/2, [np.average(final_res_far[k]) for k in range(2,250,2)], color="r")
        # # plt.fill_between(np.arange(0,100)/2, [np.max([1e-3,res_dict[k] - std_dict[k]]) for k in range(100)], [res_dict[k] + std_dict[k] for k in range(100)], alpha = 0.1, color="b")
        # # plt.yscale("log")
        # plt.xlabel("Depth from far touch (in ticks)")
        # plt.ylabel("Volume ratio")
        # plt.legend()
        # plt.title("Shape of LOB - " + ric)
        # plt.savefig("/SAN/fca/Konark_PhD_Experiments/smallTick/"+ric+"_EDA_ShapeFT.png")


        # In[ ]:


        plt.figure()
        pctiles = {}
        percentiles = [0.1,0.25, 0.5, 0.75, 0.9, 0.99]
        for i in range(len(final_res[key0])):
            x, y = [], []
            for k in range(1,250):
                if (final_res.get(k, None) is not None)and(len(final_res.get(k, [])) == len(final_res[key0])):
                    x+= [k]
                    y+= [final_res[k][i]]
            x = np.array(x)
            vols = np.array(y)
            volsNorm = vols/np.sum(vols)
            cumVols = np.cumsum(volsNorm)

            ticks = x/2
            for p in percentiles:
                idx = np.where(cumVols >= p)[0][0]
                # print("Percentile " + str(100*p) + " at " + str(ticks[idx]) + " ticks from mid")
                pctiles[p] = pctiles.get(p, []) + [ticks[idx]]
        # print("Perc Mean Std")
        # for p in percentiles:
        #     print(100*p, np.round(np.mean(pctiles[p]),1), np.round(np.std(pctiles[p]),2))
        plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab10.colors)

        for p in percentiles[:-1]:
            plt.plot(pctiles[p],  label=str(100*p) + "th percentile : " + str(np.round(np.mean(pctiles[p]),1)) +" +- " + str(np.round(np.std(pctiles[p]),2)), alpha = 0.5)
            # plt.text(0, 10*p, )
        plt.title("Shape of Book Percentiles Stationarity : "+ ric)
        plt.legend()
        plt.xlabel("Days")
        plt.ylabel("Depth in ticks")
        plt.savefig("/SAN/fca/Konark_PhD_Experiments/smallTick/"+ric+"_EDA_ShapeTimeSeries.png")
        with open("/SAN/fca/Konark_PhD_Experiments/smallTick/"+ric+"_EDA_ShapePctiles", "wb") as f:
            pickle.dump(pctiles, f)

    if edasparse:
        # # Sparse Book

        # In[ ]:


        res, res_distr, wts_distr = {}, {}, {}
        for j in pd.date_range(dt.date(2019,1,2), dt.date(2019,5,31)):
            if j == dt.date(2019,1,9): continue
            l = dataLoader.Loader(ric, j, j, nlevels = 10, dataPath = "/SAN/fca/Konark_PhD_Experiments/extracted/GOOG/")
            data = l.load()
            if len(data):
                data = data[0]
            else:
                continue
            data['timeDiff'] = data['Time'].diff()
            for i in range(2,11):
                data['tmp'] = data["Ask Price " + str(i)] - data["Ask Price " + str(i-1)] - 0.01
                res[i-1] = res.get(i-1, []) + [np.sum(data['tmp']*data['timeDiff'])/23400]
                res_distr[i-1] = np.append(res_distr.get(i-1, []), data['tmp'])
                wts_distr[i-1] = np.append(wts_distr.get(i-1, []), data['timeDiff'])
                data['tmp'] = data["Bid Price " + str(i)] - data["Bid Price " + str(i-1)] + 0.01
                res[i-1] = res.get(i-1, []) + [np.abs(np.sum(data['tmp']*data['timeDiff'])/23400)]
                res_distr[i-1] = np.append(res_distr.get(i-1, []), np.abs(data['tmp']))
                wts_distr[i-1] = np.append(wts_distr.get(i-1, []), data['timeDiff'])


        # In[ ]:


        print(ric)
        for i in range(1,10):
            print("Avg number of empty levels between top + " + str(i-1) + " to top + " + str(i) +" " + str(np.round(100*np.mean(res[i]),2)) + " +- " + str(np.round(100*np.std(res[i]),2)))


        # In[ ]:


        plt.figure(figsize=(20,10))
        plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab10.colors)
        cycler = itertools.cycle(plt.rcParams['axes.prop_cycle'])
        hists = {}
        for i in range(1,10):
            a = plt.hist(100*res_distr[i], weights = np.nan_to_num(wts_distr[i]) , bins = np.arange(25), density =True, label ="Top + " +str(i-1) + " to Top + " + str(i)+ " Median : " + str(np.round(100*np.median(res[i]),2))+" Mean : " + str(np.round(100*np.mean(res[i]),2)) + " +- " + str(np.round(100*np.std(res[i]),2)), histtype=u'step')
            hists[i] = a
        # plt.ylabel("Ratio")
        plt.xlabel("Number of Empty levels")
        plt.legend()
        # plt.xscale("log")
        # plt.yscale("log")
        # plt.text(150, 0.025, "Mean = " + str(np.round(np.average(np.average(100*np.array([s for s in spreads if len(s) == 390]), axis = 0)),2)), bbox=dict(alpha=0.5))
        plt.title("Empty Levels density : " + ric)
        plt.savefig("/SAN/fca/Konark_PhD_Experiments/smallTick/"+ric+"_EDA_SparsityDistri.png")
        with open("/SAN/fca/Konark_PhD_Experiments/smallTick/"+ric+"_EDA_SparsityDistri", "wb") as f:
            pickle.dump(hists, f)


        # In[ ]:


        plt.figure(figsize=(20,10))
        plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab10.colors)
        cycler = itertools.cycle(plt.rcParams['axes.prop_cycle'])
        for i in range(1,10):
            plt.hist(100*res_distr[i], weights = np.nan_to_num(wts_distr[i]), bins = np.arange(200), density = True, label ="Top + " +str(i-1) + " to Top + " + str(i)+ " Median : " + str(np.round(100*np.median(res[i]),2))+" Mean : " + str(np.round(100*np.mean(res[i]),2)) + " +- " + str(np.round(100*np.std(res[i]),2)), histtype=u'step')
        # plt.ylabel("Ratio")
        plt.xlabel("Number of Empty levels")
        plt.legend()
        # plt.xscale("log")
        plt.yscale("log")
        # plt.text(150, 0.025, "Mean = " + str(np.round(np.average(np.average(100*np.array([s for s in spreads if len(s) == 390]), axis = 0)),2)), bbox=dict(alpha=0.5))
        plt.title("Empty Levels density : " + ric)
        plt.savefig("/SAN/fca/Konark_PhD_Experiments/smallTick/"+ric+"_EDA_SparsityDistriLog.png")


        # In[ ]:


        plt.figure()
        plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab10.colors)
        for i in range(1,10):
            plt.plot(res[i], alpha = 0.5, label ="Top + " +str(i-1) + " to Top + " + str(i)+" Mean : " + str(np.round(100*np.mean(res[i]),2)) + " +- " + str(np.round(100*np.std(res[i]),2)))
        plt.xlabel("Days")
        plt.ylabel("Number of Empty levels")
        # plt.text(150, 0.025, "Mean = " + str(np.round(np.average(np.average(100*np.array([s for s in spreads if len(s) == 390]), axis = 0)),2)), bbox=dict(alpha=0.5))
        plt.title("Empty Levels TimeSeries : " + ric)
        plt.legend()
        plt.savefig("/SAN/fca/Konark_PhD_Experiments/smallTick/"+ric+"_EDA_SparsityTimeSeries.png")
        with open("/SAN/fca/Konark_PhD_Experiments/smallTick/"+ric+"_EDA_SparsityTimeSeries", "wb") as f:
            pickle.dump(res, f)

    if edarest:
        # # Volume at best vs ADV

        # In[ ]:


        bests, advs = [], []
        for j in pd.date_range(dt.date(2019,1,2), dt.date(2019,12,31)):
            if j == dt.date(2019,1,9): continue
            l = dataLoader.Loader(ric, j, j, nlevels = 10, dataPath = "/SAN/fca/Konark_PhD_Experiments/extracted/GOOG/")
            data = l.load()
            if len(data):
                data = data[0]
            else:
                continue
            data['timeDiff'] = data['Time'].diff()
            adv = data.loc[data['Type'] == 4]['Size'].sum()
            best = ((data['Ask Size 1']+data['Bid Size 1'])*data['timeDiff']).sum()
            bests.append(best)
            advs.append(adv)


        # In[ ]:


        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        # ax1.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab10.colors)
        ax1.plot(np.array(bests)/2, alpha = 0.5, label ="Best Volume (both sides)")
        ax1.plot(advs, alpha = 0.5, label = "Daily Traded Volume")
        ax2.plot(np.array(bests)/(2*np.array(advs)), color = "r", label = "Volume at Best/ADV")
        ax2.axhline(np.array(bests).sum()/(2*np.array(advs).sum()), 0, len(bests), ls = "-.", color="r",  label ="Average")
        ax2.set_ylabel("Ratio", color="r")
        ax1.set_xlabel("Days")
        ax1.set_ylabel("Shares")
        # plt.text(150, 0.025, "Mean = " + str(np.round(np.average(np.average(100*np.array([s for s in spreads if len(s) == 390]), axis = 0)),2)), bbox=dict(alpha=0.5))
        ax1.set_title("Best Volume/ ADV TimeSeries : " + ric)
        ax1.legend()
        ax2.legend()
        plt.savefig("/SAN/fca/Konark_PhD_Experiments/smallTick/"+ric+"_EDA_BestVolumeADVTimeSeries.png")
        with open("/SAN/fca/Konark_PhD_Experiments/smallTick/"+ric+"_EDA_BestVolumeADVTimeSeries", "wb") as f:
            pickle.dump((np.array(bests)/2, advs), f)


        # # MOs vs number and levels of queue depletions

        # In[ ]:


        ratios_day = []
        ratios , wts = [], []
        for j in pd.date_range(dt.date(2019,1,2), dt.date(2019,12,31)):
            if j == dt.date(2019,1,9): continue
            l = dataLoader.Loader(ric, j, j, nlevels = 10, dataPath = "/SAN/fca/Konark_PhD_Experiments/extracted/GOOG/")
            data = l.load()
            if len(data):
                data = data[0]
            else:
                continue
            data['timeDiff'] = data['Time'].diff()
            data_trade = data.loc[data['Type'] == 4]
            trades = data_trade['Size']
            best = data_trade['Ask Size 1']*(1 - data_trade['TradeDirection'])/2 +data_trade['Bid Size 1']*(data_trade['TradeDirection'] + 1)/2
            ratios_day.append(((trades/best)*data_trade['timeDiff']).sum()/23400)
            ratios = np.append(ratios, (trades/best).values)
            wts = np.append(wts, data_trade['timeDiff'].values)


        # In[ ]:


        plt.figure()
        plt.hist(ratios, weights = np.nan_to_num(wts), bins = np.power(10, np.linspace(-3,3, num  = 100)),  density = True, color="mediumpurple")
        plt.xlabel("Ratio")
        plt.xlabel("Ratio of MO size vs Queue Size")
        plt.text(1e2, 0.5, "Mean = " + str(np.round(np.average(ratios_day),2)), bbox=dict(alpha=0.5))
        plt.text(1e2, 15, "Median = " + str(np.round(np.median(ratios),2)), bbox=dict(alpha=0.5))
        plt.xscale("log")
        plt.yscale("log")
        plt.title("MO size vs Queue Size density : " + ric)
        plt.savefig("/SAN/fca/Konark_PhD_Experiments/smallTick/"+ric+"_EDA_MORatioDistri.png")
        with open("/SAN/fca/Konark_PhD_Experiments/smallTick/"+ric+"_EDA_MORatioDistri", "wb") as f:
            pickle.dump(ratios, f)

        # # Arrival rates wrt mid

        # In[8]:


        final_res= {}
        for j in pd.date_range(dt.date(2019,1,2), dt.date(2019,12,31)):
            if j == dt.date(2019,1,9): continue
            l = dataLoader.Loader(ric, j, j, nlevels = 10, dataPath = "/SAN/fca/Konark_PhD_Experiments/extracted/GOOG/")
            data = l.load()
            master_dict = {}
            if len(data):
                data = data[0]
            else:
                continue
            data['timeDiff'] = data['Time'].diff()
            data['mid'] = (data['Ask Price 1'] + data['Bid Price 1'])*0.5
            for i in range(10,0, -1):
                data["Ask Price1 " + str(i)] = np.round(200*(data["Ask Price " + str(i)] - data['mid'])).astype(int)
                data["Bid Price1 " + str(i)] = np.round(200*(data['mid'] - data["Bid Price " + str(i)])).astype(int)
                data_dict = (data.groupby("Ask Price1 " + str(i))['Time'].count()).to_dict()
                # print(sum(data_dict.values()))
                for k, v in data_dict.items():
                    master_dict[k] = master_dict.get(k, [])+ [v]
                data_dict = (data.groupby("Bid Price1 " + str(i))['Time'].count()).to_dict()
                for k, v in data_dict.items():
                    master_dict[k] = master_dict.get(k, [])+ [v]
            for k,v in master_dict.items():
                final_res[k] = final_res.get(k, [])+ [np.sum(master_dict[k])/23400]
        for k,v in final_res.items():
            final_res[k] = np.mean(v)


        # In[10]:


        for k,v in final_res.items():
            final_res[k] = np.mean(v)


        # In[21]:


        #get_ipython().run_line_magic('matplotlib', 'inline')
        plt.figure()
        plt.scatter([i/2 for i in list(final_res.keys())], list(final_res.values()), s= 5, c= "lightcoral")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Distance from mid (ticks)")
        plt.ylabel("Arrival Rate")
        plt.title("Arrival Rate vs distance from mid : " + ric)
        plt.savefig("/SAN/fca/Konark_PhD_Experiments/smallTick/"+ric+"_EDA_ArrivalRateVsDistanceFromMid.png")
        with open("/SAN/fca/Konark_PhD_Experiments/smallTick/"+ric+"_EDA_MORatioDistri", "wb") as f:
            pickle.dump(([i/2 for i in list(final_res.keys())], list(final_res.values())), f)
    if edaqd:

        # In[ ]:
        wts , chgs = [], []
        for j in pd.date_range(dt.date(2019,1,2), dt.date(2019,12,31)):
            if j == dt.date(2019,1,9): continue
            l = dataLoader.Loader(ric, j, j, nlevels = 10, dataPath = "/SAN/fca/Konark_PhD_Experiments/extracted/GOOG/")
            data = l.load()
            if len(data):
                data = data[0]
            else:
                continue
            data['timeDiff'] = data['Time'].diff()
            data['Ask Price change' ] = data['Ask Price 1'] - data['Ask Price 1'].shift(1)
            data['Bid Price change' ] = data['Bid Price 1'] - data['Bid Price 1'].shift(1)
            data_trade = data.loc[data['Type'] == 4]
            wts = np.append(wts, data_trade.timeDiff.values)
            chgs = np.append(chgs, (data_trade['Ask Price change'] + data_trade['Bid Price change']).values)
        with open("/SAN/fca/Konark_PhD_Experiments/smallTick/"+ric+"_EDA_MOQDs", "wb") as f:
            pickle.dump((wts, chgs), f)

    if edashapemaxima:
        dict_res = {}
        for j in pd.date_range(dt.date(2019,1,2), dt.date(2019,12,31)):
            if j == dt.date(2019,1,9): continue
            l = dataLoader.Loader(ric, j, j, nlevels = 10, dataPath = "/SAN/fca/Konark_PhD_Experiments/extracted/GOOG/")
            data = l.load()
            if len(data):
                data = data[0]
            else:
                continue

            data['mid'] = (data['Ask Price 1'] + data['Bid Price 1'])*0.5
            for i in range(10,0, -1):
                data["Ask Price1 " + str(i)] = np.round(200*(data["Ask Price " + str(i)] - data['mid'])).astype(int)
                data["Bid Price1 " + str(i)] = np.round(200*(data['mid'] - data["Bid Price " + str(i)])).astype(int)

            maxidxs= np.argmax(data[["Ask Size " + str(i) for i in range(1,11)]].values, axis = 1)
            maxdepths = data[["Ask Price1 " + str(i) for i in range(1,11)]].values[np.arange(len(data)), maxidxs]
            maxidxs= np.argmax(data[["Bid Size " + str(i) for i in range(1,11)]].values, axis = 1)
            maxdepths = np.append(maxdepths, data[["Bid Price1 " + str(i) for i in range(1,11)]].values[np.arange(len(data)), maxidxs])
            counts = np.unique(maxdepths, return_counts=True)
            for k, v in zip(counts[0], counts[1]):
                dict_res[k] = dict_res.get(k, 0) + v
        with open("/SAN/fca/Konark_PhD_Experiments/smallTick/"+ric+"_EDA_shapeMaxima", "wb") as f:
            pickle.dump(dict_res, f)


    if edashapesparsity:
        #avg shape calc

        s=ric
        with open("/SAN/fca/Konark_PhD_Experiments/smallTick/"+s+"_EDA_Shape", "rb") as f:
            data = pickle.load(f)
        key0 = list(data.keys())[0]
        x, y = [], []
        for k in range(1,250):
            if (data.get(k, None) is not None)and(len(data.get(k, [])) == len(data[key0])):
                x+= [k]
        x = np.array(x)
        if s in ['SIRI','BAC', 'INTC','CSCO','ORCL','MSFT']:
            x = x[0:-1:2]
            # print(x)
            x = [i for i in x if i <= 20]
        elif s in ['AAPL','ABBV', 'PM','IBM']:
            x = x[0:-1:2]
            x = [i for i in x if i <= 20]
        x = np.array(x)
        avgShapeLvls, avgShapeVols = x, [np.average(data[k]) for k in x]/(np.sum([np.average(data[k]) for k in x]))
        avgShape = np.zeros(int(max(avgShapeLvls))+1)
        avgShape[avgShapeLvls.astype(int)] = avgShapeVols
        #insttshape calc
        wassDistances_mean, wassDistances_var = [], []
        for j in pd.date_range(dt.date(2019,1,2), dt.date(2019,12,31)):
            if j == dt.date(2019,1,9): continue
            l = dataLoader.Loader(ric, j, j, nlevels = 10, dataPath = "/SAN/fca/Konark_PhD_Experiments/extracted/GOOG/")
            data = l.load()
            if len(data):
                data = data[0]
            else:
                continue
            data['timeDiff'] = data['Time'].diff()
            data['mid'] = (data['Ask Price 1'] + data['Bid Price 1'])*0.5
            data['total v_a'] = data[["Ask Size " + str(i) for i in range(1,11) ]].sum(axis=1)
            data['total v_b'] = data[["Bid Size " + str(i) for i in range(1,11) ]].sum(axis=1)
            data['mid'] = (data['Ask Price 1'] + data['Bid Price 1'])*0.5
            for i in range(10,0, -1):
                data["Ask Size " + str(i)] = data["Ask Size " + str(i)]/data['total v_a']
                data["Bid Size " + str(i)] = data["Ask Size " + str(i)]/data['total v_b']
                data["Ask Price1 " + str(i)] = np.round(200*(data["Ask Price " + str(i)] - data['mid'])).astype(int)
                data["Bid Price1 " + str(i)] = np.round(200*(data['mid'] - data["Bid Price " + str(i)])).astype(int)
            insttLvlsAsk = data[["Ask Price1 " + str(i) for i in range(1,11)]].values
            insttVolsAsk = data[["Ask Size " + str(i) for i in range(1,11)]].values
            wassDistancesAsk = []
            for insttLvl, insttVol in zip(insttLvlsAsk, insttVolsAsk):
                insttShape = np.zeros(int(max(avgShapeLvls))+1)

                insttVol = insttVol[insttLvl <= max(avgShapeLvls)]
                insttLvl = insttLvl[insttLvl <= max(avgShapeLvls)]
                insttShape[insttLvl] = insttVol
                wassDistancesAsk.append(np.abs(avgShape - insttShape).sum())
            # print(wassDistancesAsk)
            mean_wassDistancesAsk = np.nansum(data['timeDiff'].values*np.array(wassDistancesAsk))/23400
            var_wassDistancesAsk = np.average((np.array(wassDistancesAsk)-mean_wassDistancesAsk)**2, weights=np.nan_to_num(data['timeDiff'].values))
            insttLvlsBid = data[["Bid Price1 " + str(i) for i in range(1,11)]].values
            insttVolsBid = data[["Bid Size " + str(i) for i in range(1,11)]].values
            wassDistancesBid = []
            for insttLvl, insttVol in zip(insttLvlsBid, insttVolsBid):
                insttShape = np.zeros(int(max(avgShapeLvls))+1)

                insttVol = insttVol[insttLvl <= max(avgShapeLvls)]
                insttLvl = insttLvl[insttLvl <= max(avgShapeLvls)]
                insttShape[insttLvl] = insttVol
                wassDistancesBid.append(np.abs(avgShape - insttShape).sum())
            mean_wassDistancesBid = np.nansum(data['timeDiff'].values*np.array(wassDistancesBid))/23400
            var_wassDistancesBid = np.average((np.array(wassDistancesBid)-mean_wassDistancesBid)**2, weights=np.nan_to_num(data['timeDiff'].values))
            print(mean_wassDistancesAsk, var_wassDistancesAsk)
            print(mean_wassDistancesBid, var_wassDistancesBid)
            wassDistances_mean.append(mean_wassDistancesAsk)
            wassDistances_mean.append(mean_wassDistancesBid)
            wassDistances_var.append(var_wassDistancesAsk)
            wassDistances_var.append(var_wassDistancesBid)
        with open("/SAN/fca/Konark_PhD_Experiments/smallTick/"+ric+"_EDA_shapeWasserstein", "wb") as f:
            pickle.dump((wassDistances_mean,wassDistances_var), f)

    if edaleverage:
        orderTypeDict = {'limit' : [1], 'cancel': [2,3], 'market' : [4]}
        condCounts = []
        uncondCounts= []
        for j in pd.date_range(dt.date(2019,1,2), dt.date(2019,12,31)):
            condProb = []
            if j == dt.date(2019,1,9): continue
            l = dataLoader.Loader(ric, j, j, nlevels = 10, dataPath = "/SAN/fca/Konark_PhD_Experiments/extracted/GOOG/")
            data = l.load()
            if len(data):
                data = data[0]
            else:
                continue
            # events wrt distance from mid in ticks
            data = data.loc[data['Type'] < 5]
            data = data.loc[data['Type'] !=2]
            data['mid'] = (data['Ask Price 1'] + data['Bid Price 1'])*0.5
            data['midPrev'] = data.mid.shift(1).fillna(0)
            data['depth'] = np.round(100*(data.Price/10000 - data.midPrev), 2)
            data['depthAbs'] = data['depth'].apply(lambda x: np.abs(x))
            converter ={ 1: 0, 3: 1, 4:2}
            depthMax = int(np.max(data['depthAbs']))
            # matrix = np.zeros((int(depthMax*3*2), int(depthMax*3*2)))
            data['TypeDepth'] = data['Type'].astype(int).astype(str) +  data['depth'].astype(int).astype(str)
            data['TypeDepth_1'] = data['TypeDepth'].shift(1)
            # data.head()
            if len(condCounts)==0:
                condCounts = data.groupby(['TypeDepth','TypeDepth_1'])['Time'].count()
                uncondCounts = data.groupby(['TypeDepth'])['Time'].count()
            else:
                tmp = data.groupby(['TypeDepth','TypeDepth_1'])['Time'].count()
                # new_idx = condCounts.index.union(tmp.index)
                # condCounts = condCounts.loc[new_idx].fillna(0) + tmp.loc[new_idx].fillna(0)
                condCounts = condCounts.add(tmp, fill_value=0)
                tmp= data.groupby(['TypeDepth'])['Time'].count()
                # new_idx = uncondCounts.index.union(tmp.index)
                # uncondCounts = uncondCounts.loc[new_idx].fillna(0) + tmp.loc[new_idx].fillna(0)
                uncondCounts = uncondCounts.add(tmp, fill_value=0)
        condProb = condCounts/uncondCounts
        uncondProb = uncondCounts/uncondCounts.sum()
        leverage = condProb/uncondProb
        leverage = leverage.reset_index()
        with open("/SAN/fca/Konark_PhD_Experiments/smallTick/"+ric+"_EDA_leverage", "wb") as f:
            pickle.dump({'condCounts' : condCounts,'uncondCounts' :  uncondCounts , 'condProb' : condProb,'uncondProb' : uncondProb, 'leverage': leverage}, f)

    if edaleverage_top:
        orderTypeDict = {'limit' : [1], 'cancel': [2,3], 'market' : [4]}
        condCounts = []
        uncondCounts= []
        for j in pd.date_range(dt.date(2019,1,2), dt.date(2019,12,31)):
            condProb = []
            if j == dt.date(2019,1,9): continue
            l = dataLoader.Loader(ric, j, j, nlevels = 10, dataPath = "/SAN/fca/Konark_PhD_Experiments/extracted/GOOG/")
            data = l.load()
            if len(data):
                data = data[0]
            else:
                continue
            # events wrt distance from mid in ticks
            data = data.loc[data['Type'] < 5]
            data = data.loc[data['Type'] !=2]
            dataOrig = data.copy()
            for d, side in zip([-1,1],['Ask', 'Bid']):
                data = dataOrig.loc[dataOrig['TradeDirection'] == d]
                # data['mid'] = (data['Ask Price 1'] + data['Bid Price 1'])*0.5
                data['topPrev'] = data[side+' Price 1'].shift(1).fillna(0)
                data['depth'] = np.round(100*d*(data.Price/10000 - data.topPrev), 2)
                data['depthAbs'] = data['depth'].apply(lambda x: np.abs(x))
                converter ={ 1: 0, 3: 1, 4:2}
                depthMax = int(np.max(data['depthAbs']))
                # matrix = np.zeros((int(depthMax*3*2), int(depthMax*3*2)))
                data['TypeDepth'] = data['Type'].astype(int).astype(str) +  data['depth'].astype(int).astype(str)
                data['TypeDepth_1'] = data['TypeDepth'].shift(1)
                # data.head()
                if len(condCounts)==0:
                    condCounts = data.groupby(['TypeDepth','TypeDepth_1'])['Time'].count()
                    uncondCounts = data.groupby(['TypeDepth'])['Time'].count()
                else:
                    tmp = data.groupby(['TypeDepth','TypeDepth_1'])['Time'].count()
                    # new_idx = condCounts.index.union(tmp.index)
                    # condCounts = condCounts.loc[new_idx].fillna(0) + tmp.loc[new_idx].fillna(0)
                    condCounts = condCounts.add(tmp, fill_value=0)
                    tmp= data.groupby(['TypeDepth'])['Time'].count()
                    # new_idx = uncondCounts.index.union(tmp.index)
                    # uncondCounts = uncondCounts.loc[new_idx].fillna(0) + tmp.loc[new_idx].fillna(0)
                    uncondCounts = uncondCounts.add(tmp, fill_value=0)

        condProb = condCounts/uncondCounts
        uncondProb = uncondCounts/uncondCounts.sum()
        leverage = condProb/uncondProb
        leverage = leverage.reset_index()
        with open("/SAN/fca/Konark_PhD_Experiments/smallTick/"+ric+"_EDA_leverageOneSidedTop", "wb") as f:
            pickle.dump({'condCounts' : condCounts,'uncondCounts' :  uncondCounts , 'condProb' : condProb,'uncondProb' : uncondProb, 'leverage': leverage}, f)

    if edaleverageIS:
        orderTypeDict = {'limit' : [1], 'cancel': [2,3], 'market' : [4]}
        condCounts = []
        uncondCounts= []
        for j in pd.date_range(dt.date(2019,1,2), dt.date(2019,12,31)):
            condProb = []
            if j == dt.date(2019,1,9): continue
            l = dataLoader.Loader(ric, j, j, nlevels = 10, dataPath = "/SAN/fca/Konark_PhD_Experiments/extracted/GOOG/")
            data = l.load()
            if len(data):
                data = data[0]
            else:
                continue
            # events wrt distance from mid in ticks
            data = data.loc[data['Type'] < 5]
            data = data.loc[data['Type'] !=2]
            dataOrig = data.copy()
            for d, side in zip([-1,1],['Ask', 'Bid']):
                data = dataOrig.loc[dataOrig['TradeDirection'] == d]
                # data['mid'] = (data['Ask Price 1'] + data['Bid Price 1'])*0.5
                data['topPrev'] = data[side+' Price 1'].shift(1).fillna(0)
                data['depth'] = np.round(100*(data.Price/10000 - data.topPrev), 2)
                data['depthAbs'] = data['depth'].apply(lambda x: np.abs(x))
                data['is'] = 0
                data['diff'] = data['Ask Price 1'].shift(1) - data['Ask Price 1']
                data['is'].loc[data['diff'] > 0]  = 1
                data['diff'] = data['Bid Price 1'] - data['Bid Price 1'].shift(1)
                data['is'].loc[data['diff'] > 0]  = 1
                converter ={ 1: 0, 3: 1, 4:2}
                depthMax = int(np.max(data['depthAbs']))
                # matrix = np.zeros((int(depthMax*3*2), int(depthMax*3*2)))
                data['TypeDepth'] = data['Type'].astype(int).astype(str) +  data['is'].astype(int).astype(str) +  data['depth'].astype(int).astype(str)
                data['TypeDepth_1'] = data['TypeDepth'].shift(1)
                # data.head()
                if len(condCounts)==0:
                    condCounts = data.groupby(['TypeDepth','TypeDepth_1'])['Time'].count()
                    uncondCounts = data.groupby(['TypeDepth'])['Time'].count()
                else:
                    tmp = data.groupby(['TypeDepth','TypeDepth_1'])['Time'].count()
                    # new_idx = condCounts.index.union(tmp.index)
                    # condCounts = condCounts.loc[new_idx].fillna(0) + tmp.loc[new_idx].fillna(0)[
                    condCounts = condCounts.add(tmp, fill_value=0)
                    tmp= data.groupby(['TypeDepth'])['Time'].count()
                    # new_idx = uncondCounts.index.union(tmp.index)
                    # uncondCounts = uncondCounts.loc[new_idx].fillna(0) + tmp.loc[new_idx].fillna(0)
                    uncondCounts = uncondCounts.add(tmp, fill_value=0)
        condProb = condCounts/uncondCounts
        uncondProb = uncondCounts/uncondCounts.sum()
        leverage = condProb/uncondProb
        leverage = leverage.reset_index()
        with open("/SAN/fca/Konark_PhD_Experiments/smallTick/"+ric+"_EDA_leverageIS_truetop", "wb") as f:
            pickle.dump({'condCounts' : condCounts,'uncondCounts' :  uncondCounts , 'condProb' : condProb,'uncondProb' : uncondProb, 'leverage': leverage}, f)

    if assumptions:
        orderTypeDict = {'limit' : [1], 'cancel': [2,3], 'market' : [4]}
        condCounts_mT, condCounts_mD, condCounts_q_LO, condCounts_q_MO, condCounts_eta_IS = [], [], [] , [], []
        uncondCounts_mT, uncondCounts_mD, uncondCounts_q_LO, uncondCounts_q_MO, uncondCounts_eta_IS = [], [], [], [] ,[]
        for j in pd.date_range(dt.date(2019,1,2), dt.date(2019,12,31)):
            if j == dt.date(2019,1,9): continue
            l = dataLoader.Loader(ric, j, j, nlevels = 10, dataPath = "/SAN/fca/Konark_PhD_Experiments/extracted/GOOG/")
            data = l.load()
            if len(data):
                data = data[0]
            else:
                continue
            # events wrt distance from mid in ticks
            data = data.loc[data['Type'] < 5]
            data = data.loc[data['Type'] !=2]
            data['sec'] = data['Time'].astype(int)
            intensityPerSec = data.groupby(['sec','Type','TradeDirection'])['Time'].count()
            data['q_LO'] = np.nan
            data['q_LO'].loc[data['Type'] == 1] = data['Size'].loc[data['Type'] == 1]
            data['q_MO'] = np.nan
            data['q_MO'].loc[data['Type'] == 4] = data['Size'].loc[data['Type'] == 1]
            #
            data['is'] = 0
            data['diff'] = data['Ask Price 1'].shift(1) - data['Ask Price 1']
            data['is'].loc[data['diff'] > 0]  = 1
            data['diff'] = data['Bid Price 1'] - data['Bid Price 1'].shift(1)
            data['is'].loc[data['diff'] > 0]  = 1
            #
            data['eta_is'] = np.nan
            data['eta_is'].loc[data['is'] == 1] = data['diff'].loc[data['is'] == 1]
            varsPerSec = data.groupby(['sec','Type','TradeDirection'])[['q_LO','q_MO','eta_is']].apply(nanmed)
            perSecDF = varsPerSec.merge(intensityPerSec)
            print(perSecDF)
            dataOrig = data.copy()
            for d, side in zip([-1,1],['Ask', 'Bid']):
                data = dataOrig.loc[dataOrig['TradeDirection'] == d]
                data['m_T'] = data[side + ' Price 2'] - data[side + ' Price 1']
                data['m_T_prev'] = data['m_T'].shift(1).fillna(0).apply(lambda x: np.abs(np.round(x,decimals=2)))
                arr = data[[side + ' Size ' + str(i) for i in range(1,11)]].values
                x = abs(arr.cumsum(axis=1) - (arr.sum(axis=1)/2).reshape((len(arr),1))).argmin(axis=1)
                data['M_0.5'] = (data[[side + ' Price ' + str(i) for i in range(1,11)]].values)[x]
                data['m_D'] = data['M_0.5'] - data[side + ' Price 2']
                data['m_D_prev'] = data['m_D'].shift(1).fillna(0).apply(lambda x: np.abs(np.round(x,decimals=2)))

                data['q_LO'] = data['q_LO'].fillna(method='ffill')

                data['q_MO'] = data['q_MO'].fillna(method='ffill')

                data['eta_is'] = data['eta_is'].fillna(method='ffill')
                #
                data['Type_mT'] = data['Type'].astype(int).astype(str) +  data['is'].astype(int).astype(str) +  data['m_T'].astype(int).astype(str)
                data['Type_mT_1'] = data['Type_mT'].shift(1)
                data['Type_mD'] = data['Type'].astype(int).astype(str) +  data['is'].astype(int).astype(str) +  data['m_D'].astype(int).astype(str)
                data['Type_mD_1'] = data['Type_mD'].shift(1)
                data['Type_q_LO'] = data['Type'].astype(int).astype(str) +  data['is'].astype(int).astype(str) +  data['q_LO'].astype(int).astype(str)
                data['Type_q_LO_1'] = data['Type_q_LO'].shift(1)
                data['Type_q_MO'] = data['Type'].astype(int).astype(str) +  data['is'].astype(int).astype(str) +  data['q_MO'].astype(int).astype(str)
                data['Type_q_MO_1'] = data['Type_q_MO'].shift(1)
                data['Type_eta_is'] = data['Type'].astype(int).astype(str) +  data['is'].astype(int).astype(str) +  data['eta_is'].astype(int).astype(str)
                data['Type_eta_is_1'] = data['Type_eta_is'].shift(1)
                # data.head()
                if len(condCounts_mT)==0:
                    condCounts_mT = data.groupby(['Type_mT','Type_mT_1'])['Time'].count()
                    uncondCounts_mT = data.groupby(['Type_mT'])['Time'].count()
                    condCounts_mD = data.groupby(['Type_mD','Type_mD_1'])['Time'].count()
                    uncondCounts_mD = data.groupby(['Type_mD'])['Time'].count()
                    condCounts_q_LO = data.groupby(['Type_q_LO','Type_q_LO_1'])['Time'].count()
                    uncondCounts_q_LO = data.groupby(['Type_q_LO'])['Time'].count()
                    condCounts_q_MO = data.groupby(['Type_q_MO','Type_q_MO_1'])['Time'].count()
                    uncondCounts_q_MO = data.groupby(['Type_q_MO'])['Time'].count()
                    condCounts_eta_is = data.groupby(['Type_eta_is','Type_eta_is_1'])['Time'].count()
                    uncondCounts_eta_is = data.groupby(['Type_eta_is'])['Time'].count()
                else:
                    tmp = data.groupby(['Type_mT','Type_mT_1'])['Time'].count()
                    condCounts_mT = condCounts_mT.add(tmp, fill_value=0)
                    tmp= data.groupby(['Type_mT'])['Time'].count()
                    uncondCounts_mT = uncondCounts_mT.add(tmp, fill_value=0)
                    tmp = data.groupby(['Type_mD','Type_mD_1'])['Time'].count()
                    condCounts_mD = condCounts_mD.add(tmp, fill_value=0)
                    tmp= data.groupby(['Type_mD'])['Time'].count()
                    uncondCounts_mD = uncondCounts_mD.add(tmp, fill_value=0)
                    tmp = data.groupby(['Type_q_LO','Type_q_LO_1'])['Time'].count()
                    condCounts_q_LO = condCounts_q_LO.add(tmp, fill_value=0)
                    tmp= data.groupby(['Type_q_LO'])['Time'].count()
                    uncondCounts_q_LO = uncondCounts_q_LO.add(tmp, fill_value=0)
                    tmp = data.groupby(['Type_q_MO','Type_q_MO_1'])['Time'].count()
                    condCounts_q_MO = condCounts_q_MO.add(tmp, fill_value=0)
                    tmp= data.groupby(['Type_q_MO'])['Time'].count()
                    uncondCounts_q_MO = uncondCounts_q_MO.add(tmp, fill_value=0)
                    tmp = data.groupby(['Type_eta_is','Type_eta_is_1'])['Time'].count()
                    condCounts_eta_is = condCounts_eta_is.add(tmp, fill_value=0)
                    tmp= data.groupby(['Type_eta_is'])['Time'].count()
                    uncondCounts_eta_is = uncondCounts_eta_is.add(tmp, fill_value=0)

        with open("/SAN/fca/Konark_PhD_Experiments/smallTick/"+ric+"_EDA_assumptions", "wb") as f:
            pickle.dump({'perSecDF' : perSecDF, 'condCounts_mT' : condCounts_mT,'uncondCounts_mT' :  uncondCounts_mT, 'condCounts_mD' : condCounts_mD,'uncondCounts_mD' :  uncondCounts_mD, 'condCounts_q_LO' : condCounts_q_LO,'uncondCounts_q_LO' :  uncondCounts_q_LO, 'condCounts_q_MO' : condCounts_q_MO,'uncondCounts_q_MO' :  uncondCounts_q_MO, 'condCounts_eta_is' : condCounts_eta_is,'uncondCounts_eta_is' :  uncondCounts_eta_is }, f)

def plotLeverage(stocks):
    for s in stocks:
        with open('/SAN/fca/Konark_PhD_Experiments/smallTick/'+s+'_EDA_leverageIS_truetop', 'rb') as f:
            dict_res = pickle.load(f)
    condCounts, uncondCounts = dict_res['condCounts'], dict_res['uncondCounts']
    df = uncondCounts/uncondCounts.sum()
    df = df[df>1e-4].reset_index()
    df['depth'] = df['TypeDepth'].apply(lambda x: int(x[2:]))
    depthMax = df.depth.max()
    #depthMax = 100
    categories = np.append(-1*np.logspace(np.log(depthMax)/np.log(10),0,50), np.append([0],np.logspace(0,np.log(depthMax)/np.log(10), 50)))
    categories =np.unique(categories.astype(int))
    print(categories)
    condCounts_categorized = condCounts.reset_index().copy()
    condCounts_categorized['cat'] = condCounts_categorized['TypeDepth'].apply(lambda x : x[:2]+ str((-len(categories)//2) + np.searchsorted(categories, int(x[2:]))))
    condCounts_categorized['cat_1'] = condCounts_categorized['TypeDepth_1'].apply(lambda x : x[:2]+ str((-len(categories)//2) + np.searchsorted(categories, int(x[2:]))))
    uncondCounts_categorized = uncondCounts.reset_index().copy()
    uncondCounts_categorized['cat'] = uncondCounts_categorized['TypeDepth'].apply(lambda x : x[:2]+ str((-len(categories)//2) + np.searchsorted(categories, int(x[2:]))))
    condCounts_categorized = condCounts_categorized.groupby(['cat','cat_1'])['Time'].sum()
    uncondCounts_categorized = uncondCounts_categorized.groupby('cat')['Time'].sum()
    condProb_categorized = condCounts_categorized/uncondCounts_categorized
    uncondProb_categorized = uncondCounts_categorized/uncondCounts_categorized.sum()
    leverage_categorized = condProb_categorized/uncondProb_categorized
    leverage_categorized = leverage_categorized.reset_index()
    leverage_categorized[['Type','cat']]= np.stack(leverage_categorized['cat'].apply(lambda x: np.array([int(x[:2]) , int(x[2:])])).values)
    leverage_categorized[['Type_1','cat_1']]= np.stack(leverage_categorized['cat_1'].apply(lambda x: np.array([int(x[:2]) , int(x[2:])])).values)
    # depthMax = int(np.max(leverage_categorized.Depth.apply(np.abs)))
    leverage_categorized = leverage_categorized.set_index(['Type_1','cat_1','Type', 'cat'])
    matrix = np.zeros((int(len(categories)*4), int(len(categories)*4)))
    converter ={  0: 10, 1:11, 2: 30,  3: 41}
    for i in range(int(len(categories)*4)):
        type_i = converter[i//len(categories)]
        depth_i = (-len(categories)//2) + i%len(categories)
        for j in range(len(categories)*4):
            type_j = converter[j//len(categories)]
            depth_j =(-len(categories)//2) +  j%len(categories)
            if (int(type_i), int(depth_i), int(type_j), int(depth_j)) in leverage_categorized.index:
                matrix[i, j] = leverage_categorized.loc[(int(type_i), int(depth_i), int(type_j), int(depth_j))]['Time']

    fig, axs = plt.subplots(1, 2, figsize=(10,10), gridspec_kw={'width_ratios': [1, 5]})
    im = axs[1].imshow(matrix,norm='log', cmap='seismic', aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    locs = np.linspace(0, len(matrix), num = 41)
    labels = []
    for x, y in zip( np.array(["LO"]*10 + ['IS']*10 + ["CO"]*10 + ["MO"]*10) , np.array([" " + str(int(x)) for x in np.array(list(categories)*4)[[int(x) for x in (locs[:-1])]]])):
        labels.append(x+y)
    axs[1].set_xticks(ticks = -0.5+locs[:-1], labels = labels, rotation=90)
    axs[1].set_yticks(ticks = -0.5+locs[:-1] ,labels = labels)
    df = uncondProb_categorized.reset_index()
    df['Type'] = df['cat'].apply(lambda x: int(x[:2]))
    df['Depth'] = df['cat'].apply(lambda x: int(x[2:]))
    df = df.sort_values(['Type','Depth'])
    df = df.set_index(['Type','Depth'])
    mat = np.zeros(len(categories)*4)
    for i in range(int(len(categories)*4)):
        type_i = converter[i//len(categories)]
        depth_i = (-len(categories)//2) + i%len(categories)
        if (int(type_i), int(depth_i)) in df.index:
            mat[i] = df.loc[(int(type_i), int(depth_i))]['Time']
    im2 = axs[0].imshow(1e-7+mat.reshape((len(mat), 1)), norm='log', cmap = 'cool')
    plt.colorbar(im2, location='left', pad=0.5)
    axs[0].set_xticks(ticks = [])
    axs[0].set_yticks(ticks = -0.5+locs[:-1] ,labels = labels)
    fig.tight_layout()
    axs[0].set_title('Unconditional Prob.')
    axs[1].set_title('Leverage : Conditional Prob.')
    fig.suptitle('Leverage : '+s)
    fig.subplots_adjust(top=0.9)
    plt.savefig("/SAN/fca/Konark_PhD_Experiments/smallTick/"+s+"_leverage_truetop.png")

main( sys.argv[1] , assumptions= sys.argv[2])