import pandas as pd
import os
import itertools as it
import time
import datetime as dt
import numpy as np


class dataLoader():
    # Message file information:
    # ----------------------------------------------------------
    #
    #   - Dimension:    (NumberEvents x 6)
    #
    #   - Structure:    Each row:
    #                   Time stamp (sec after midnight with decimal
    #                   precision of at least milliseconds and
    #                   up to nanoseconds depending on the period),
    #                   Event type, Order ID, Size (# of shares),
    #                   Price, Direction
    #
    #                   Event types:
    #                       - '1'   Submission new limit order
    #                       - '2'   Cancellation (partial)
    #                       - '3'   Deletion (total order)
    #                       - '4'   Execution of a visible limit order
    #                       - '5'   Execution of a hidden limit order
    # 			- '7'   Trading Halt (Detailed
    #                               information below)
    #
    #                   Direction:
    #                       - '-1'  Sell limit order
    #                       - '1'  Buy limit order
    #                       - NOTE: Execution of a sell (buy)
    #                               limit order corresponds to
    #                               a buyer-(seller-) initiated
    #                               trade, i.e. a BUY (SELL) trade.
    #
    # ----------------------------------------------------------

    # Orderbook file information:
    # ----------------------------------------------------------
    #
    #   - Dimension:    (NumberEvents x (NumberLevels*4))
    #
    #   - Structure:    Each row:
    #                   Ask price 1, Ask volume 1, Bid price 1,
    #                   Bid volume 1, Ask price 2, Ask volume 2,
    #                   Bid price 2, Bid volume 2, ...
    #
    #   - Note:         Unoccupied bid (ask) price levels are
    #                   set to -9999999999 (9999999999) with volume 0.
    #
    # ----------------------------------------------------------

    def __init__(self, ric, sDate, eDate, **kwargs):
        self.ric = ric
        self.sDate = sDate
        self.eDate = eDate
        self.nlevels = kwargs.get("nlevels", 10)
        self.dataPath = kwargs.get("dataPath", "D:\\Work\\PhD\\Data\\")

    def load(self):
        """

        :return:
        """
        data = []
        ric = self.ric.split(".")[0]
        for d in pd.date_range(self.sDate, self.eDate): # TODO: business days try catch
            theMessageBookFileName = self.dataPath + ric + "_" + d.strftime("%Y-%m-%d") + "_34200000_57600000_message_10.csv"
            if (ric + "_" + d.strftime("%Y-%m-%d") + "_34200000_57600000_message_10.csv") not in os.listdir(self.dataPath): continue
            print(theMessageBookFileName)
            theOrderBookFileName = self.dataPath +  ric + "_" + d.strftime("%Y-%m-%d") + "_34200000_57600000_orderbook_10.csv"
            theMessageBook = pd.read_csv(theMessageBookFileName,
                                         names=['Time', 'Type', 'OrderID', 'Size', 'Price', 'TradeDirection', 'tmp'])
            startTrad = 9.5 * 60 * 60  # 9:30:00.000 in ms after midnight
            endTrad = 16 * 60 * 60  # 16:00:00.000 in ms after midnight

            theMessageBookFiltered = theMessageBook[theMessageBook['Time'] >= startTrad]
            theMessageBookFiltered = theMessageBookFiltered[theMessageBookFiltered['Time'] <= endTrad]
            auctTime = theMessageBookFiltered.loc[theMessageBookFiltered.Type == 6].iloc[0].Time
            if len(theMessageBookFiltered.loc[theMessageBookFiltered.Type == 6]) > 1:
                auctTime2 = theMessageBookFiltered.loc[theMessageBookFiltered.Type == 6].iloc[1].Time
            else:
                auctTime2 = endTrad
            theMessageBookFiltered = theMessageBookFiltered[theMessageBookFiltered['Time'] >= auctTime] # TODO: doesnt remove < 1 ns events before auction
            theMessageBookFiltered = theMessageBookFiltered[theMessageBookFiltered['Time'] <= auctTime2]

            col = ['Ask Price ', 'Ask Size ', 'Bid Price ', 'Bid Size ']

            theNames = []
            cols = []
            for i in range(1, 11):
                for j in col:
                    if i <= self.nlevels:
                        cols.append(str(j)+str(i))
                    theNames.append(str(j) + str(i))

            theOrderBook = pd.read_csv(theOrderBookFileName, names=theNames)
            theOrderBook = theOrderBook[cols]
            timeIndex = theMessageBook.index[(theMessageBook.Time >= auctTime) & (theMessageBook.Time <= auctTime2)]

            theOrderBookFiltered = theOrderBook.iloc[timeIndex]
            # Convert prices into dollars
            #    Note: LOBSTER stores prices in dollar price times 10000

            for i in list(range(0,len(theOrderBookFiltered.columns),2)):
                theOrderBookFiltered[theOrderBookFiltered.columns[i]]  = theOrderBookFiltered[theOrderBookFiltered.columns[i]]/10000
            combinedDf = pd.concat([theMessageBookFiltered , theOrderBookFiltered], axis = 1)
            combinedDf["Date"] = d.strftime("%Y-%m-%d")
            data += [combinedDf]
        return data

    def loadBinned(self, binLength = 1, filterTop=False):
        """

        :param binLength:
        :param filterTop:
        :return:
        """
        # TODO : binLength = 0.01 gives me memory error already -
        #   - will pyKX help?
        #   - store binned data to disk and use later?

        data = self.load()
        orderTypeDict = {'limit' : [1], 'cancel': [2,3], 'market' : [4]}
        binnedData = {}
        for d in data:
            binnedL = {}
            for k, v in orderTypeDict.items():
                for s in [1, -1]:
                    side = "bid" if s == 1 else "ask"
                    l = d.loc[(d.Type.apply(lambda x: x in v)) & (d.TradeDirection == s)]
                    if filterTop:
                        l = l.loc[l.apply(lambda x: (x["Price"]/10000 <= x['Ask Price 1'] + 1e-3) and (x["Price"]/10000 >= x['Bid Price 1'] - 1e-3), axis=1)]
                    l['count'] = 1
                    bins = np.arange(d.Time.min() - 1e-3, d.Time.max(), binLength)
                    labels = np.arange(0, len(bins)-1)
                    l['binIndex'] = pd.cut(l['Time'], bins=bins, labels=labels)
                    binL = l.groupby("binIndex").sum()[['count','Size']]
                    binL.reset_index(inplace=True)
                    binnedL[k + "_" + side] = binL
            binnedData[d.Date.iloc[0]] = binnedL
        return binnedData

    def loadRollingWindows(self, binLength = 1, filterTop = False):
        """

        :param binLength:
        :param filterTop:
        :return:
        """
        data = self.load()
        orderTypeDict = {'limit' : [1], 'cancel': [2,3], 'market' : [4]}
        binnedData = {}
        for d in data:
            binnedL = {}
            for k, v in orderTypeDict.items():
                for s in [1, -1]:
                    side = "bid" if s == 1 else "ask"
                    l = d.loc[(d.Type.apply(lambda x: x in v)) & (d.TradeDirection == s)]
                    if filterTop:
                        l = l.loc[l.apply(lambda x: (x["Price"]/10000 <= x['Ask Price 1'] + 1e-3) and (x["Price"]/10000 >= x['Bid Price 1'] - 1e-3), axis=1)]
                    l['count'] = 1
                    binL =l.set_index(l.Time.apply(lambda x : dt.datetime.strptime(l.Date.iloc[0], "%Y-%m-%d")+dt.timedelta(seconds=x)))[['count', 'Size']].rolling(window = dt.timedelta(seconds=binLength)).sum()
                    binnedL[k + "_" + side] = binL.loc[binL.index[binL.index > binL.index[0] + dt.timedelta(seconds=binLength)]]
            binnedData[d.Date.iloc[0]] = binnedL
        return binnedData

    def load12DTimestamps(self):
        """

        :return:
        """
        data = self.load()
        if len(data) == 0: return []
        offset = 9.5*3600
        orderTypeDict = {'limit' : [1], 'cancel': [2,3], 'market' : [4]}
        res = {}
        for df in data:
            df['Time'] = df['Time'] - offset
            df['BidDiff'] = df['Bid Price 1'].diff()
            df['AskDiff'] = df['Ask Price 1'].diff()
            df['BidDiff2']= df['Bid Price 2'].diff()
            df['AskDiff2']= df['Ask Price 2'].diff()
            arr = []
            df_res_l = []
            for s in [1, -1]:
                side = "Bid" if s == 1 else "Ask"
                lo = df.loc[(df.Type.apply(lambda x: x in orderTypeDict['limit']))&(df.TradeDirection == s)]
                lo_deep = lo.loc[(lo.apply(lambda x: np.isclose(x.Price/10000, x[side + " Price 2"]), axis =1))]
                lo_deep['event'] = "lo_deep_" + side
                co = df.loc[(df.Type.apply(lambda x: x in orderTypeDict['cancel']))&(df.TradeDirection == s)]
                co_deep = co.loc[co.apply(lambda x: np.isclose(x.Price/10000, x[side + " Price 2"]), axis=1)|(((co['BidDiff2'] < 0)&(co['BidDiff'] == 0))|((co['AskDiff2'] > 0)&(co['AskDiff'] == 0)))]
                co_deep['event'] = "co_deep_" + side
                lo_inspread = lo.loc[((lo['BidDiff'] > 0)|(lo['AskDiff'] < 0))]
                lo_inspread['event'] = "lo_inspread_" + side
                lo_top = lo.loc[(lo.apply(lambda x: (x["Price"]/10000 <= x['Ask Price 1'] + 1e-3) and (x["Price"]/10000 >= x['Bid Price 1'] - 1e-3), axis=1))]
                lo_top = lo_top.loc[lo_top[side+"Diff"]==0]
                lo_top['event'] = "lo_top_" + side
                co_top = co.loc[(co.apply(lambda x: (x["Price"]/10000 <= x['Ask Price 1'] + 1e-3) and (x["Price"]/10000 >= x['Bid Price 1'] - 1e-3), axis=1))]
                co_top['event'] = "co_top_" + side
                mo = df.loc[(df.Type.apply(lambda x: x in orderTypeDict['market']))&(df.TradeDirection == s)]
                mo['event'] = 'mo_' + side
                df_res = pd.concat([lo_deep, co_deep, lo_top, co_top, mo, lo_inspread])
                df_res_l += [df_res]
                l = [lo_deep.Time.values, co_deep.Time.values, lo_top.Time.values, co_top.Time.values, mo.Time.values, lo_inspread.Time.values]
                if s == 1: l.reverse()
                arr += l
            df_res_l = pd.concat(df_res_l)
            df_res_l.to_csv(self.dataPath + self.ric + "_" + df.Date.iloc[0] +"_12D.csv")
            res[df.Date.iloc[0]] = arr

        return res

    def load8DTimestamps_Bacry(self):
        """

        :return:
        """
        data = self.load()
        if len(data) == 0: return []
        offset = 9.5*3600
        orderTypeDict = {'limit' : [1], 'cancel': [2,3], 'market' : [4]}
        res = {}
        for df in data:
            df['Time'] = df['Time'] - offset
            df['BidDiff'] = df['Bid Price 1'].diff()
            df['AskDiff'] = df['Ask Price 1'].diff()
            df['BidDiff2']= df['Bid Price 2'].diff()
            df['AskDiff2']= df['Ask Price 2'].diff()
            arr = []
            df_res_l = []
            for s in [1, -1]:
                side = "Bid" if s == 1 else "Ask"
                P = df.loc[df[side+"Diff"]!= 0]
                P["event"] = "pc_" + side
                mo = df.loc[(df.Type.apply(lambda x: x in orderTypeDict['market']))&(df.TradeDirection == s)&(df[side+"Diff"]== 0)]
                mo['event'] = 'mo_' + side
                lo = df.loc[(df.Type.apply(lambda x: x in orderTypeDict['limit']))&(df.TradeDirection == s)&(df[side+"Diff"]== 0)]
                co = df.loc[(df.Type.apply(lambda x: x in orderTypeDict['cancel']))&(df.TradeDirection == s)&(df[side+"Diff"]== 0)]
                lo_top = lo.loc[(lo.apply(lambda x: (x["Price"]/10000 <= x['Ask Price 1'] + 1e-3) and (x["Price"]/10000 >= x['Bid Price 1'] - 1e-3), axis=1))]
                lo_top['event'] = "lo_top_" + side
                co_top = co.loc[(co.apply(lambda x: (x["Price"]/10000 <= x['Ask Price 1'] + 1e-3) and (x["Price"]/10000 >= x['Bid Price 1'] - 1e-3), axis=1))]
                co_top['event'] = "co_top_" + side
                df_res = pd.concat([P, mo, lo_top, co_top])
                df_res_l += [df_res]
                l = [P.Time.values, mo.Time.values, lo_top.Time.values, co_top.Time.values]
                if s == 1: l.reverse()
                arr += l
            df_res_l = pd.concat(df_res_l)
            df_res_l.to_csv(self.dataPath + self.ric + "_" + df.Date.iloc[0] +"_8D_Bacry.csv")
            res[df.Date.iloc[0]] = arr

        return res



