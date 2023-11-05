import pandas as pd
import os
import itertools as it
import time
import datetime as dt
import numpy as np


class Loader():
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
        self.dataPath = "D:\\Work\\PhD\\Data\\"

    def load(self):
        data = []
        ric = self.ric.split(".")[0]
        for d in pd.date_range(self.sDate, self.eDate): # TODO: business days try catch
            theMessageBookFileName = self.dataPath + ric + "_" + d.strftime("%Y-%m-%d") + "_34200000_57600000_message_10.csv"
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
            theMessageBookFiltered = theMessageBookFiltered[theMessageBookFiltered['Time'] >= auctTime]
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
            combinedDf["Date"] = theMessageBookFileName.split("_")[1]
            data += [combinedDf]
        return data

    def loadBinned(self, binLength = 1, filterTop=False):
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

