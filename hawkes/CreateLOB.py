import pickle
import numpy as np
import pandas as pd
import time
import numpy as np
import os 
cols = ["lo_deep_Ask", "co_deep_Ask", "lo_top_Ask","co_top_Ask", "mo_Ask", "lo_inspread_Ask" ,
            "lo_inspread_Bid" , "mo_Bid", "co_top_Bid", "lo_top_Bid", "co_deep_Bid","lo_deep_Bid" ]
indextocol={}
for i in range(len(cols)):
    indextocol[i]=cols[i]
class LimitOrderBook:
    def __init__(self, Pi_Q0, priceMid0=260, spread0=4, ticksize=0.01, numOrdersPerLevel=10):
        levels = ["Ask_deep", "Ask_touch", "Bid_touch", "Bid_deep"]
        colsToLevels = {
            "lo_deep_Ask" : "Ask_deep",
            "lo_top_Ask" : "Ask_touch",
            "lo_top_Bid" : "Bid_touch",
            "lo_deep_Bid" : "Bid_deep"
        }
        #init prices
        self.ticksize=ticksize
        self.bids={}
        self.asks={}
        self.askprices={ 
            "Ask_touch" : priceMid0 + np.floor(spread0/2)*self.ticksize,
            "Ask_deep" : priceMid0 + np.floor(spread0/2)*self.ticksize + self.ticksize
        }
        self.bidprices={
            "Bid_touch" : priceMid0 - np.ceil(spread0/2)*self.ticksize,
            "Bid_deep" : priceMid0 - np.ceil(spread0/2)*self.ticksize - self.ticksize
        }
        #init random sizes 
        for k, pi in Pi_Q0.items():
            #geometric + dirac deltas; pi = (p, diracdeltas(i,p_i))
            p = pi[0]
            dd = pi[1]
            pi = np.array([p*(1-p)**k for k in range(1,100000)])
            # pi = pi*(1-sum([d[1] for d in dd]))/sum(pi)
            for i, p_i in dd:
                pi[i-1] = p_i + pi[i-1]
            pi = pi/sum(pi)
            cdf = np.cumsum(pi)
            a = np.random.uniform(0, 1)
            qSize = np.argmax(cdf>=a) + 1
            if "Ask" in k:
                self.asks[self.askprices[k]]=[qSize]
            elif "Bid" in k:
                self.bids[self.bidprices[k]]=[qSize]

    def processorders(self, order):
        t, event, size=order[0], order[1], order[2]
        if "Ask" in event:
            side="Ask"
        elif "Bid" in event:
            side="Bid"
        else:
            pass
        if "lo" in event:
            #Limit_order
            if "deep" in event:
                if side=="Ask":
                    if np.abs(self.askprices["Ask_touch"] - self.askprices["Ask_deep"])> 2.5 * self.ticksize:
                elif side=="Bid":
                    if np.abs(self.askprices["Ask_touch"] - self.askprices["Ask_deep"])> 2.5 * self.ticksize:

            elif "top" in event:
                self.asks[self.askprices[side+"_touch"]].append(size)
            else: #inspread
                





        elif "mo" in event:
            #Market_order
        elif "co" in event:
            #Cancel_order
            


def peaklob0():

def peaklobl3(data):
    p

def logtofile(data, destination):

