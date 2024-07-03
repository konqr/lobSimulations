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
        self.bids={} #holding list of pricelevels which are in the form of (key, value)=(price, [list of orders])
        self.asks={}
        self.askprices={ 
            "Ask_touch" : priceMid0 + np.floor(spread0/2)*self.ticksize,
            "Ask_deep" : priceMid0 + np.floor(spread0/2)*self.ticksize + self.ticksize
        }
        self.bidprices={
            "Bid_touch" : priceMid0 - np.ceil(spread0/2)*self.ticksize,
            "Bid_deep" : priceMid0 - np.ceil(spread0/2)*self.ticksize - self.ticksize
        }
        self.spread=abs(self.askprices["Ask_touch"]-self.bidprices["Bid_touch"])
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
                        #generate new ask_deep
                        pricelvl=self.askprices["Ask_touch"]+self.ticksize
                        self.askprices["Ask_deep"]=pricelvl
                        self.asks[pricelvl]=[size]
                    else:
                        self.asks[self.askprices["Ask_deep"]].append(size)
                elif side=="Bid":
                    if np.abs(self.askprices["Bid_touch"] - self.askprices["Bid_deep"])> 2.5 * self.ticksize:
                        #generate new bid_deep
                        pricelvl=self.bidprices["Bid_touch"]-self.ticksize
                        self.bidprices["Bid_deep"]=pricelvl
                        self.bids[pricelvl]=[size]
                    else:
                        self.bids[self.bidprices["Bid_deep"]].append(size)
            elif "top" in event:
                if side=="Ask":
                    self.asks[self.askprices[side+"_touch"]].append(size)    
                else:
                    self.bids[self.bidprices[side+"_touch"]].append(size)
            else: #Inspread
                if side=="Ask":
                    self.asks
                
                else:
                    
                
            else: #inspread
        elif "mo" in event:
            #Market_order
        elif "co" in event:
            #Cancel_order
                

"""
MO: asktouch, askdeep
LO: asktouch, askdeep, bidtouch, biddeep, askinspread, bidinspread
CO: asktouch, askdeep, bidtouch, biddeep

Ask_Deep:  (45,08, [150])
Ask_touch: (45.05, [100])
Bid_touch: (45.03, [100])
Bid_Deep: (45.02, [150])

Order comes in: LO_askdeep, size=50

Ask_touch:  (45.08, [200])

Bid_touch: (45.02, [150]) 

Ask_3:  (45.08, [150])
Ask_Deep: (45.06, [50])
Ask_touch: (45.05, [100])
Bid_touch: (45.03, [100])
Bid_Deep: (45.02, [150])
"""



    def peaklob0(self)-> dict: 
        rtn={
            "Ask_deep": (self.askprices["Ask_deep"], sum(self.asks[self.askprices["Ask_deep"]])),
            "Ask_touch": (self.askprices["Ask_touch"], sum(self.asks[self.askprices["Ask_touch"]])),
            "Bid_touch": (self.bidprices["Bid_touch"], sum(self.bids[self.bidprices["Bid_touch"]])),
            "Bid_deep": (self.bidprices["Bid_touch"], sum(self.bids[self.bidprices["Bid_touch"]]))
        }
        return rtn

    def peaklobl3(self):
        rtn={
            "Ask_deep": (self.askprices["Ask_deep"], self.asks[self.askprices["Ask_deep"]]),
            "Ask_touch": (self.askprices["Ask_touch"], self.asks[self.askprices["Ask_touch"]]),
            "Bid_touch": (self.bidprices["Bid_touch"], self.bids[self.bidprices["Bid_touch"]]),
            "Bid_deep": (self.bidprices["Bid_touch"], self.bids[self.bidprices["Bid_touch"]])
        }
        return rtn

    def logtofile(self, destination):
        

