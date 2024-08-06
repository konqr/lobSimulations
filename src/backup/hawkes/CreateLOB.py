import pickle
import numpy as np
import pandas as pd
import time
import numpy as np
import os 
import random
cols = ["lo_deep_Ask", "co_deep_Ask", "lo_top_Ask","co_top_Ask", "mo_Ask", "lo_inspread_Ask" ,
            "lo_inspread_Bid" , "mo_Bid", "co_top_Bid", "lo_top_Bid", "co_deep_Bid","lo_deep_Bid" ]
indextocol={}
for i in range(len(cols)):
    indextocol[i]=cols[i]
class LimitOrderBook:
    def __init__(self, Pi_Q0, priceMid0=260, spread0=4, ticksize=0.01, numOrdersPerLevel=10):
        self.levels = ["Ask_deep", "Ask_touch", "Bid_touch", "Bid_deep"]
        self.numOrdersPerLevel=numOrdersPerLevel
        colsToLevels = {
            "lo_deep_Ask" : "Ask_deep",
            "lo_top_Ask" : "Ask_touch",
            "lo_top_Bid" : "Bid_touch",
            "lo_deep_Bid" : "Bid_deep"
        }
        #init prices
        self.ticksize=ticksize
        self.spread0=spread0
        self.priceMid0=priceMid0
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
        self.Pi_Q0=Pi_Q0
        #init random sizes 
    
    def updatebidaskprices(self): #to be implemented
        pass
        if self.asks[self.askprices["Ask_touch"]]==[]:
            del self.asks[self.askprices["Ask_touch"]]
            self.askprices["Ask_touch"]=self.askprices["Ask_deep"]
        if self.bids[self.bidprices["Bid_touch"]]==[]
            del self.bids[self.bidprices["Bid_touch"]]
            self.bidprices=["Bid_touch"]-self.bidprices["Bid_deep"]
        if self.asks[self.askprices["Ask_deep"]]==[]:
            self.askprices["Ask_deep"]=self.askprices["Ask_touch"]+self.ticksize
        if self.bids[self.bidprices["Bid_deep"]]==[]:
            self.bidprices["Bid_deep"]=self.bidprices["Bid_touch"]-self.ticksize
        
             
    def populateLoBlevel(self, k):    
        pi=self.Pi_Q0[k]
            #geometric + dirac deltas; pi = (p, diracdeltas(i,p_i))
        qSize=generateqsize(pi)
        if "Ask" in k:
            print("qsize init: ", k, ": ", qSize)
            d=qSize//self.numOrdersPerLevel
            if(d!=0):
                r=qSize%self.numOrdersPerLevel
                self.asks[self.askprices[k]]=[d+r] +[d]*(self.numOrdersPerLevel-1)
            else:
                self.asks[self.askprices[k]]=[qSize]
        else:
            print("qsize init: ", k, ": ", qSize)
            d=qSize//self.numOrdersPerLevel
            if (d!=0):
                r=qSize%self.numOrdersPerLevel
                self.bids[self.bidprices[k]]=[d+r] +[d]*(self.numOrdersPerLevel-1)
            else:
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
                    pricelvl=self.askprices["Ask_touch"]-self.ticksize
                    self.askprices["Ask_deep"]=self.askprices["Ask_touch"]
                    self.askprices["Ask_touch"]=pricelvl
                    self.asks[pricelvl]=[size]
                else:
                    pricelvl=self.bidprices["Bid_touch"]+self.ticksize
                    self.bidprices["Bid_deep"]=self.bidprices["Bid_touch"]
                    self.bidrices["Bid_touch"]=pricelvl
                    self.bids[pricelvl]=[size]
        elif "mo" in event: #market order
            if side=="Ask":
                bidprice=self.bidprices["Bid_touch"]
                while size>0:
                    totalquantity=sum(self.bids[bidprice])
                    if(totalquantity<=size): #consume all orders at a pricelevel
                        size-=totalquantity
                        del self.bids[bidprice]
                        ###update bid touch price
                        self.updatebidaskprices
                        #queue depletion
                    else: #partially consume a pricelevel
                        j=0
                        while size>0:
                            if self.bids[bidprice][j]<=size:
                                size-=self.bids[bidprice][j]
                                j+=1
                            else:
                                self.bids[bidprice][j]-=size
                                size=0
                                break
                        self.bids[bidprice]=self.bids[bidprice][j:]
            else: #bid market order
                askprice=self.askprices["Ask_touch"]
                while size>0:
                    totalquantity=sum(self.asks[askprice])
                    if(totalquantity<=size): #consume all orders at a pricelevel
                        size-=totalquantity
                        del self.asks[askprice]
                        ###update ask touch price


                        #queue depletion
                    else: #partially consume a pricelevel
                        j=0
                        while size>0:
                            if self.asks[askprice][j]<=size:
                                size-=self.asks[askprice][j]
                                j+=1
                            else:
                                self.asks[askprice][j]-=size
                                size=0
                                break
                        self.asks[askprice]=self.asks[askprice][j:]

        elif "co" in event: #Cancel Order
            a=None
            k=None
            if "deep" in event:
                if side=="Ask":
                    k="Ask_deep"
                    a=self.asks[self.askprices["Ask_deep"]]
                    a.pop(random.randrange(0,len(a)))
                        
                else:
                    k="Bid_deep"
                    a=self.asks[self.bidprices["Bid_deep"]]
                    a.pop(random.randrange(0,len(a)))
            elif "top" in event:
                if side=="Ask":
                    k="Ask_deep"
                    a=self.asks[self.askprices["Ask_touch"]]
                    a.pop(random.randrange(0,len(a)))
                else:
                    k="Bid_deep"
                    a=self.asks[self.bidprices["Bid_touch"]]
                    a.pop(random.randrange(0,len(a)))
            if a==[]: #check queue depletion
                #update touch and deep prices
                self.updatebidaskprices
                pi=self.Pi_Q0[k]
                qSize=generateqsize(pi)
                self.populateLoBlevel(k)
        #update spread:
        self.spread=abs(self.askprices["Ask_touch"]-self.bidprices["Bid_touch"])

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

    def getlob(self):
        rtn=[[self.askprices["Ask_deep"], self.asks[self.askprices["Ask_deep"]]], [self.askprices["Ask_touch"], self.asks[self.askprices["Ask_touch"]]], [self.bidprices["Bid_touch"], self.bids[self.bidprices["Bid_touch"]]], [self.bidprices["Bid_deep"], self.bids[self.bidprices["Bid_deep"]]], self.spread]
        return rtn
        
        
        
def logtofile(data, destination):
    #data=[order book, spread]
    
    return
def generateqsize(pi):
    #k, the level of the bid/ask spread i.e Ask_touch, Bid_touch...
    #pi=(p, diracdeltas(i, p_i))
    p = pi[0]
    dd = pi[1]
    pi = np.array([p*(1-p)**k for k in range(1,100000)])
    # pi = pi*(1-sum([d[1] for d in dd]))/sum(pi)
    for i, p_i in dd:
        pi[i-1] = p_i + pi[i-1]
    pi = pi/sum(pi)
    cdf = np.cumsum(pi)
    a = np.random.uniform(0, 1)
    qSize = np.argmax(cdf>=a)+1
    return qSize
        
        
