import pickle
import numpy as np
import pandas as pd
import time
import numpy as np
import os 
import random
import logging
from typing import Any, List, Optional, Tuple, ClassVar
from RLenv.SimulationEntities.Entity import Entity
from RLenv.Exceptions import *
from RLenv.SimulationEntities.TradingAgent import TradingAgent
from RLenv.Stochastic_Processes.Arrival_Models import ArrivalModel, HawkesArrival
from RLenv.Orders import *
from RLenv.Messages.ExchangeMessages import *

logger=logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
class Exchange(Entity):
    def __init__(self, symbol: str ="AAPL", ticksize: float =0.01, numOrdersPerLevel: int =10, Arrival_model: ArrivalModel=None, agents: List[TradingAgent]=None):
        super().__init__(type="Exchange", seed=1, log_events=True, log_to_file=False)
        self.levels = ["Ask_deep", "Ask_touch", "Bid_touch", "Bid_deep"]
        self.cols=["lo_deep_Ask", "co_deep_Ask", "lo_top_Ask","co_top_Ask", "mo_Ask", "lo_inspread_Ask" ,
            "lo_inspread_Bid" , "mo_Bid", "co_top_Bid", "lo_top_Bid", "co_deep_Bid","lo_deep_Bid" ]
        
        
        
        
        self.LOBhistory=[] #History of an order book is an array of the form (t, LOB, spread)
        self.numOrdersPerLevel=numOrdersPerLevel
        self.colsToLevels = {
            "lo_deep_Ask" : "Ask_deep",
            "lo_top_Ask" : "Ask_touch",
            "lo_top_Bid" : "Bid_touch",
            "lo_deep_Bid" : "Bid_deep",
            "co_deep_Ask" : "Ask_deep",
            "co_top_Ask" : "Ask_touch",
            "co_top_Bid" : "Bid_touch",
            "co_deep_Bid" : "Bid_deep"
        }
        #init prices
        self.ticksize=ticksize
        if not Arrival_model:
            raise ValueError("Please specify Arrival_model for Exchange")
        else:
            logger.debug(f"Arrival_model of Exchange specified as {self.Arrival_model.__class__.__name__}")
            self.Arrival_model=Arrival_model
        if not agents:
            raise ValueError("Please provide a list of Trading Agents")
        else:
            self.agentIDs=[j.id for j in agents]
            self.agents={j.id: j for j in agents}
            self.agentcount=len(agents)
            #Keep a registry of which agents are linked to this exchange
        self.kernel=None
        self.symbol=symbol
        
        
    
    def initialize_exchange(self, priceMid0: int=20, spread0: int =4): 
        """
        The initialize_exchange method is called by the simulation kernel. By this time, the exchange should be linked to the kernel.
        """
        if self.kernel is None:
            raise ValueError("Exchange is initialized but is not linked to a simulation kernel")
        #Initialize prices
        self.priceMid=priceMid0
        self.spread=spread0
        self.askprices={ 
            "Ask_touch" : self.priceMid + np.floor(self.spread/2)*self.ticksize,
            "Ask_deep" : self.priceMid + np.floor(self.spread/2)*self.ticksize + self.ticksize
        }
        self.bidprices={
            "Bid_touch" : self.priceMid - np.ceil(self.spread/2)*self.ticksize,
            "Bid_deep" : self.priceMid - np.ceil(self.spread/2)*self.ticksize - self.ticksize
        }
        self.bids={} #holding list of pricelevels which are in the form of (key, value)=(price, [list of orders])
        self.asks={} 
        #Initialize order sizes
        for k in self.levels:
            queue=self.Arrival_model.generate_orders_in_queue(loblevel=k, numorders=self.numOrdersPerLevel)
            if "Ask" in k:
                self.asks[self.askprices[k]]=queue
            elif "Bid" in k:
                self.asks[self.bidprices[k]]=queue
            else:
                raise AssertionError(f"Level is not an ASK or BID string")
        logger.debug("Stock Exchange initalized at time")
        #Send a message to agents to begin trading
        message=BeginTradingMsg(time=self.current_time)
        self.sendbatchmessage(recipientIDs=self.agentIDs, message=message, current_time=self.current_time)
        
    
            

    def generate_ordersize(self, loblevel):
        """
        generate ordersize
        loblevel: one of the 4 possible levels in the LOB as a string
        Returns: qsize the size of the order
        """
        #use generaeordersize method in hawkes arrival
        if self.Arrival_model is not None:
            return self.Arrival_model.generate_ordersize(loblevel)
        else:
            raise ValueError("Arrival_model is not provided")
        
    
    def nextarrival(self):
        """
        Automatically generates the next arrival and passes the order to the kernel to check for validity.
        """
        t, k, s=self.Arrival_model.get_nextarrival()
        if k<0 or k>11:
            raise IndexError(f"Event index is out of bounds. Expected 0<=k<=11, but received k={k}")
        if "Ask" in self.cols[k]:
            side="Ask"
        else:
            side="Bid"
        if self.cols[k][0:2]=="lo":
            if k==5: #inspread ask
                price=self.askprices["Ask_touch"]+self.ticksize
            elif k==6: #inspread bid
                price=self.bidprices["Bid_touch"]+self.ticksize
            else:    
                level=self.colsToLevels[self.cols[k]]
                if side=="Ask":
                    price=self.askprices[level]
                else:
                    price=self.bidprices[level]
            order=LimitOrder(time_placed=t, event_type=k, side=side, size=s, symbol=self.symbol, agent_id=-1, price=price, loblevel=level)
            
        elif self.cols[k][0:2]=="mo":
            #marketorder
            order=MarketOrder(time_placed=t, side=side,  event_type=k, size=s, symbol=self.symbol, agent_id=-1)
        else:
            #cancelorder
            level=self.colsToLevels[self.cols[k]]
            if side=="Ask":
                price=self.askprices[level]
            else:
                price=self.bidprices[level]
            order=CancelOrder(time_placed=t,  event_type=k, side=side, size=-1, symbol=self.symbol, agent_id=-1, loblevel=level)
        
        message=OrderPending(order=order)
        self.sendmessage(recipientID=-1, message=message)
        



    def processorder(self, order: Order):
        """
        Called by the kernel: Takes an order in the form (t, k, s) and processes it in the limit order book, performing the correct matching as necessary. It also updates the arrival_model history
        """
        self.current_time=order.time_placed
        #update the modelhistory
        self.update_model_state(t=order.time_placed, k=order.event_type, s=order.size)
        #process the order
        trade_happened=False
        side=order.side
        if isinstance(order, LimitOrder):
            #Limit_order
            if "deep" in order.loblevel:
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
            elif "top" in order.loblevel:
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
        elif isinstance(order, MarketOrder): #market order
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

        elif isinstance(order, CancelOrder): #Cancel Order completed
            if order.agent_id==-1:
                public_cancel_flag=True
            else:
                public_cancel_flag=False
                #test if it's a valid agent_id
                if order.agent_id in self.agentIDs:
                    pass
                else:
                    raise AgentNotFoundError(f"Agent ID {order.agent_ID} is not listed in Exchange book")
            queue=[]        
            if order.side=="Ask":
                queue=self.asks[order.price]
            else:
                queue=self.bids[order.price]
            validcancels=[]
            if public_cancel_flag==True:
                validcancels=[item for item in queue if item.agent_id==-1]
                if len(validcancels)==0:
                    logger.info(f"Random cancel order issued, but only existing orders in queue are private agent orders so cancel order at time {order.time_placed} ignored.")  
                else:
                    cancelled=validcancels.pop(random.randrange(0, len(validcancels)))
                    queue=[item for item in queue if item.order_id != cancelled.order_id]
            else: #Cancel order from an agent
                queue=[item for item in queue if item.order_id!=order.cancelID]
                assert len(queue)>0, f"Agent {order.agent_id} attemptd to place cancel orders wihout pre-existing limit orders in the book at the same price"
            #non-empty queue for order cancellation, public and private           
            if order.side=="Ask":
                self.asks[order.price]=queue
            else:
                self.bids[order.price]=queue
            order.cancelled=True
            if public_cancel_flag==False:
                notif=OrderExecutedMsg(order=order)
                self.sendmessage(recipientID=order.agent_id, message=notif)
            self.regeneratequeuedepletion()
        else:
            raise InvalidOrderType(f"Invalid Order type with ID {order.order_id} passed to exchange")
            pass
        #log event:
        self._logevent(event=[order.current_time, order.ordertype(), order.agent_id, order.order_id])
        #update spread and notify agents if a trade has happened:
        newspread=abs(self.askprices["Ask_touch"]-self.bidprices["Bid_touch"])
        if self.spread==newspread:
            pass
        if trade_happened==False:
            pass
        if self.spread!=newspread:
            
            self.spread=newspread
            #Send Batch message to agents
            message=SpreadNotificationMsg()
            self.sendbatchmessage(recipientIDs=self.agentIDs, message=message)
        else:
            if trade_happened==True:
                message=TradeNotificationMsg()
                self.sendbatchmessage(recipientIDs=self.agentIDs, message=message)
        self.updatehistory()
    
    
    def regeneratequeuedepletion(self): #to be implemented
        """
        Regenerates LOB from queue depletion and updates prices as necessary
        """
        if len(self.asks[self.askprices["Ask_touch"]])==0:
            del self.asks[self.askprices["Ask_touch"]]
            self.askprices["Ask_touch"]=self.askprices["Ask_deep"]
            self.askprices["Ask_deep"]=self.askprices["Ask_touch"]+self.ticksize
            self.asks[self.askprices["Ask_deep"]]=self.Arrival_model.generate_orders_in_queue(loblevel="Ask_deep", numorders=self.numOrdersPerLevel)
            
            
        elif len(self.bids[self.bidprices["Bid_touch"]])==0:
            del self.bids[self.bidprices["Bid_touch"]]
            self.bidprices["Bid_touch"]=self.bidprices["Bid_deep"]
            self.bidprices["Bid_deep"]=self.bidprices["Bid_touch"]-self.ticksize
            self.bids[self.bidprices["Bid_deep"]]=self.Arrival_model.generate_orders_in_queue(loblevel="Bid_deep", numorders=self.numOrdersPerLevel)
        
        elif len(self.asks[self.askprices["Ask_deep"]])==0:
            del self.asks[self.askprices["Ask_deep"]]
            self.askprices["Ask_deep"]=self.askprices["Ask_touch"]+self.ticksize
            self.asks[self.askprices["Ask_deep"]]=self.Arrival_model.generate_orders_in_queue(loblevel="Ask_deep", numorders=self.numOrdersPerLevel)
        
        elif len(self.bids[self.bidprices["Bid_deep"]])==0:
            del self.bids[self.bidprices["Bid_deep"]]
            self.bidprices["Bid_deep"]=self.bidprices["Bid_touch"]-self.ticksize
            self.bids[self.bidprices["Bid_deep"]]=self.Arrival_model.generate_orders_in_queue(loblevel="Bid_deep", numorders=self.numOrdersPerLevel)
        else:
            #queue is not depleted
            pass
    
    #information getters and setters
    def lob0(self)-> dict: 
        rtn={
            "Ask_deep": (self.askprices["Ask_deep"], sum(self.asks[self.askprices["Ask_deep"]])),
            "Ask_touch": (self.askprices["Ask_touch"], sum(self.asks[self.askprices["Ask_touch"]])),
            "Bid_touch": (self.bidprices["Bid_touch"], sum(self.bids[self.bidprices["Bid_touch"]])),
            "Bid_deep": (self.bidprices["Bid_touch"], sum(self.bids[self.bidprices["Bid_touch"]]))
        }
        return rtn

    def lobl3(self):
        rtn={
            "Ask_deep": (self.askprices["Ask_deep"], self.asks[self.askprices["Ask_deep"]]),
            "Ask_touch": (self.askprices["Ask_touch"], self.asks[self.askprices["Ask_touch"]]),
            "Bid_touch": (self.bidprices["Bid_touch"], self.bids[self.bidprices["Bid_touch"]]),
            "Bid_deep": (self.bidprices["Bid_touch"], self.bids[self.bidprices["Bid_touch"]])
        }
        return rtn

    def getlob(self):
        return [[self.askprices["Ask_deep"], self.asks[self.askprices["Ask_deep"]]], [self.askprices["Ask_touch"], self.asks[self.askprices["Ask_touch"]]], [self.bidprices["Bid_touch"], self.bids[self.bidprices["Bid_touch"]]], [self.bidprices["Bid_deep"], self.bids[self.bidprices["Bid_deep"]]], self.spread]
        
    def updatehistory(self):
        #data=[order book, spread]
        data=(self.current_time, self.getlob(), self.spread)
        self.LOBhistory.append(data)

    def update_model_state(self, s: float, k: int):
        """
        Adds a point to the arrival model, updates the spread
        """
        self.Arrival_model.update(s, k)
        