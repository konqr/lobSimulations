import pprint
import numpy as np
import pandas as pd
import time
import numpy as np
import os 
import random
import logging
from typing import Any, List, Optional, Tuple, ClassVar
from HawkesRLTrading.src.SimulationEntities.Entity import Entity
from HawkesRLTrading.src.Utils.Exceptions import *
from HawkesRLTrading.src.SimulationEntities.TradingAgent import TradingAgent
from HawkesRLTrading.src.Stochastic_Processes.Arrival_Models import ArrivalModel, HawkesArrival
from HawkesRLTrading.src.Orders import *
from HawkesRLTrading.src.Messages.ExchangeMessages import *
logging.basicConfig()
logging.getLogger(__name__).setLevel(logging.INFO)
logger=logging.getLogger(__name__)
class Exchange(Entity):
    """Arguments:
        symbol
        ticksize
        LOBlevels
        numOrdersPerLevel"""
    def __init__(self, seed: int=1, log_events=True, log_to_file=True, **kwargs):
        
        super().__init__(type="Exchange", seed=seed, log_events=log_events, log_to_file=log_to_file)
        #symbol: str ="XYZ", ticksize: float =0.01, LOBlevels: int=2, numOrdersPerLevel: int =10, 
        defaults={  "symbol": "XYZ",
                    "ticksize": 0.01,
                    "LOBlevels": 2,
                    "numOrdersPerLevel": 10,
                    "PriceMid0": 45,
                    "spread0": 0.05
                    }
        defaults.update(kwargs)
        self.ticksize=np.round(defaults["ticksize"], 2)
        self.LOBlevels=defaults["LOBlevels"]
        self.symbol=defaults["symbol"]
        self.numOrdersPerLevel=defaults["numOrdersPerLevel"]
        self.priceMid=defaults["PriceMid0"]
        self._spread=defaults["spread0"]
        self.levels = [f"Ask_L{i}" for i in range(1, self.LOBlevels + 1)]+[f"Bid_L{i}" for i in range(1, self.LOBlevels + 1)]
        self.LOBhistory=[] #History of an order book is an array of the form (t, LOB, spread)
        self.log: List[Any]=[] #time, type, agentID, orderID
        #Empty Attributes:
        self.kernel=None
        self.agentcount=None
        self.agentIDs=None
        self.agents=None
        self.agentswithTradeNotifs=None
        self.Arrival_model=None
        self.askprices={}
        self.bidprices={}
        self.bids={} #holding list of pricelevels which are in the form of (key, value)=(price, [list of orders])
        self.asks={}
        self.askprice=None
        self.bidprice=None
        
        #Initialize prices
        for i in range(self.LOBlevels):
            key="Ask_L"+str(i+1)
            val=np.round(self.priceMid+np.floor((self._spread//self.ticksize)/2)*self.ticksize + i*self.ticksize, 2)
            self.askprices[key]=val
            key="Bid_L"+str(i+1)
            val=np.round(self.priceMid-np.floor((self._spread//self.ticksize)/2)*self.ticksize - i*self.ticksize, 2)
            self.bidprices[key]=val   
        self.askprice=self.askprices["Ask_L1"]
        self.bidprice=self.bidprices["Bid_L1"]
        logger.debug(f"Exchange prices initialized -- Askprice: {self.askprice}, Bidprice: {self.bidprice}, Spread: {self._spread}, MidPrice: {self.priceMid}")
        self._spread=abs(self.askprice-self.bidprice)
    
    def initialize_exchange(self, agents: Optional[List[TradingAgent]]=None, kernel=None, Arrival_model: ArrivalModel=None):
        """
        The initialize_exchange method is called by the simulation kernel.
        """
        if kernel is None:
            raise ValueError("Exchange is initialized but is not linked to a simulation kernel")
        else:
            self.kernel=kernel
            if type(kernel).__name__ =="Kernel":
                pass
            else:
                raise TypeError(f"Expected Kernel object to be passed into kernel parameter, received {type(kernel).__name__}")
        if agents is None:
            raise ValueError("Please provide a list of Trading Agents")
        else:
            for agent in agents:
                if isinstance(agent, TradingAgent):
                    pass
                else:
                    raise TypeError(f"Expected list of TradingAgent objects, received {type(agent).__name__}")
            self.agentIDs=[j.id for j in agents]
            self.agents={j.id: j for j in agents}
            self.agentcount=len(self.agentIDs)
            #Keep a registry of which agents have a subscription to trade notifs
            self.agentswithTradeNotifs=[j.id for j in agents if j.tradeNotif==True]
            self.agentswithTradeOnMO=[id for id in self.agentswithTradeNotifs if self.agents[id].wake_on_MO]
            self.agentswithTradeOnSpread=[id for id in self.agentswithTradeNotifs if self.agents[id].wake_on_Spread]
            print(f"Agent with tradenotifs: {self.agentswithTradeNotifs}")
        if Arrival_model is None:
            raise ValueError("Please specify Arrival_model for Exchange")
        else:
            logger.debug(f"Arrival_model of Exchange specified as {type(Arrival_model).__name__}")
            self.Arrival_model: ArrivalModel=Arrival_model
            #Check that Arrival model spread matches exchange spread
            assert self.Arrival_model.spread==self._spread, "Arrival model spread is not the same as Exchange spread"
        
        

        #Initialize order sizes
        for k in self.levels:
            queue=self.generate_orders_in_queue(loblevel=k)
            if "Ask" in k:
                self.asks[self.askprices[k]]=queue
            elif "Bid" in k:
                self.bids[self.bidprices[k]]=queue
            else:
                raise AssertionError(f"Level is not an ASK or BID string")
        logger.debug("Stock Exchange initialized")
        #Send a message to agents to Tr trading
                
    def generate_orders_in_queue(self, loblevel) -> List[LimitOrder]:
        assert self.Arrival_model is not None, "Arrival_model is not provided"
        queue=self.Arrival_model.generate_orders_in_queue(loblevel=loblevel, numorders=self.numOrdersPerLevel)
        side=loblevel[0:3]
        price=0
        if side=="Ask":
            price=self.askprices[loblevel]
        else:
            price=self.bidprices[loblevel]
        orderqueue=[]
        for size in queue:
            order=LimitOrder(time_placed=self.current_time, price=price, side=side, size=size, symbol=self.symbol, agent_id=-1)
            orderqueue.append(order)
        return orderqueue
             

    def nextarrival(self, timelimit=None) -> Optional[bool]:
        """
        Automatically generates the next arrival and passes the order to the kernel to check for validity.
        """
        order=None
        self.Arrival_model.spread = self.askprice - self.bidprice
        tmp=self.Arrival_model.get_nextarrival(timelimit=timelimit)
        if tmp is None:
            logger.debug("Simulation candidate rejected, no event happening")
            return None
        else:
            time, side, order_type, level, size=tmp
        if order_type=="lo":
            if level=="Ask_inspread": #inspread ask
                price=np.round(self.askprices["Ask_L1"]-self.ticksize, 2)
            elif level=="Bid_inspread": #inspread bid
                price=np.round(self.bidprices["Bid_L1"]+self.ticksize, 2)
            else:    
                if side=="Ask":
                    price=self.askprices[level]
                else:
                    price=self.bidprices[level]
            order=LimitOrder(time_placed=time, side=side, size=size, symbol=self.symbol, agent_id=-1, price=price)
            
        elif order_type=="mo":
            #marketordernon-agent placed
            order=MarketOrder(time_placed=time, side=side, size=size, symbol=self.symbol, agent_id=-1)
        else:
            #cancelorder
            if side=="Ask":
                price=self.askprices[level]
            else:
                price=self.bidprices[level]
            order=CancelOrder(time_placed=time, side=side, price=price, size=-1, symbol=self.symbol, agent_id=-1)
        order._level=level
        logger.debug(f"\nNon-Agent placed: {order}")
        self.processorder(order=order)
        return True
        
        
    def processorder(self, order: Order):
        """
        Called by the kernel: Takes an order in the form (t, k, s) and processes it in the limit order book, performing the correct matching as necessary. It also updates the arrival_model history
        """
        #print("Before processing order: \n")
        #print(self.printlob())
        logger.debug(f"Processing Order {order.order_id}\n")
        assert self.current_time<=order.time_placed,  f"Order {order.order_id} time placed: {order.time_placed} but exchange is in the future at {self.current_time}"
        self.current_time=order.time_placed
        #process the order
        order_type=None
        trade_happened=False
        if isinstance(order, LimitOrder):
            #Limit_order
            self.processLimitOrder(order=order)
            order_type="lo"
            if order.agent_id!=-1:
                notif=LimitOrderAcceptedMsg(order=order)
                self.sendmessage(recipientID=order.agent_id, message=notif)
        elif isinstance(order, MarketOrder): #market order
            totalvalue=self.processMarketOrder(order)
            order.total_value=totalvalue
            if order.agent_id==-1:
                if len(self.agentswithTradeOnMO)>0:
                    tmp=TradeNotificationMsg(time=self.current_time)
                    self.sendbatchmessage(recipientIDs=self.agentswithTradeOnMO, message=tmp)
            else:
                #Agent doesn't get a chance to trade again
                recipientIDs=[j for j in self.agentswithTradeOnMO if j!=order.agent_id]
                if len(recipientIDs)>0:
                    tmp=TradeNotificationMsg(time=self.current_time)
                    self.sendbatchmessage(recipientIDs=recipientIDs, message=tmp)
                notif=OrderExecutedMsg(order=order)
                self.sendmessage(recipientID=order.agent_id, message=notif)
            trade_happened=True
            order_type="mo"
        elif isinstance(order, CancelOrder): #Cancel Order completed
            self.processCancelOrder(order=order)
            order_type="co"
            if order.agent_id!=-1:
                notif=OrderExecutedMsg(order=order)
                self.sendmessage(recipientID=order.agent_id, message=notif)
        else:
            raise InvalidOrderType(f"Invalid Order type with ID {order.order_id} passed to exchange")
            pass
        #print("After processing order: \n")
        #print(self.printlob(), "\n")
        if order.agent_id==-1:
            pass
        else:
            print(f"Information for model update: {order}")
            self.update_model_state(order=order, order_type=order_type)
        
        #update spread and notify agents if a trade has happened:
        newspread=abs(self.askprice-self.bidprice)
        if np.round(self._spread, 2)==np.round(newspread, 2):
            pass
        if np.round(self._spread, 2)!=np.round(newspread, 2):
            self._spread=np.round(newspread, 2)
            #Send Batch message to agents
            if trade_happened==True:
                pass
            else:
                if order.agent_id==-1:
                    if len(self.agentswithTradeOnSpread)>0:
                        message=SpreadNotificationMsg(time=self.current_time)
                        self.sendbatchmessage(recipientIDs=self.agentswithTradeOnSpread, message=message)
                else:
                    message=SpreadNotificationMsg(time=self.current_time)
                    recipientIDs=[j for j in self.agentswithTradeOnSpread if j!=order.agent_id]
                    if len(recipientIDs)>0:
                        self.sendbatchmessage(recipientIDs=recipientIDs, message=message)
        
        test=self.checkLOBValidity()
        if test:
            logger.debug("LOB Valid test passed")
            pass
        else:
            # print(self.asks)
            # print(self.bids)
            # print(self.askprices)
            # print(self.bidprices)
            raise LOBProcessingError("LOB in Exchange processed incorrectly")
        #log event:
        self._logevent(event=[order.time_placed, order.ordertype(), order.agent_id, order.order_id])
        self.updatehistory()

    
    def processLimitOrder(self, order: LimitOrder):
        #Limit_order
        side=order.side
        price=order.price
        queue=None
        if side=="Ask":
            queue=self.asks.get(price)
        else:
            queue=self.bids.get(price)
        if queue is not None:
            queue.append(order)
        else: #Inspread
            lastlvl=side+"_L"+str(self.LOBlevels)
            todelete=[]
            crossed=False
            if side=="Ask":

                if order.price==np.round(self.bidprice, 2): #Check if order book is crossed
                    crossed=True
                    logger.debug("Orderbook entering crossed state")
                else:
                    if order.price==np.round(self.askprice-self.ticksize, 2): #Check the price is correct for an ask_inspread order
                        pass
                    else:
                        raise InvalidOrderType(f"Order {order.order_id} is an invalid inspread limit order")
                
                todelete=self.asks[self.askprices[lastlvl]]
            else:
                if order.price==np.round(self.askprice, 2): #Check if order book is crossed
                    crossed=True
                    logger.debug("Orderbook entering crossed state")
                else:
                    if order.price==np.round(self.bidprice+self.ticksize, 2): #Check the price is correct for a bid_inspread order
                        pass
                    else:
                        logger.debug(f"Order price: {order.price}, bidprice: {self.bidprice}\n")
                        #print(self.lob0)

                        raise InvalidOrderType(f"Order {order.order_id} is an invalid inspread limit order")
                todelete=self.bids[self.bidprices[lastlvl]]
            agentorders=[order for order in todelete if order.agent_id!=-1]
            if len(agentorders)>0:
                self.autocancel(agentorders)
            if side=="Ask":
                del self.asks[self.askprices[lastlvl]]
                del self.askprices[lastlvl]
                self.askprice=order.price
                new_askprices = {}
                new_asks = {}  
                new_askprices["Ask_L1"]=self.askprice
                new_asks[self.askprice]=[order]
                for i in range(2, self.LOBlevels + 1):
                    new_key= f"Ask_L{i}"
                    old_key=f"Ask_L{i-1}"
                    new_askprices[new_key]=self.askprices[old_key]
                    new_asks[new_askprices[new_key]]=self.asks[new_askprices[new_key]]
                self.askprices = new_askprices
                self.asks = new_asks  
            else:
                del self.bids[self.bidprices[lastlvl]]
                del self.bidprices[lastlvl]
                self.bidprice=order.price
                new_bidprices = {}
                new_bids = {}  
                new_bidprices["Bid_L1"]=self.bidprice
                new_bids[self.bidprice]=[order]
                for i in range(2, self.LOBlevels + 1):
                    new_key= f"Bid_L{i}"
                    old_key=f"Bid_L{i-1}"
                    new_bidprices[new_key]=self.bidprices[old_key]
                    new_bids[new_bidprices[new_key]]=self.bids[new_bidprices[new_key]]
                self.bids=new_bids
                self.bidprices = new_bidprices      
                logger.debug(f"Post inspread bids: {self.bids}")
            if crossed:
                #Resolving crossed orderbook state
                self.resolve_crossedorderbook()
        
    def processMarketOrder(self, order: MarketOrder)-> int:
        side=order.side
        remainingsize=order.size
        totalvalue=0
        if side=="Ask":
            side="Bid"
            level=1
            while remainingsize>0.5:
                pricelvl=self.bidprices[side+"_L"+str(level)]
                while len(self.bids[pricelvl])>0:
                    item: LimitOrder=self.bids[pricelvl][0]
                    logger.debug(f"\nItem size: {item.size}")
                    if remainingsize<item.size:
                        consumed=remainingsize
                        totalvalue+=pricelvl*remainingsize
                        item.size=item.size-remainingsize
                        remainingsize=0
                        if item.agent_id==-1:
                            pass
                        else:
                            notif=PartialOrderFill(order=item, consumed=consumed)
                            self.sendmessage(recipientID=item.agent_id, message=notif)
                        return totalvalue
                    else:
                        filled_order: Order=self.bids[pricelvl].pop(0)
                        remainingsize-=filled_order.size
                        totalvalue+=filled_order.size*pricelvl
                        filled_order.fill_time=self.current_time
                        filled_order.filled=True
                        if filled_order.agent_id==-1:
                            pass
                        else:
                            notif=OrderExecutedMsg(order=filled_order)
                            self.sendmessage(recipientID=filled_order.agent_id, message=notif)
                #Touch level has been depleted
                del self.bids[pricelvl]
                del self.bidprices["Bid_L1"]
                new_bidprices = {}
                new_bids = {}  
                for i in range(1, self.LOBlevels):
                    new_key= f"Bid_L{i}"
                    old_key=f"Bid_L{i+1}"
                    new_bidprices[new_key]=self.bidprices[old_key]
                    new_bids[new_bidprices[new_key]]=self.bids[new_bidprices[new_key]]
                new_bidprices[f"Bid_L{self.LOBlevels}"]=np.round(new_bidprices[f"Bid_L{self.LOBlevels-1}"] - self.ticksize, 2)
                new_bids[new_bidprices[f"Bid_L{self.LOBlevels}"]]=self.generate_orders_in_queue(loblevel=f"Bid_L{self.LOBlevels}")
                self.bidprices = new_bidprices
                self.bids = new_bids  
                self.bidprice=self.bidprices["Bid_L1"]
        else:
            side="Ask"
            level=1
            while remainingsize>0.5:
                pricelvl=self.askprices[side+"_L"+str(level)]
                logger.debug(pricelvl)
                while len(self.asks[pricelvl])>0:
                    item: LimitOrder=self.asks[pricelvl][0]
                    logger.debug(f"Remaining size: {remainingsize}, item size: {item.size}")
                    if remainingsize<item.size:
                        consumed=remainingsize
                        item.size=item.size-remainingsize
                        totalvalue+=pricelvl*remainingsize
                        remainingsize=0
                        if item.agent_id==-1:
                            pass
                        else:
                            notif=PartialOrderFill(order=item, consumed=consumed)
                            self.sendmessage(recipientID=item.agent_id, message=notif)
                        return totalvalue
                    else:
                        filled_order: Order=self.asks[pricelvl].pop(0)
                        #print(f"Filled Order: {filled_order}")
                        remainingsize-=filled_order.size
                        totalvalue+=filled_order.size*pricelvl
                        #print(f"totalvalue: {totalvalue}")
                        filled_order.fill_time=self.current_time
                        filled_order.filled=True
                        if filled_order.agent_id==-1:
                            pass
                        else:
                            notif=OrderExecutedMsg(order=filled_order)
                            self.sendmessage(recipientID=filled_order.agent_id, message=notif)
                del self.asks[pricelvl]
                del self.askprices["Ask_L1"]
                new_askprices = {}
                new_asks = {}  
                for i in range(1, self.LOBlevels):
                    new_key= f"Ask_L{i}"
                    old_key=f"Ask_L{i+1}"
                    new_askprices[new_key]=self.askprices[old_key]
                    new_asks[new_askprices[new_key]]=self.asks[new_askprices[new_key]]
                new_askprices[f"Ask_L{self.LOBlevels}"]=np.round(new_askprices[f"Ask_L{self.LOBlevels-1}"] + self.ticksize, 2)
                new_asks[new_askprices[f"Ask_L{self.LOBlevels}"]]=self.generate_orders_in_queue(loblevel=f"Ask_L{self.LOBlevels}")
                self.askprices = new_askprices
                self.asks = new_asks  
                self.askprice=self.askprices["Ask_L1"]
        return totalvalue
    def processCancelOrder(self, order: CancelOrder):
        if order.agent_id==-1:
            public_cancel_flag=True
        else:
            public_cancel_flag=False
            #test if it's a valid agent_id
            if order.agent_id in self.agentIDs:
                pass
            else:
                raise AgentNotFoundError(f"Agent ID {order.agent_ID} is not listed in Exchange book")
            #Check that the cancel order came from the same agent who placed the order
        side=order.side
        price=order.price
        queue=[]        
        if side=="Ask":
            queue=self.asks[price]
        else:
            queue=self.bids[price]
        validcancels=[]
        cancelled=None
        if public_cancel_flag==True:
            validcancels=[item for item in queue if item.agent_id==-1]
        else: #Cancel order from an agent
            validcancels=[item for item in queue if item.agent_id==order.agent_id]
        if len(validcancels)==0:
            logger.info(f"Cancel order issued, but no eligible orders in queue at time {order.time_placed} ignored.")  
            return
        cancelled: LimitOrder=validcancels.pop(random.randrange(0, len(validcancels)))
        order.cancelID=cancelled.order_id
        queue=[item for item in queue if item.order_id != cancelled.order_id]
    #non-empty queue for order cancellation, public and private           
        if order.side=="Ask":
            self.asks[price]=queue
        else:
            self.bids[price]=queue
        cancelled.cancelled=True
        self.regeneratequeuedepletion()   

    def autocancel(self, orders: List[Order]):
        logger.info(f"Autocancelling agent orders{[j.order_id for j in orders]} due to LOB inspread shift")
        for order in orders:
            if isinstance(order, LimitOrder):
                pass
            else:
                raise InvalidOrderType(f"Expected Limit Order for autocancelling, received {order.ordertype()}")
            price=order.price
            side=order.side
            order.cancelled=True
            order.cancel_time=self.current_time
            oldqueue=None
            if side=="Ask":
                oldqueue=self.asks[price]
            else:
                oldqueue=self.bids[price]
            queue=[j for j in oldqueue if j.order_id!=order.order_id]
            oldqueue=queue
            
            notif=OrderAutoCancelledMsg(order=order)
            self.sendmessage(recipientID=order.agent_id, message=notif)
    
    def resolve_crossedorderbook(self):
        assert self.askprice==self.bidprice==self.askprices["Ask_L1"]==self.bidprices["Bid_L1"], f'Crossed Orderbook: ASKPRICE: {self.askprice}, BIDPRICE: {self.bidprice}, ASK_L1{self.askprices["Ask_L1"]}, BID_L1{self.bidprices["Bid_L1"]}'
        #totalask1=sum([j.size for j in self.asks[self.askprices["Ask_L1"]]])
        #totalbid1=sum([j.size for j in self.bids[self.bidprices["Bid_L1"]]])
        ask_q=self.asks[self.askprices[self.askprice]]
        bid_q=self.asks[self.bidprices[self.bidprice]]
        while len(ask_q)>0 and len(bid_q)>0:
            bid_order=bid_q[0]
            ask_order=ask_q[0]
            executionsize=min(bid_order.size, ask_order.size)
            bid_order.size-=executionsize
            ask_order.size-=executionsize
            if bid_order.size==0:
                bid_order.size
                filled_order=bid_q.pop(0)
                filled_order.fill_time=self.current_time
                filled_order.filled=True
                if filled_order.agent_id==-1:
                    pass
                else:
                    notif=OrderExecutedMsg(order=filled_order)
                    self.sendmessage(recipientID=filled_order.agent_id, message=notif)    
            elif bid_order.size>0:
                assert ask_order.size==0, "Error in order matching at crossed orderbook"
                consumed=executionsize
                notif=PartialOrderFill(order=bid_order, consumed=consumed)
                self.sendmessage(recipientID= bid_order.agent_id, message=notif)
            else:
                logger.debug("Code should be unreachable")
            if ask_order.size==0:
                filled_order=ask_q.pop(0)
                filled_order.fill_time=self.current_time
                filled_order.filled=True
                if filled_order.agent_id==-1:
                    pass
                else:
                    notif=OrderExecutedMsg(order=filled_order)
                    self.sendmessage(recipientID=filled_order.agent_id, message=notif)
            elif ask_order.size>0:
                assert bid_order.size==0, "Error in order matching at crossed orderbook"
                consumed=executionsize
                notif=PartialOrderFill(order=ask_order, consumed=consumed)
                self.sendmessage(recipientID=ask_order.agent_id, message=notif)
            else:
                logger.debug("Code should be unreachable")
            
                
        if len(bid_q)==0:
            del self.bids[self.bidprice]
            del self.bidprices["Bid_L1"]
        if len(ask_q)==0:
            del self.asks[self.askprice]
            del self.askprices["Ask_L1"]
        self.regeneratequeuedepletion()
        assert self.askprice<self.bidprice, "Error in processing crossed orderbook"




            


    def regeneratequeuedepletion(self): #to be implemented
        """
        Regenerates LOB from queue depletion and updates prices as necessary
        """
        if len(self.asks[self.askprices["Ask_L1"]])==0:
            del self.asks[self.askprices["Ask_L1"]]
            self.askprices["Ask_L1"]=self.askprices["Ask_L2"]
            self.askprices["Ask_L2"]=np.round(self.askprices["Ask_L1"]+self.ticksize, 2)
            self.asks[self.askprices["Ask_L2"]]=self.generate_orders_in_queue(loblevel="Ask_L2")
            self.askprice=self.askprices["Ask_L1"]
        elif len(self.asks[self.askprices["Ask_L2"]])==0:
            del self.asks[self.askprices["Ask_L2"]]
            self.askprices["Ask_L2"]=np.round(self.askprices["Ask_L1"]+self.ticksize, 2)
            self.asks[self.askprices["Ask_L2"]]=self.generate_orders_in_queue(loblevel="Ask_L2")

        if len(self.bids[self.bidprices["Bid_L1"]])==0:
            del self.bids[self.bidprices["Bid_L1"]]
            self.bidprices["Bid_L1"]=self.bidprices["Bid_L2"]
            self.bidprices["Bid_L2"]=np.round(self.bidprices["Bid_L1"]-self.ticksize, 2)
            self.bids[self.bidprices["Bid_L2"]]=self.generate_orders_in_queue(loblevel="Bid_L2")
            self.bidprice=[self.bidprices["Bid_L1"]]
        elif len(self.bids[self.bidprices["Bid_L2"]])==0:
            del self.bids[self.bidprices["Bid_L2"]]
            self.bidprices["Bid_L2"]=np.round(self.bidprices["Bid_L1"]-self.ticksize, 2)
            self.bids[self.bidprices["Bid_L2"]]=self.generate_orders_in_queue(loblevel="Bid_L2")
        else:
            #queue is not depleted
            pass
    def checkLOBValidity(self) -> bool:
        condition1= (self.askprice==self.askprices["Ask_L1"])
        condition2=(self.askprice==min(self.asks.keys()) )     
        condition3=(self.bidprice==self.bidprices["Bid_L1"])
        condition4=(self.bidprice==max(self.bids.keys()))
        condition5=(self.spread==np.round(abs(self.askprice-self.bidprice)), 2)
        return condition1 and condition2 and condition3 and condition4 and condition5
        
        
    #information getters and setters
    @property
    def lob0(self)-> dict: 
        rtn={
            "Ask_L2": (self.askprices["Ask_L2"], sum([j.size for j in self.asks[self.askprices["Ask_L2"]]])),
            "Ask_L1": (self.askprices["Ask_L1"], sum([j.size for j in self.asks[self.askprices["Ask_L1"]]])),
            "Bid_L1": (self.bidprices["Bid_L1"], sum([j.size for j in self.bids[self.bidprices["Bid_L1"]]])),
            "Bid_L2": (self.bidprices["Bid_L2"], sum([j.size for j in self.bids[self.bidprices["Bid_L2"]]]))
        }
        return rtn
    @property
    def lobl3(self):
        rtn={
            "Ask_L2": (self.askprices["Ask_L2"], self.asks[self.askprices["Ask_L2"]]),
            "Ask_L1": (self.askprices["Ask_L1"], self.asks[self.askprices["Ask_L1"]]),
            "Bid_L1": (self.bidprices["Bid_L1"], self.bids[self.bidprices["Bid_L1"]]),
            "Bid_L2": (self.bidprices["Bid_L2"], self.bids[self.bidprices["Bid_L2"]])
        }
        return rtn
    
    def printlob(self):
        rtn="LOB: "
        for i in range(self.LOBlevels, 0, -1):
            rtn+="\n"
            string = f"Ask_L{i}: {self.askprices[f'Ask_L{i}']}, ({sum([j.size for j in self.asks[self.askprices[f'Ask_L{i}']]])}, {[j.size for j in self.asks[self.askprices[f'Ask_L{i}']]]})"

            rtn+=string
        for i in range(1, self.LOBlevels+1):
            rtn+="\n"
            string = f"Bid_L{i}: {self.bidprices[f'Bid_L{i}']}, ({sum([j.size for j in self.bids[self.bidprices[f'Bid_L{i}']]])}, {[j.size for j in self.bids[self.bidprices[f'Bid_L{i}']]]})"
            rtn+=string
        return rtn

    def returnlobl3sizes(self):
        rtn= {}
        for i in range(self.LOBlevels, 0, -1):
            rtn[f"Ask_L{i}"] = (self.askprices[f'Ask_L{i}'],
                                ([j.size for j in self.asks[self.askprices[f'Ask_L{i}']]]))

        for i in range(1, self.LOBlevels+1):
            rtn[f"Bid_L{i}"] = (self.bidprices[f'Bid_L{i}'],
                                ([j.size for j in self.bids[self.bidprices[f'Bid_L{i}']]]))
        return rtn

    def returnintensity(self):
        return self.Arrival_model.current_intensity

    def returnpasteventimes(self):
        times = np.array([])
        ts = np.array(self.Arrival_model.timeseries)
        if self.Arrival_model.timeseries is None or len(ts) == 0: return -1*np.ones(60)
        for i in range(len(self.Arrival_model.cols)):
            ts_i = ts[ts[:,1] == i, 0][-5:]
            while len(ts_i) < 5:
                ts_i = np.append(ts_i, -1)
            ts_i = np.sort(ts_i)
            times = np.append(times, ts_i)
        return times

    @property
    def spread(self):
        return self._spread
    
    def getlevel(self, price: float):
        logger.debug(f"\nGetting level for price: {price}")
        logger.debug(f"Price: {price}, askprice: {self.askprice}, bidprice: {self.bidprice}")
        if price>=self.askprice:
            return list(filter(lambda key: self.askprices[key]==price, self.askprices))[0]
        elif price <= self.bidprice:
            return list(filter(lambda key: self.bidprices[key]==price, self.bidprices))[0]
        else:
            raise Exception("Price-level not in limit order book")
    def updatehistory(self):
        #data=[order book, spread]
        data=(self.current_time, self.lobl3, self.spread)
        self.LOBhistory.append(data)

    def get_orders_from_level(self, level: str):
        assert level in self.levels, f"Level {level} is not in exchange levels"
        if level[0:3]=="Ask":
            return self.asks[self.askprices[level]]
        else:
            return self.bids[self.bidprices[level]]

    def update_model_state(self, order: Order, order_type: str):
        """
        Adds a point to the arrival model, updates the spread
        """
        time=order.time_placed
        side=order.side
        order_type=order_type
        level=order._level
        size=order.size
        self.Arrival_model.update(time=time, side=side, order_type=order_type, level=level, size=size)
    
    def __str__(self):
        return f"Current_time: {self.current_time} \nLOB: {self.printlob()}\nSpread: {self.spread}"
        
if __name__=="__main__":
    rtn=Exchange(None)
    print(f"Class {type(rtn).__name__} compiles")