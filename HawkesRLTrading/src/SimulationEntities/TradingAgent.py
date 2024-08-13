import numpy as np
from abc import ABC, abstractmethod 
from typing import Any, List, Optional, Tuple, ClassVar, List, Dict
import pandas as pd
import logging
import copy
from HawkesRLTrading.src.Utils.logging_config import *
from HawkesRLTrading.src.Orders import *
from HawkesRLTrading.src.Messages.Message import *
from HawkesRLTrading.src.Messages.ExchangeMessages import *
from HawkesRLTrading.src.Messages.AgentMessages import *
from HawkesRLTrading.src.SimulationEntities.Entity import Entity
from HawkesRLTrading.src.Utils.Exceptions import *
logger = logging.getLogger(__name__)

class TradingAgent(Entity):
    """
    Base Agent class, inherited from entities

    Entity Attributes:
        id: Must be a unique number (usually autoincremented) belonging to the subclass.
        seed: Every entity is given a random seed for stochastic purposes.
        log_events: flag to log or not the events during the simulation 
            Logging format:
                time: time of event
                event_type: label of the event (e.g., Order submitted, order accepted, last trade etc....)
                event: actual event to be logged (co_deep_Ask, lo_deep_Ask...)
                size: size of the order (if applicable)
        log_to_file: flag to write on disk or not the logged events
    Agent Child Class attributes
        strategy: Is it a random agent? a RL agent?
        cash: cash of an agent
        Inventory: Initial inventory of an agent
        action_freq: what is the time period betweeen an agent's actions in seconds
        on_trade: decides whether agent gets suscribed to new information every time a trade happens
        """
    def __init__(self, seed=1, log_events: bool = True, log_to_file: bool = False, strategy: str= "Random", Inventory: Optional[Dict[str, Any]]=None, cash: int=5000, action_freq: float =0.5, on_trade: bool=False) -> None:
        super().__init__(type="TradingAgent", seed=seed, log_events=log_events, log_to_file=log_to_file)
        self.strategy=strategy #string that describes what strategy the agent has
        assert Inventory is not None, f"Agent needs inventory for initialisation"
        self.Inventory=Inventory #Dictionary of how many shares the agent is holding
        self.action_freq=action_freq
        self.on_trade: bool=on_trade #Does this agent get notified to make a trade whenever a trade happens
        self.positions: Dict[str: List[Order]]={key: [] for key in self.Inventory.keys()} #A Dictionary that describes an agent's active orders in the market
        self.cash=cash
        self.profit=0
        self.statelog=[(0, self.cash, self.profit, Inventory.copy(), self.positions.copy())] #List of [timecode, cash, #realized profit, inventory, positions]
        self.actions=["lo_deep_Ask", "co_deep_Ask", "lo_top_Ask","co_top_Ask", "mo_Ask", "lo_inspread_Ask" ,
            "lo_inspread_Bid" , "mo_Bid", "co_top_Bid", "lo_top_Bid", "co_deep_Bid","lo_deep_Bid", None]
        self.actionsToLevels = {
            "lo_deep_Ask" : "Ask_L2",
            "lo_top_Ask" : "Ask_L1",
            "lo_top_Bid" : "Bid_L1",
            "lo_deep_Bid" : "Bid_L2",
            "co_deep_Ask" : "Ask_L2",
            "co_top_Ask" : "Ask_L1",
            "co_top_Bid" : "Bid_L1",
            "co_deep_Bid" : "Bid_L2",
            "mo_Ask": "Ask_MO",
            "mo_Bid": "Bid_MO",
            "lo_inspread_Ask": "Ask_inspread",
            "lo_inspread_Bid": "Bid_inspread"
        }
        # Simulation attributes
        
        self.exchange=None
        self.isterminated=False
        #What time does the agent think it is?
        self.current_time: int = 0 
        
        #Agents will maintain a log of their activities, events will likely be stored in the format of (time, event_type, eventname, size)
        self.log: List[Tuple[int, Order, Dict[str, int], int]]=[] #Tuple(time, order, inventory, cash)
        
    def kernel_start(self, current_time: int=0) -> None:
        assert self.kernel is not None, f"Kernel not linked to TradingAgent: {self.id}"
        wakeuptime=self.current_time + self.action_freq
        logger.debug(f"{type(self).__name__} {self.id} requesting kernel wakeup at time {wakeuptime}")
        self.set_wakeup(requested_time= wakeuptime)
        
    
    def kernel_terminate(self) -> None:
        if self.log and self.log_to_file:
            df_log=pd.DataFrame(self.log, columns=("EventTime", "Order", "Inventory", "Cash"))
            self.write_log(df_log)
            
    
    def submitorder(self, order: Optional[Order]) -> int:
        """
        Converts an action k into an order object and submits it
        """
        assert self.exchange is not None, f"Agent {self.id} is not linked to an Exchange"
        if order is None:
            self.sendmessage(recipientID=self.exchange.id, message=DoNothing(time_placed=self.current_time))
            return -1
        else:
            if isinstance(order, LimitOrder):
                message=LimitOrderMsg(order=order)
                self.sendmessage(recipientID=self.exchange.id, message=message)
            elif isinstance(order, MarketOrder):
                message=MarketOrderMsg(order=order)
                self.sendmessage(recipientID=self.exchange.id, message=message)
            else:
                message=CancelOrderMsg(order=order)
                self.sendmessage(recipientID=self.exchange.id, message=message)
            return 0
        
    def action_to_order(self, action: Optional[Tuple[int, int]]=None, symbol: str=None) -> Optional[Order]:
        """
        Converts an action to an order object
        Action is a (k, size) tuple where k refers to the event, or None.
        Symbol is the stock symbol
        cancelID takes on non-none values when the event is a cancel order.
        """
        if action==None:
            raise ValueError("Action should not be none")
        if action[0]==12:
            return None
        k=action[0]
        size=action[1]
        if k<0 or k>11:
            raise IndexError(f"Event index is out of bounds. Expected 0<=k<=11, but received k={k}")
        order=None
        lob=self.peakLOB()
        if "Ask" in self.actions[k]:
            side="Ask"
        else:
            side="Bid"
        if self.actions[k][0:2]=="lo":
            if k==5: #inspread ask
                price=lob["Ask_L1"][0]+self.exchange.ticksize
            elif k==6: #inspread bid
                price=lob["Bid_L1"][0]+self.exchange.ticksize
            else:    
                level=self.actionsToLevels[self.actions[k]]
                if side=="Ask":
                    price=lob[level][0]
                else:
                    price=lob[level][0]
            order=LimitOrder(time_placed=self.current_time, side=side, size=size, symbol=self.exchange.symbol, agent_id=self.id, price=price, _level=level)
            
        elif self.actions[k][0:2]=="mo":
            #marketorder
            level=self.actionsToLevels[self.actions[k]]
            order=MarketOrder(time_placed=self.current_time, side=side, size=size, symbol=self.exchange.symbol, agent_id=self.id, _level=level)
        else:
            
            #cancelorder
            level=self.actionsToLevels[self.actions[k]]
            if side=="Ask":
                price=lob[level][0]
            else:
                price=lob[level][0]
            positions=[j for j in self.positions[self.exchange.symbol] if j.side==side and j.price==price]
            if len(positions)==0:
                raise InvalidActionError(f"Agent {self.id} cannot cancel orders without placing any orders")
            else:
                tocancel: LimitOrder=np.random.choice(positions)
            order=CancelOrder(time_placed=self.current_time, side=side, size=-1, symbol=self.exchange.symbol, agent_id=self.id,  event_type=k, cancelID=tocancel.order_id, _level=level)
        return order
    
    
    def receivemessage(self, current_time: float, senderID: int, message: Message):
        """
        Called each time a message destined for this agent reaches the front of the
        kernel's priority queue.
        Arguments:
            current_time: The simulation time at which the kernel is delivering this message -- the agent should treat this as "now".
            sender: The object that send this message(can be the exchange, another agent or the kernel).
            message: An object guaranteed to inherit from the Message.Message class.
        """
        assert self.kernel is not None, f"Kernel not linked to TradingAgent: {self.id}"
        assert message is not None, f"Trading Agent {self.id} received a blank message with ID{message.message_id}"
        self.current_time=current_time
        #Check identity of the sender
        if isinstance(message, OrderExecutedMsg):
            #update agent state
            order=message.order
            if isinstance(order, MarketOrder):
                if order.side=="Ask":
                    self.cash+=order.fill_price
                    self.Inventory[order.symbol]-=order.size
                else:
                    self.cash-=order.fill_price
                    self.Inventory[order.symbol]+=order.size
                    pass
                
            elif isinstance(order, LimitOrder):
                if order.side=="Ask":
                    self.Inventory[order.symbol]-=order.size
                    self.cash+=order.price*order.size    
                else:
                    self.Inventory[order.symbol]+=order.size
                    self.cash-=order.price*order.size
                self.positions[order.symbol].append(order)
            elif isinstance(order, CancelOrder):
                self.positions[order.symbol]=[j for j in self.positions[order.symbol] if j.id!=order.cancelID]
            else:
                raise UnexpectedMessageType(f"Agent {self.id} received uenxpected message type from Exchange {self.id}")
                pass
            self.profit=self.cash-self.statelog[0][1]
            self.updatestatelog()
        elif isinstance(message, WakeAgentMsg):
            
            action=self.get_action()
            order=self.action_to_order(action)
            self.submitorder(order)
            self.set_wakeup(requested_time=self.current_time+self.action_freq)
        elif isinstance(message, PartialOrderFill):
            newsize=message.newsize
            order: LimitOrder=message.order
            price=order.price
            if order.side=="Ask":
                self.cash+=order.price*(order.size-newsize)
                self.Inventory[order.symbol]-=(order.size-newsize)
            else:
                self.cash-=order.price*(order.size-newsize)
                self.Inventory[order.symbol]+=(order.size-newsize)
            order.size=newsize
            self.profit=self.cash-self.statelog[0][1]
            self.updatestatelog()
        elif isinstance(message, OrderAutoCancelledMsg):
            self.positions[order.symbol]=[j for j in self.positions[order.symbol] if j.id!=order.cancelID]
        else:
            raise TypeError(f"Unexpected message type: {type(message).__name__}")
        super().receivemessage(current_time, senderID, message)
        
    def wakeup(self, current_time: int) -> None:
        """
        Agents can request a wakeup call at a future simulation time using
        the Agent.set_wakeup(time) method

        This is the method called when the wakeup time arrives.

        Arguments:
            current_time: The simulation time at which the kernel is delivering this
                message -- the agent should treat this as "now".
        """

        assert self.kernel is not None

        self.current_time = current_time

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "At {}, agent {} received wakeup.".format(
                    current_time, self.id
                )
            )
    def peakLOB(self) -> Dict[str, Tuple[float, int]]:
        assert self.exchange is not None, f"Trading Agent: {self.id} is not linked to an exchange"
        
        return self.exchange.lob0
    
    def countInventory(self):
        return sum([self.Inventory[j] for j in self.Inventory.keys()])     
    
    def reset(self) -> None:
        pass

    def updatestatelog(self):
        if self.statelog[-1][0]==self.current_time:
            logger.info(f"Last agent LOG is {self.statelog[-1]} and new log to be appended is {(self.current_time, self.cash, self.profit, self.Inventory.copy(), self.positions.copy())}")
            return False
        self.statelog.append((self.current_time, self.cash, self.profit, self.Inventory.copy(), self.positions.copy()))
    
    def getobservations(self):
        rtn={"Cash":self.cash,
             "Inventory": self.Inventory[self.exchange.symbol],
             "Positions": self.positions[self.exchange.symbol]}
        return rtn