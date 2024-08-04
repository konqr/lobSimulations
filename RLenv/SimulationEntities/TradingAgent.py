import numpy as np
from abc import ABC, abstractmethod 
from typing import Any, List, Optional, Tuple, ClassVar, List, Dict
import pandas as pd
import logging
from RLenv.logging_config import *
from RLenv.Orders import *
from RLenv.Messages.Message import *
from RLenv.Messages.ExchangeMessages import *
from RLenv.SimulationEntities.Entity import Entity
from RLenv.Exceptions import *
logger = logging.getLogger(__name__)

class TradingAgent(Entity):
    """
    Base Agent class, inherited from entities

    Entity Attributes:
        id: Must be a unique number (usually autoincremented) belonging to the subclass.
        type: is it a trading agent or exchange?
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
        """
    def __init__(self, type: str = "TradingAgent", seed=1, log_events: bool = True, log_to_file: bool = False, strategy: str= "Random", Inventory: int=Dict[str, Any], action_freq: float =0.5) -> None:
        
        self.strategy=strategy #string that describes what strategy the agent has
        self.Inventory=Inventory #Dictionary of how many shares the agent is holding
        self.action_freq=action_freq
        self.positions: Dict[str, List[LimitOrder]]={} #A Dictionary that describes an agent's active orders in the market
        self.actions=["lo_deep_Ask", "co_deep_Ask", "lo_top_Ask","co_top_Ask", "mo_Ask", "lo_inspread_Ask" ,
            "lo_inspread_Bid" , "mo_Bid", "co_top_Bid", "lo_top_Bid", "co_deep_Bid","lo_deep_Bid" ]
        self.actionsToLevels = {
            "lo_deep_Ask" : "Ask_deep",
            "lo_top_Ask" : "Ask_touch",
            "lo_top_Bid" : "Bid_touch",
            "lo_deep_Bid" : "Bid_deep",
            "co_deep_Ask" : "Ask_deep",
            "co_top_Ask" : "Ask_touch",
            "co_top_Bid" : "Bid_touch",
            "co_deep_Bid" : "Bid_deep"
        }
        # Simulation attributes
        
        self.kernel=None
        self.exchange=None
        #What time does the agent think it is?
        self.current_time: int = 0 
        
        #Agents will maintain a log of their activities, events will likely be stored in the format of (time, event_type, eventname, size)
        self.log: List[Tuple[int, Order, Dict[str, int], int]] #Tuple(time, order, inventory, cash)
        super().__init__(type=type, seed=seed, log_events=log_events, log_to_file=log_to_file)
        
    def kernel_start(self, start_time: float) -> None:
        assert self.kernel is not None, f"Kernel not linked to TradingAgent: {self.id}"
        assert start_time is not None, f"Start time not provided to Trading Agent"
        logger.debug(
            "Trading Agent {} requesting kernel wakeup at time {}".format(self.id, start_time))
        wakeuptime=self.current_time + self.action_freq
        self.set_wakeup(requested_time= wakeuptime)
        
    
    def kernel_terminate(self) -> None:
        if self.log and self.log_to_file:
            df_log=pd.DataFrame(self.log, columns=("EventTime", "Order", "Inventory", "Cash"))
            self.write_log(df_log)
            
    
    def submitorder(self, order: Order):
        """
        Converts an action k into an order object and submits it
        """
        pass
        
    def action_to_order(self, action: Tuple[int, int], symbol: str=None, cancelID: Optional[int]=None) -> Order:
        """
        Converts an action to an order object
        Action is a (k, size) tuple where k refers to the event.
        Symbol is the stock symbol
        cancelID takes on non-none values when the event is a cancel order.
        """
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
                price=lob["Ask_touch"][0]+self.exchange.ticksize
            elif k==6: #inspread bid
                price=lob["Bid_touch"][0]+self.exchange.ticksize
            else:    
                level=self.actionsToLevels[self.actions[k]]
                if side=="Ask":
                    price=lob[level][0]
                else:
                    price=lob[level][0]
            order=LimitOrder(time_placed=self.current_time, event_type=k, side=side, size=size, symbol=self.exchange.symbol, agent_id=self.id, event_type=k, price=price, loblevel=level)
            
        elif self.actions[k][0:2]=="mo":
            #marketorder
            order=MarketOrder(time_placed=self.current_time, side=side,  event_type=k, size=size, symbol=symbol, agent_id=self.id, event_type=k, )
        else:
            #cancelorder
            assert cancelID is not None, f"CancelID not provided to agent {self.id}"
            level=self.actionsToLevels[self.actions[k]]
            if side=="Ask":
                price=lob[level][0]
            else:
                price=lob[level][0]
            order=CancelOrder(time_placed=self.current_time,  event_type=k, side=side, size=-1, symbol=self.exchange.symbol, agent_id=self.id,  event_type=k, cancelID=cancelID)
        return order
    
    
    def peakLOB(self) -> Dict[str, Tuple[float, int]]:
        assert self.exchange is not None, f"Trading Agent: {self.id} is not linked to an exchange"
        
        return self.exchange.lob0()
    
    
    def receivemessage(self, current_time, senderID: Any, message: Message):
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
                    self.inventory[order.symbol]-=order.size
                else:
                    self.cash-=order.fill_price
                    self.Inventory[order.symbol]+=order.size
                    pass
                pass
            elif isinstance(order, LimitOrder):
                if order.side=="Ask":
                    self.inventory[order.symbol]-=order.size
                    self.cash+=order.price*order.size    
                else:
                    self.inventory[order.symbol]+=order.size
                    self.cash-=order.price*order.size
                self.positions[order.symbol]=[j for j in self.positions[order.symbol] if j.id!=order.id]
            elif isinstance(order, CancelOrder):
                self.positions[order.symbol]=[j for j in self.positions[order.symbol] if j.id!=order.id]
            else:
                raise UnexpectedMessageType(f"Agent {self.id} received uenxpected message type from Exchange {self.id}")
                pass
        elif isinstance(message, WakeAgentMsg):
            action=self.get_action()
            order=self.action_to_order(action)
            self.submitorder(order)
            self.set_wakeup(requested_time=self.current_time+self.action_freq)
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
         
    
          
    @abstractmethod        
    def update_state(self, kernelmessage): #update internal state given a kernel message
        pass
    
    @abstractmethod
    def resetseed(self, seed):
        np.random.seed(1)
        return None
    
    @abstractmethod
    def get_action(self, data) -> Tuple[int, int]:
        pass
    
        
