from HawkesRLTrading.src.SimulationEntities.Entity import Entity
from HawkesRLTrading.src.SimulationEntities.TradingAgent import TradingAgent
from typing import List, Tuple, Dict, Optional, Any
from abc import abstractmethod, ABC
import numpy as np

class GymTradingAgent(TradingAgent):
    """Abstract class to inherit from to create usable specific Gym Experiemental Agents """
    @abstractmethod
    def get_action(self, data) -> Optional[Tuple[int, int]]:
        pass
    
    @abstractmethod
    def calculaterewards(self) -> Any:
        pass
    
    @abstractmethod
    def resetseed(self, seed):
        pass


class RandomGymTradingAgent(GymTradingAgent):
    def __init__(self, seed=1, log_events: bool = True, log_to_file: bool = False, strategy: str= "Random", Inventory: Optional[Dict[str, Any]]=None, cash: int=5000, action_freq: float =0.5, on_trade: bool=True, rewardpenalty: float=0.4, cashlimit=1000000):
        """
        Rewardpenalty: absolute value of lambda ratio for the quadratic inventory penalty
        """
        super().__init__(seed=seed, log_events = log_events, log_to_file = log_to_file, strategy=strategy, Inventory=Inventory, cash=cash, action_freq=action_freq, on_trade=on_trade, cashlimit=cashlimit)
        if rewardpenalty is None:
            raise ValueError(f"Rewardpenalty value for agent {self.id} not specified")
        self.rewardpenalty=abs(rewardpenalty)
        self.resetseed(seed=seed)

    def get_action(self, data=None) -> Optional[Tuple[int, int]]:
        """
        Action is a (k, size) tuple where 0<=k<=12 refers to the event. If k=12, then it refers to the NoneAction
        """
        
        action=np.random.choice(range(13))
        size=np.random.choice([50,75,100,125])
        if action in [1,3,8,10]: # cancels
            a = self.actions[action]
            lvl = self.actionsToLevels[a]
            if len(data['Positions'][lvl]) == 0: #no position to cancel
                return self.get_action(data) # retry
        if (action in [5,6]) and np.isclose( self.exchange.spread, 0.01): #inspread
            return self.get_action(data) # retry
        return action, size
    
    
    def resetseed(self, seed):
        np.random.seed(seed)
    
    def calculaterewards(self) -> Any:
        penalty= self.rewardpenalty * self.countInventory()
        self.profit=self.cash - self.statelog[0][1]
        self.updatestatelog()
        deltaPNL=self.statelog[-1][2] - self.statelog[-2][2]
        return deltaPNL - penalty

class TWAPGymTradingAgent(GymTradingAgent):
    def __init__(self, seed, log_events:bool, log_to_file:bool, strategy:str, Inventory:Optional[Dict[str, Any]], cash:int, action_freq:float, total_order_size:int, total_time:int, window_size:int, side:str, order_target:str, on_trade:bool = True):
        if side=="sell": assert Inventory[order_target] >= total_order_size, "Not enough volume in inventory to execute sell order"
        assert total_order_size%window_size == 0, f"Order size {total_order_size} cannot be executed with window size {window_size}"
        assert total_order_size % (total_time//action_freq) == 0, f"Order size {total_order_size} cannot be executed evenly with time {total_time} and action frequency {action_freq} "

        super().__init__(seed=seed, log_events = log_events, log_to_file = log_to_file, strategy=strategy, Inventory=Inventory, cash=cash, action_freq=action_freq, on_trade=on_trade)
        self.total_time = total_time
        self.total_order_size = total_order_size
        self.side = side
        self.order_target = order_target

        self.urgent = False
        self._set_slices()
        self._set_windows()



    def _set_slices(self):
        self.num_time_slices:int = self.total_time // self.action_freq #the number of individual time slices, of equal size i.e. the number of orders
        self.individual_order_size:int = self.total_order_size // self.num_time_slices #the size of the individual orders
    
    def _set_windows(self):
        self.num_windows:int = self.total_time // 900 #number of 15 minute windows 
        self.orders_per_window:int = self.num_time_slices // self.num_windows #the amount of trades to execute per window
        self.windows_left:int = self.num_windows #to keep track of how many windows we have gone through
        self.order_per_window_count:int = 0 #for the purposes of checking if we have executed all the trades in the window
        self.agent_volume:int = 0
        self.total_volume_window:int = self.total_order_size // self.num_windows #the amount of volume to trade in each window
        self.market_orders_this_window:int = 0
        self.market_order_size:int = 0
        self.volume_traded_in_window:int = 0
        self.window_time_elapsed:float = 0.0

        
        
        

    def get_action(self, data) -> Optional[Tuple[int, int]]:
        #calculate the time elapsed in this window so far
        self.window_time_elapsed += (self.current_time/1000000 - self.window_time_elapsed - self.window_size*(self.num_windows-self.windows_left))
        time_ratio = self.window_time_elapsed/self.window_size #this gives a ratio for how much time we have spent on this window

        #when the agent reaches the end of a trade window, i.e. we have no more time left, then 
        #decrement the number of windows left, and reset all window-dependent variables
        if time_ratio >= 1:
            self.window_time_elapsed = 0
            self.market_orders_this_window = 0
            self.urgent = False
            self.order_per_window_count = 0
            self.volume_traded_in_window = 0
            self.windows_left -= 1

        if self.volume_traded_in_window == self.total_volume_window or self.windows_left < 0:
            #if we have already traded all of our volume for this window (i.e. all of our orders have "gone through") 
            return (12, 0)
        
        #at every step, calculate the volume that has been executed so far in this window
        old_volume = self.agent_volume
        self.agent_volume = self.Inventory.get(self.order_target) 
        if self.side == "buy":
            self.volume_traded_in_window += self.agent_volume - old_volume
        elif self.side == "sell":
            self.volume_traded_in_window += old_volume - self.agent_volume

        #if we have gone through 75% of our window time and have less than a 90% execution rate
        if time_ratio > 0.75 and self.volume_traded_in_window/(self.total_volume_window*time_ratio) < 0.9:
            self.urgent = True
        
        if not self.urgent: #place limit orders like usual 
            self.order_per_window_count += 1 #this variable will likely be obsolete at the end
            return (9, self.individual_order_size) if self.side == "buy" else (2, self.individual_order_size)
        
        #otherwise, place market orders
        else:
            #cancel existing limit orders
            lvl = self.actionsToLevels[self.actions[9]] if self.side=="buy" else self.actionsToLevels[self.actions[2]]
            if len(data["Positions"][lvl]) > 0:
                return(8, self.individual_order_size) if self.side == "buy" else (3, self.individual_order_size)
            #on the first market order calculate the size of the market orders we will have to place, based on how much time we have left
            if self.market_orders_this_window == 0:
                self.market_order_size:int =  (self.total_volume_window - self.volume_traded_in_window) // (self.window_size - self.window_time_elapsed)
            return(7, self.market_order_size) if self.side == "buy" else (4, self.market_order_size)

    

    def calculaterewards(self) -> Any:
        pass
    
    def resetseed(self, seed):
        pass