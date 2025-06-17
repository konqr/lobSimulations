from HawkesRLTrading.src.SimulationEntities.GymTradingAgent import GymTradingAgent
from typing import List, Tuple, Dict, Optional, Any
from abc import abstractmethod, ABC
import numpy as np
from decimal import Decimal

class TWAPGymTradingAgent(GymTradingAgent):
    def __init__(self, seed, log_events:bool, log_to_file:bool, strategy:str, Inventory:Optional[Dict[str, Any]], cash:int, cashlimit:int, action_freq:float, total_order_size:int, total_time:int, window_size:int, side:str, order_target:str,  start_trading_lag:int=0, wake_on_MO:bool=False, wake_on_Spread:bool=False):
        if side=="sell": assert Inventory[order_target] >= total_order_size, "Not enough volume in inventory to execute sell order"
        assert total_order_size%window_size == 0, f"Order size {total_order_size} cannot be executed with window size {window_size}"
        assert total_order_size % (total_time/action_freq) == 0, f"Order size {total_order_size} cannot be executed evenly with time {total_time} and action frequency {action_freq} "

        super().__init__(seed=seed, log_events = log_events, log_to_file = log_to_file, strategy=strategy, Inventory=Inventory, cash=cash, action_freq=action_freq, wake_on_MO=wake_on_MO, wake_on_Spread=wake_on_Spread, cashlimit=cashlimit,start_trading_lag=start_trading_lag)
        self.total_time = total_time
        self.actions_per_second:int = 1/self.action_freq
        self.total_order_size:int = total_order_size
        self.side:str = side
        self.order_target:str = order_target
        self.window_size:int = window_size
        self.traded_so_far:int = 0
        self.agent_volume:int = self.Inventory[self.order_target]
        self.old_volume:int = self.agent_volume
        self.starting_volume:int = self.old_volume
        self.urgent = False

        self._set_slices()
        self._set_windows()



    def _set_slices(self):
        self.num_time_slices:int = round(self.total_time / self.action_freq) #the number of individual time slices, of equal size i.e. the number of orders
        # self.individual_order_size:int = self.total_order_size // self.num_time_slices #the size of the individual orders
    
    def _set_windows(self):
        self.num_windows:int = self.total_time // self.window_size 
        
        self.total_volume_window:int = self.total_order_size // self.num_windows #the amount of volume to trade in each window
        self.market_order_size:int = 0
        self.volume_traded_in_window:int = 0
        self.window_time_elapsed:float = 0.0
        

        
        
        

    def get_action(self, data) -> Optional[Tuple[int, int]]:
        self.agent_volume = data["Inventory"]
        self.traded_so_far = self.agent_volume - self.starting_volume if self.side == "buy" else self.starting_volume - self.agent_volume
        individual_order_size = round((self.total_order_size-self.traded_so_far) / ((self.total_time - self.current_time) / self.action_freq))
        
        #update the time elapsed in this window so far
        self.window_time_elapsed += self.action_freq
        
        time_ratio = self.window_time_elapsed/self.window_size  # this gives a ratio for how much time we have spent on this window

        #when the agent reaches the end of a trade window, i.e. we have no more time left, then 
        #reset all window-dependent variables
        if time_ratio>=1:
            self.window_time_elapsed = 0
            self.urgent = False
            self.volume_traded_in_window = 0
            time_ratio = 0

        #at every step, calculate the volume that has been executed so far in this window
        
        if self.side == "buy":
            self.volume_traded_in_window += (self.agent_volume - self.old_volume)
        elif self.side == "sell":
            self.volume_traded_in_window += self.old_volume - self.agent_volume
        self.old_volume = self.agent_volume

        #if we have traded more than or equal to how much we should have traded so far, cancel some of the most viable limit orders
        if (self.traded_so_far >= self.total_order_size*(self.current_time/self.total_time) or self.traded_so_far >= self.total_order_size or self.volume_traded_in_window>=self.total_volume_window):
            lvl = self.actionsToLevels[self.actions[9]] if self.side=="buy" else self.actionsToLevels[self.actions[2]]
            if len(data["Positions"][lvl]) > 0:
                 return(8, 0) if self.side == "buy" else (3, 0)
            #if we dont have any positions left, skip
            return (12, 0)
        

        
        # #if we have gone through 75% of our window time and have less than a 90% execution rate
        if  time_ratio > 0.75 and self.volume_traded_in_window/(self.total_volume_window*time_ratio) < 0.9 and not self.urgent:
            self.urgent = True
        
        if not self.urgent: #place limit orders like usual 
            return (9, individual_order_size) if self.side == "buy" else (2, individual_order_size)
        #otherwise, place market orders
        else:
            lvl = self.actionsToLevels[self.actions[9]] if self.side=="buy" else self.actionsToLevels[self.actions[2]]
            time_left:int = self.window_size - self.window_time_elapsed
            num_actions_left_this_window:int = round(time_left * self.actions_per_second)
            if num_actions_left_this_window <= 0 : return(12, 0)
            self.market_order_size = round((self.total_volume_window - self.volume_traded_in_window)/num_actions_left_this_window)
            return(7, self.market_order_size) if self.side == "buy" else (4, self.market_order_size)


class POVGymTradingAgent(GymTradingAgent):
    def __init__(self, seed, log_events:bool, log_to_file:bool, strategy:str, Inventory:Optional[Dict[str, Any]], cash:int, cashlimit:int, action_freq:float, total_order_size:int, side:str, order_target:str, participation_rate:int, window_size:int, start_trading_lag:int=100, wake_on_MO:bool=False, wake_on_Spread:bool=False, total_time:int = 23400):
        super().__init__(seed=seed, log_events = log_events, log_to_file = log_to_file, strategy=strategy, Inventory=Inventory, cash=cash, action_freq=action_freq, wake_on_MO=wake_on_MO, wake_on_Spread=wake_on_Spread, cashlimit=cashlimit)
        self.side:str = side
        if side=="sell": assert Inventory[order_target] >= total_order_size, "Not enough volume in inventory to execute sell order"
        self.starting_volume = self.Inventory[order_target]
        self.agent_volume = self.starting_volume
        self.partipation_rate:float = participation_rate
        self.total_order_size:int = total_order_size
        self.window_size:int = window_size
        self.traded_so_far:int = 0
        self.old_volume:int = self.agent_volume
        self.total_time:int = total_time
        self.urgent = False
        self._set_windows()

    def _set_windows(self):
        self.num_windows:int = self.total_time // self.window_size 
        
        self.window_order_volume:int = 0 #depends on the market volume at the start of each window
        self.market_order_size:int = 0
        self.volume_traded_in_window:int = 0
        self.window_time_elapsed:float = 0.0

        self.calc_new_window_volume:bool = True


    def get_action(self, data):
        self.agent_volume:int = data["Inventory"]
        self.traded_so_far:int = abs(self.agent_volume - self.starting_volume)
        if(self.traded_so_far>=self.total_order_size):
            return(12, 0)
        
        marketvolume:int = data["market_volume"]
        #update the time elapsed in this window so far
        self.window_time_elapsed += self.action_freq

        time_ratio = self.window_time_elapsed/self.window_size

        #recaulcate the new volume of the window when needed
        if self.calc_new_window_volume:
            self.window_order_volume = self.partipation_rate*marketvolume*(self.window_size//self.action_freq)
            self.calc_new_window_volume = self.window_order_volume == 0

        #when the agent reaches the end of a trade window, i.e. we have no more time left, then 
        #reset all window-dependent variables
        if time_ratio>=1:
            self.window_time_elapsed = 0
            self.urgent = False
            self.volume_traded_in_window = 0

        individual_order_size = (self.window_order_volume-self.volume_traded_in_window)//(self.window_size//self.action_freq)

        #if we have nothing left to trade, then skip
        if individual_order_size <= 0:
            return (12, 0)
        #at every step, calculate the volume that has been executed so far in this window
        if self.side == "buy":
            self.volume_traded_in_window += (self.agent_volume - self.old_volume)
        elif self.side == "sell":
            self.volume_traded_in_window += self.old_volume - self.agent_volume
        self.old_volume = self.agent_volume

        #if we have traded more than or equal to how much we should have traded so far, cancel some of the most viable limit orders
        if (self.traded_so_far >= self.total_order_size*(self.current_time/self.total_time) or self.traded_so_far >= self.total_order_size or self.volume_traded_in_window>=self.window_order_volume):
            lvl = self.actionsToLevels[self.actions[9]] if self.side=="buy" else self.actionsToLevels[self.actions[2]]
            if len(data["Positions"][lvl]) > 0:
                 return(8, 0) if self.side == "buy" else (3, 0)
            #if we dont have any positions left, skip
            return (12, 0)
        
        if  time_ratio > 0.75 and self.volume_traded_in_window/(self.window_order_volume*time_ratio) < 0.9 and not self.urgent:
            self.urgent = True
        
        if not self.urgent:
            return (9, individual_order_size) if self.side == "buy" else (2, individual_order_size)
        #otherwise, place market orders
        else:
            lvl = self.actionsToLevels[self.actions[9]] if self.side=="buy" else self.actionsToLevels[self.actions[2]]
            time_left:int = self.window_size - self.window_time_elapsed
            num_actions_left_this_window:int = round(time_left * (1/self.action_freq))
            if num_actions_left_this_window <= 0 : return(12, 0)
            self.market_order_size = round((self.window_order_volume - self.volume_traded_in_window)/num_actions_left_this_window)
            return(7, self.market_order_size) if self.side == "buy" else (4, self.market_order_size)
        

