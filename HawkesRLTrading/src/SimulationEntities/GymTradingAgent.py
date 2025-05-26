from HawkesRLTrading.src.SimulationEntities.Entity import Entity
from HawkesRLTrading.src.SimulationEntities.TradingAgent import TradingAgent
from typing import List, Tuple, Dict, Optional, Any
from abc import abstractmethod, ABC
import numpy as np

class GymTradingAgent(TradingAgent):
    """Abstract class to inherit from to create usable specific Gym Experimental Agents """
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
    def __init__(self, seed=1, log_events: bool = True, log_to_file: bool = False, strategy: str= "Random", Inventory: Optional[Dict[str, Any]]=None, cash: int=5000, action_freq: float =0.5, wake_on_MO: bool=True, wake_on_Spread: bool=True , rewardpenalty: float=0.4, cashlimit=1000000, inventorylimit=10000):
        """
        Rewardpenalty: absolute value of lambda ratio for the quadratic inventory penalty
        """
        super().__init__(seed=seed, log_events = log_events, log_to_file = log_to_file, strategy=strategy, Inventory=Inventory, cash=cash, action_freq=action_freq, wake_on_MO=wake_on_MO, wake_on_Spread=wake_on_Spread, cashlimit=cashlimit, inventorylimit=inventorylimit)
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
            if len(self.positions[self.exchange.symbol][lvl]) == 0: #no position to cancel
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

