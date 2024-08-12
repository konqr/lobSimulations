from RLenv.SimulationEntities.Entity import Entity
from RLenv.SimulationEntities.TradingAgent import TradingAgent
from typing import List, Tuple, Dict, Optional, Any
from abc import abstractmethod, ABC
import numpy as np

class GymTradingAgent(TradingAgent, ABC):
    """Abstract class to inherit from to create usable specific ABIDES Gym Experiemental Agents """

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
    def __init__(self, type: str = "TradingAgent", seed=1, log_events: bool = True, log_to_file: bool = False, strategy: str= "Random", Inventory: int=Dict[str, Any], cash: int=5000, action_freq: float =0.5, on_trade: bool=True, rewardpenalty: Optional[float]=None):
        """
        Rewardpenalty: absolute value of lambda ratio for the quadratic inventory penalty
        """
        super().__init__(type=type, seed=seed, log_events = log_events, log_to_file = log_to_file, strategy=strategy, Inventory=Inventory, cash=cash, action_freq=action_freq, on_trade=on_trade)
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
        return action, size
    
    
    def resetseed(self, seed):
        np.random.seed(seed)
    
    def calculaterewards(self) -> Any:
        penalty= self.rewardpenalty * self.countInventory()
        deltaPNL=self.statelog[0][1]
        #deltaPNL=self.profitlog[-1] - self.profitlog[-2]
        return deltaPNL - penalty