import gymnasium as gym 
from gymnasium import spaces
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from RLenv.SimulationEntities.TradingAgent import TradingAgent
from RLenv.SimulationEntities.GymTradingAgent import GymTradingAgent, RandomGymTradingAgent
from RLenv.Stochastic_Processes.Arrival_Models import ArrivalModel, HawkesArrival
from RLenv.SimulationEntities.Exchange import Exchange
from RLenv.Kernel import Kernel
class tradingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", "text"], "render_fps": 4}
    
    def __init__(self, render_mode="text", stop_time: int=100, wall_time_limit: int=300, kernel_name: str="Alpha", seed=1, log_to_file=True,params=None):
        """
        Initiates a trading environment. Arrivalmodel contains the simulation of the trading orders and params contains all other relevant information
        Relevant arguments:
        render_mode
        stop_time: simulation stop time in simulationseconds
        wall_time_limit: simulation wall time limit
        kernel_name: for human readability 
        seed
        log_to_file
        params -- Simulation parameters specified:
            Dictionary{
                "Agent"
                "Exchange"              
            }

        """   
        self.observation_space=spaces.Dict({
            "Cash": spaces.Box(low=0),
            "Inventory": spaces.Box(low=0),
            "LOB_State": spaces.Dict({
                "Ask_L2": spaces.Tuple(spaces.Box(low=0), spaces.Sequence(spaces.Box(low=0))),
                "Ask_L1": spaces.Tuple(spaces.Box(low=0), spaces.Sequence(spaces.Box(low=0))),
                "Bid_L1": spaces.Tuple(spaces.Box(low=0), spaces.Sequence(spaces.Box(low=0))),
                "Bid_L2": spaces.Tuple(spaces.Box(low=0), spaces.Sequence(spaces.Box(low=0)))}),
            "Current_Positions": spaces.Sequence(spaces.Tuple(spaces.Text(), spaces.Box(low=0), spaces.Box(low=0)))
        })
        self.action_space=spaces.Tuple(spaces.Discrete(13), spaces.Box(0, 5000))
        assert render_mode is None or render_mode in self.metadata["render_modes"], "Render mode must be None, human or rgb_array or text"
        self.seed=seed
        self.kernel=Kernel(agents=, exchange=Exchange(symbol="XYZ", ticksize=0.01, numOrdersPerLevel=10, seed=self.seed), seed=seed, kernel_name=kernel_name, stop_time=stop_time, wall_time_limit=wall_time_limit, log_to_file=log_to_file, parameters=params)
        self.kernel.initialize_kernel()
            
    def step(self, action: Optional[Any]):
        """
        Wrapper for state step -- a step ends when the next trade happens or when tau seconds have passed by, whichever occurs first for the agent
        Input: Action
        Output: Observations, Rewards, termination, truncation, Logging info+metrics 
        """
        if self.kernel.isrunning==False:
            self.kernel.begin()
        #Observations=cash, inventory, LOB state, current positions
        simstate=self.kernel.run(action)
        Observations=self.getobservations()
        rewards=self.calculaterewards()
        termination=self.isterminated()
        truncation=self.istruncated()
        info=self.getinfo()
        return Observations, rewards, termination, truncation
        #return observations, rewards, dones, infos    
        
        
    def reset(self, seed=None):
        """
        Reset env to default starting state + clear all simulations
        Params: Seed
        Output: Observations, info
        """
        self.kernel.reset(seed=seed)
        observations=self.kernel.getobservations()
        info=self.kernel.getinfos()
        return observations
    
    def render(self):
        """
        Render an environment
        """
        pass
    
    def close(self):
        """
        close any active running windows/tasks
        """
        self.kernel.terminate()
    
    #Wrappers
    def getobservation(self):
        
        return self.kernel.getobservations()
    def calculaterewards(self):
        rewards={}
        for gymagent in self.kernel.gymagents:
            rewards[gymagent.id]=gymagent.calculaterewards()
        return rewards        
    def isterminated(self):
        return self.kernel.isterminated()
    def istruncated(self):
        return self.kernel.istruncated()  
    def getinfo(self):
        """
        Returns auxiliary info: 
        """
        return None
