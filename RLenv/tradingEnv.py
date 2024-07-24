import gymnasium as gym 
from gymnasium import spaces
import numpy as np
from RLenv.SimulationEntities.TradingAgent import TradingAgent
from RLenv.Stochastic_Processes.Arrival_Models import ArrivalModel, HawkesArrival
from RLenv.SimulationEntities.Exchange import Exchange

class tradingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", "text"], "render_fps": 4}
    def __init__(self, render_mode, Arrival_Model: ArrivalModel=None, params=None):
        """
        Initiates a trading environment. Arrivalmodel contains the simulation of the trading orders and params contains all other relevant information
        Relevant parameters:
        starttime
        endtime
        Rewardfunction
        seed
        Arrivalmodel
        """
        self.observation_space
        self.action_space
        assert render_mode is None or render_mode in self.metadata["render_modes"], "Render mode must be None, human or rgb_array"
        self.Orderbook= LimitOrderBook(Pi_Q0=None, priceMid0=260, spread0=4, ticksize=0.01, numOrdersPerLevel=10, Arrival_model=Arrival_Model)
        self.reset()
            
    def step(self, action):
        """
        Wrapper for state step -- a step ends when the next trade happens or when tau seconds have passed by, whichever occurs first for the agent
        Input: Action
        Output: Observations, Rewards, termination, truncation, Logging info+metrics 
        """
        return None
        #return observations, rewards, dones, infos    
        
        
    def reset(self, seed=None):
        """
        Reset env to default starting state + clear all simulations
        Params: Seed
        Output: Observations, info
        """
        
        #Reset Random seed
        #Sample LOB State
        return None
        #return Observations, info
    
    def render(self):
        """
        Render an environment
        """
        pass
    def updatestate(self, action):
        """
        Update agent and market states
        """
        return None
    
    def close(self):
        """
        close any active running windows/tasks
        """
    
    def _get_obs(self):
        """
        Private method to return observations on a state
        """
        return 

    def _get_info(self):
        """
        Returns auxiliary info: 
        """
        return
    
    
