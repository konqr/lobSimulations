import gymnasium as gym 
from gymnasium import spaces
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from src.SimulationEntities.TradingAgent import TradingAgent
from src.SimulationEntities.GymTradingAgent import GymTradingAgent, RandomGymTradingAgent
from src.Stochastic_Processes.Arrival_Models import ArrivalModel, HawkesArrival
from src.SimulationEntities.Exchange import Exchange
from src.Kernel import Kernel
class tradingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", "text"], "render_fps": 4}
    
    def __init__(self, render_mode="text", stop_time: int=100, wall_time_limit: int=300, kernel_name: str="Alpha", seed=1, log_to_file=True,**kwargs):
        """
        Initiates a trading environment. Arrivalmodel contains the simulation of the trading orders and params contains all other relevant information
        Relevant arguments:
        render_mode
        stop_time: simulation stop time in simulationseconds
        wall_time_limit: simulation wall time limit
        kernel_name: for human readability 
        seed
        log_to_file
        kwargs -- Simulation default parameters specified as following:
            Dictionary{
                "TradingAgent": None,
                "GymTradingAgent": [{"cash": 5000, 
                                    "strategy": "Random",
                                    "action_freq": 0.5
                                    "rewardpenalty": 0.4
                                    "Inventory": {"XYZ": 1000}
                                    "log_to_file": True}],
                "Exchange": {"symbol": "XYZ",
                "ticksize":0.01,
                "LOBlevels": 2,
                "numOrdersPerLevel": 10,
                "PriceMid0": 45
                "spread0": 5},
                "Arrival_model": {"name": "Hawkes",
                                  "parameters"={"kernelparams": None, 
                                            "tod": None, 
                                            "Pis": None, 
                                            "beta": None, 
                                            "avgSpread": None, 
                                            "Pi_Q0": None}}

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
        
        #Construct agents: Right now only compatible with 1 trading agent
        assert len(kwargs["GymTradingAgent"])==1 and len(kwargs["TradingAgent"]==0), "Kernel simulation can only take a total of 1 agent currently, and it should be a GYM agent"
        agents=[]
        if len(kwargs["TradingAgent"])>0:
            pass
        if len(kwargs["GymTradingAgent"])>0:
            for j in kwargs["GymTradingAgent"]:
                new_agent=None
                if j["strategy"]=="Random":
                    new_agent=RandomGymTradingAgent(seed=self.seed, log_events=True, log_to_file=log_to_file, strategy=j["strategy"], Inventory=j["Inventory"], cash=j["cash"], action_freq=j["action_freq"] , rewardpenalty=j["rewardpenalty"], on_trade=True)
                else:
                    raise Exception("Program only supports RandomGymTrading Agents for now")      
                agents.append(new_agent)
        
        #Construct Exchange:
        ExchangeKwargs={}
        for j in [ "symbol", "ticksize", "LOBlevels", "numOrdersPerLevel", "PriceMid0", "spread0"]:
            rtn=kwargs.get(j)
            ExchangeKwargs[j]=rtn
        exchange=Exchange(**ExchangeKwargs)
        #Arrival Model setup
        Arrival_model=None
        if kwargs.get("Arrival_model"):
            if kwargs["Arrival_model"].get("name") == "Hawkes":
                params = kwargs["Arrival_model"]["parameters"]
                Arrival_model: HawkesArrival=HawkesArrival(spread0=exchange.spread, seed=self.seed, **params)
            else:
                raise Exception("Program currently only supports Hawkes Arrival model")
        else:
            raise ValueError("Please specify Arrival model")
            
               
        
        self.kernel=Kernel(agents=agents, exchange=exchange, seed=seed, kernel_name=kernel_name, stop_time=stop_time, wall_time_limit=wall_time_limit, log_to_file=log_to_file, Arrival_model=Arrival_model)
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

if __name__=="__main__":
    print("compiles")
    env=tradingEnv()