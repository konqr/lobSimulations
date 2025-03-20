import sys
import os
sys.path.append(os.path.abspath('/Users/alirazajafree/Documents/GitHub/lobSimulations/'))

import gymnasium as gym
import numpy as np
from typing import Any, Optional
import logging
from HawkesRLTrading.src.SimulationEntities.GymTradingAgent import GymTradingAgent, RandomGymTradingAgent, TWAPGymTradingAgent
from HawkesRLTrading.src.Stochastic_Processes.Arrival_Models import ArrivalModel, HawkesArrival
from HawkesRLTrading.src.SimulationEntities.Exchange import Exchange
from HawkesRLTrading.src.Kernel import Kernel
logger=logging.getLogger(__name__)
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
                "TradingAgent": [],
                "GymTradingAgent": [{"cash": 5000, 
                                    "strategy": "Random",
                                    "action_freq": 2,
                                    "rewardpenalty": 0.4,
                                    "Inventory": {"XYZ": 1000},
                                    "log_to_file": True}],
                "Exchange": {"symbol": "XYZ",
                "ticksize":0.01,
                "LOBlevels": 2,
                "numOrdersPerLevel": 10,
                "PriceMid0": 45,
                "spread0": 0.05},
                "Arrival_model": {"name": "Hawkes",
                                  "parameters": {"kernelparams": None, 
                                            "tod": None, 
                                            "Pis": None, 
                                            "beta": None, 
                                            "avgSpread": None, 
                                            "Pi_Q0": None}}

            }

        """               
        # self.observation_space=Dict({
        #     "Cash": Box(low=0, high=100000),
        #     "Inventory": Box(low=0, high=10000),
        #     "LOB_State": Dict({
        #         "Ask_L2": Tuple(Box(low=0, high=2000), Sequence(Box(low=0, high=10000))),
        #         "Ask_L1": Tuple(Box(low=0, high=2000), Sequence(Box(low=0, high=10000))),
        #         "Bid_L1": Tuple(Box(low=0, high=2000), Sequence(Box(low=0, high=10000))),
        #         "Bid_L2": Tuple(Box(low=0, high=2000), Sequence(Box(low=0, high=10000)))}),
        #     "Current_Positions": Sequence(Tuple(Text(), Box(low=0), Box(low=0)))
        # })
        # self.action_space=Tuple(Discrete(13), Box(0, 5000))
        
        assert render_mode is None or render_mode in self.metadata["render_modes"], "Render mode must be None, human or rgb_array or text"
        self.seed=seed
        
        #Construct agents: Right now only compatible with 1 trading agent
        assert len(kwargs["GymTradingAgent"])==1 and len(kwargs["TradingAgent"])==0, "Kernel simulation can only take a total of 1 agent currently, and it should be a GYM agent"
        self.agents=[]
        if len(kwargs["TradingAgent"])>0:
            pass
        if len(kwargs["GymTradingAgent"])>0:
            for j in kwargs["GymTradingAgent"]:
                new_agent=None
                if j["strategy"]=="Random":
                    new_agent=RandomGymTradingAgent(seed=self.seed, log_events=True, log_to_file=log_to_file, strategy=j["strategy"], Inventory=j["Inventory"], cash=j["cash"], action_freq=j["action_freq"] , rewardpenalty=j["rewardpenalty"], on_trade=True, cashlimit=j["cashlimit"])
                elif j["strategy"] == "TWAP":
                    new_agent = TWAPGymTradingAgent(seed=self.seed, log_events=True, log_to_file=log_to_file, strategy=j["strategy"], Inventory=j["Inventory"], cash=j["cash"], action_freq=j["action_freq"], total_order_size = j["total_order_size"], total_time = j["total_time"], window_size = j["window_size"], side = j["side"], order_target = j["order_target"], on_trade=True)
                else:
                    raise Exception("Program only supports RandomGymTrading Agents for now")      
                self.agents.append(new_agent)
        print("Agent IDs: " + str([agent.id for agent in self.agents]))
        #Construct Exchange:
        ExchangeKwargs=kwargs.get("Exchange")
        exchange=Exchange(**ExchangeKwargs)
        print("Exchange ID: "+ str(exchange.id))
        #Arrival Model setup
        Arrival_model=None
        if kwargs.get("Arrival_model"):
            if kwargs["Arrival_model"].get("name") == "Hawkes":
                params = kwargs["Arrival_model"]["parameters"]
                Arrival_model=HawkesArrival(spread0=exchange.spread, seed=self.seed, **params)
            else:
                raise Exception("Program currently only supports Hawkes Arrival model")
        else:
            raise ValueError("Please specify Arrival model for exchange")
            
               
        
        self.kernel=Kernel(agents=self.agents, exchange=exchange, seed=seed, kernel_name=kernel_name, stop_time=stop_time, wall_time_limit=wall_time_limit, log_to_file=log_to_file, Arrival_model=Arrival_model)
        self.kernel.initialize_kernel()
        np.random.seed(self.seed)
            
    def step(self, action: Optional[Any]):
        """
        Wrapper for state step -- a step ends when the next trade happens or when tau seconds have passed by, whichever occurs first for the agent
        Input: Action
        Output: Observations, Rewards, termination, truncation, Logging info+metrics 
        """
        #Observations=cash, inventory, LOB state, current positions
        simstate=self.kernel.run(action=action)
        Observations=self.getobservations()
        # rewards=self.calculaterewards()
        termination=self.isterminated()
        # truncation=self.istruncated()
        return simstate, Observations, termination
        # return Observations, rewards, termination, truncation
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
    def getobservations(self):
        """
        Returns a dictionary with keys: LOB0, Cash, Inventory, Positions
        """
        return self.kernel.getobservations(agentID=self.agents[0].id)
    def calculaterewards(self):
        rewards={}
        for gymagent in self.kernel.gymagents:
            rewards[gymagent.id]=gymagent.calculaterewards()
        return rewards        
    def isterminated(self):
        return self.kernel.isterminated()==len(self.kernel.gymagents)
    def istruncated(self):
        return self.kernel.istruncated()  
    def getinfo(self):
        """
        Returns auxiliary info: 
        """
        return None
    def getAgent(self, ID):
        return self.kernel.entity_registry[ID]

if __name__=="__main__":
    kwargs={
                "TradingAgent": [],
                "GymTradingAgent": [{"cash": 1000000,
                                    "strategy": "Random",
                                    "action_freq": .2, #makes an action every .2 seconds
                                    "rewardpenalty": 0.4,
                                    "Inventory": {"XYZ": 1000},
                                    "log_to_file": True,
                                    "cashlimit": 100000000}, 
                                     {"cash":100000000,
                                      "strategy": "TWAP",
                                      "total_order_size":100000,
                                      "order_target":"XYZ",
                                      "total_time":3600,
                                      "window_size":900, #window size, measured in seconds
                                      "side":"buy", #buy or sell
                                      "action_freq":.2, 
                                      "Inventory": {} #note: add a checker in the twap agent initialisation which makes sure if its a sell order, that the agent has enough of that particular stock in the inventory
                                     }],
                "Exchange": {"symbol": "XYZ",
                "ticksize":0.01,
                "LOBlevels": 2,
                "numOrdersPerLevel": 10,
                "PriceMid0": 45,
                "spread0": 0.05},
                "Arrival_model": {"name": "Hawkes",
                                  "parameters": {"kernelparams": None, 
                                            "tod": None, 
                                            "Pis": None, 
                                            "beta": None, 
                                            "avgSpread": None, 
                                            "Pi_Q0": None}}

            }
    
    env=tradingEnv(stop_time=100, seed=1, **kwargs)
    print("Initial Observations"+ str(env.getobservations()))
    Simstate, observations, termination=env.step(action=None)
    logger.debug(f"\nSimstate: {Simstate}\nObservations: {observations}\nTermination: {termination}")
    i=0
    while Simstate["Done"]==False and termination!=True:
        logger.debug(f"ENV TERMINATION: {termination}")
        AgentsIDs=[k for k,v in Simstate["Infos"].items() if v==True]
        print(f"Agents with IDs {AgentsIDs} have an action available")
        if len(AgentsIDs)>1:
            raise Exception("Code should be unreachable: Multiple gym agents are not yet implemented")
        agent: GymTradingAgent=env.getAgent(ID=AgentsIDs[0])
        assert isinstance(agent, GymTradingAgent), "Agent with action should be a GymTradingAgent"
        action=(agent.id, agent.get_action(data=observations))   
        print(f"Limit Order Book: {observations['LOB0']}")
        print(f"Action: {action}")
        Simstate, observations, termination=env.step(action=action) 
        logger.debug(f"\nSimstate: {Simstate}\nObservations: {observations}\nTermination: {termination}")
        i+=1
        print(f"ACTION DONE{i}")
    
