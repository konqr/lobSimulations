import gymnasium as gym
import numpy as np
from typing import Any, Optional
import logging
import matplotlib.pyplot as plt
from HawkesRLTrading.src.SimulationEntities.GymTradingAgent import GymTradingAgent, RandomGymTradingAgent
from HawkesRLTrading.src.SimulationEntities.MetaOrderTradingAgents import TWAPGymTradingAgent
from HawkesRLTrading.src.SimulationEntities.ImpulseControlAgent import ImpulseControlAgent
from HawkesRLTrading.src.SimulationEntities.ICRLAgent import ICRLAgent
from HawkesRLTrading.src.Stochastic_Processes.Arrival_Models import ArrivalModel, HawkesArrival
from HawkesRLTrading.src.SimulationEntities.Exchange import Exchange
from HawkesRLTrading.src.Kernel import Kernel
import pickle
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
                "GymTradingAgent": [{"cash": 10000000, 
                                    "strategy": "Random",
                                    "action_freq": 2,
                                    "rewardpenalty": 0.4,
                                    "Inventory": {"XYZ": 1000},
                                    "log_to_file": True, 
                                    "cashlimit": 1000000000000,
                                    "inventorylimit": 100000,
                                    "wake_on_MO": True,
                                    "wake_on_Spread": True}],
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
                    new_agent=RandomGymTradingAgent(seed=self.seed, log_events=True, log_to_file=log_to_file, strategy=j["strategy"], Inventory=j["Inventory"], cash=j["cash"], action_freq=j["action_freq"] , wake_on_MO=j["wake_on_MO"], wake_on_Spread=j["wake_on_Spread"], rewardpenalty=j["rewardpenalty"], cashlimit=j["cashlimit"], inventorylimit=j["inventorylimit"])
                elif j["strategy"] == "TWAP":
                     new_agent = TWAPGymTradingAgent(seed=self.seed, log_events=True, log_to_file=log_to_file, strategy=j["strategy"], Inventory=j["Inventory"], cash=j["cash"], cashlimit=j["cashlimit"], action_freq=j["action_freq"], total_order_size = j["total_order_size"], total_time = j["total_time"], window_size = j["window_size"], side = j["side"], order_target = j["order_target"], on_trade=j["on_trade"])
                elif j["strategy"]=="Random":
                    new_agent=RandomGymTradingAgent(seed=self.seed, log_events=True, log_to_file=log_to_file, strategy=j["strategy"], Inventory=j["Inventory"], cash=j["cash"], action_freq=j["action_freq"] , rewardpenalty=j["rewardpenalty"], on_trade=j['on_trade'], cashlimit=j["cashlimit"])
                elif j['strategy'] == 'ImpulseControl':
                    new_agent = ImpulseControlAgent(j['label'], j['epoch'], j['model_dir'], seed=self.seed, log_events=True, log_to_file=log_to_file, strategy=j["strategy"], Inventory=j["Inventory"], cash=j["cash"], action_freq=j["action_freq"],
                                                    on_trade=j['on_trade'], cashlimit=j["cashlimit"])
                elif j['strategy'] == 'ICRL':
                    new_agent = ICRLAgent(self.getobservations(), self, seed=self.seed, log_events=True, log_to_file=log_to_file, strategy=j["strategy"], Inventory=j["Inventory"], cash=j["cash"], action_freq=j["action_freq"],
                                     on_trade=j['on_trade'], cashlimit=j["cashlimit"])
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
        truncation=self.istruncated()
        return simstate, Observations, termination, truncation
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
    def istruncated(self):
        return all(self.kernel.istruncated())
    def isterminated(self):
        return self.kernel.isterminated()  
    def getinfo(self):
        """
        Returns auxiliary info: 
        """
        return None
    def getAgent(self, ID):
        return self.kernel.entity_registry[ID]

if __name__=="__main__":
    with open("D:\\PhD\\calibrated params\\INTC.OQ_ParamsInferredWCutoff_2019-01-02_2019-03-31_poisson", 'rb') as f:
        kernelparams = pickle.load(f)
    # with open("D:\\PhD\\calibrated params\\INTC.OQ_Params_2019-01-02_2019-03-29_dictTOD_constt", 'rb') as f:
    #     tod = pickle.load(f)
    cols= ["lo_deep_Ask", "co_deep_Ask", "lo_top_Ask","co_top_Ask", "mo_Ask", "lo_inspread_Ask" ,
           "lo_inspread_Bid" , "mo_Bid", "co_top_Bid", "lo_top_Bid", "co_deep_Bid","lo_deep_Bid" ]
    kernelparams = [[np.zeros((12,12))]*4, np.array([[kernelparams[c]] for c in cols])]
    faketod = {}
    for k in cols:
        faketod[k] = {}
        for k1 in np.arange(13):
            faketod[k][k1] = 1.0
    tod=np.zeros(shape=(len(cols), 13))
    for i in range(len(cols)):
        tod[i]=[faketod[cols[i]][k] for k in range(13)]
    kwargs={
                "TradingAgent": [],

                "GymTradingAgent": [{"cash": 2000, 
                                    "strategy": "Random",
                                    "action_freq": 2,
                                    "rewardpenalty": 0.4,
                                    "Inventory": {"XYZ": 500},
                                    "log_to_file": True,
                                    "cashlimit": 500000, 
                                    "inventorylimit": 1000000,
                                    "wake_on_MO": True,
                                    "wake_on_Spread": False}],
                # "GymTradingAgent": [{"cash": 1000000,
                #                     "strategy": "Random",
                #                      'on_trade':False,
                #                     "action_freq": .2,
                #                     "rewardpenalty": 0.4,
                #                     "Inventory": {"XYZ": 1000},
                #                     "log_to_file": True,
                #                     "cashlimit": 100000000}],
                #"GymTradingAgent": [{"cash":10000000,
                #                     "cashlimit": 1000000000,
                #                      "strategy": "TWAP",
                #                      "on_trade":False,
                #                      "total_order_size":500,
                #                      "order_target":"XYZ",
                #                      "total_time":100,
                #                      "window_size":20, #window size, measured in seconds
                #                      "side":"buy", #buy or sell
                #                      "action_freq":0.2, 
                #                      "Inventory": {"XYZ":1} #inventory cant be 0
                #                     }],
                "Exchange": {"symbol": "XYZ",
                #"GymTradingAgent": [{"cash": 1000000,
                #                    "strategy": "ImpulseControl",
                #                     'on_trade':True,
                #                    "action_freq": .2,
                #                    "rewardpenalty": 0.4,
                #                    "Inventory": {"INTC":5},
                #                    "log_to_file": True,
                #                    "cashlimit": 100000000,
                #                     'label' : '20250325_160949_INTC_SIMESPPL',
                #                     'epoch':2180,
                #                     'model_dir' : 'D:\\PhD\\calibrated params\\'}],
                #"Exchange": {"symbol": "INTC",
                "ticksize":0.01,
                "LOBlevels": 2,
                "numOrdersPerLevel": 10,
                "PriceMid0": 100,
                "spread0": 0.03},
                "Arrival_model": {"name": "Hawkes",
                                  "parameters": {"kernelparams": kernelparams,
                                            "tod": tod,
                                            "Pis": None, 
                                            "beta": 0.91,
                                            "avgSpread": 0.0101,
                                            "Pi_Q0": None}}

            }
    


    env=tradingEnv(stop_time=200, wall_time_limit=23400, seed=1, **kwargs)
    print("Initial Observations"+ str(env.getobservations()))
    Simstate, observations, termination, truncation =env.step(action=None)
    logger.debug(f"\nSimstate: {Simstate}\nObservations: {observations}\nTermination: {termination}")
    i=0

    cash, inventory, t, actions = [], [], [], []
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
        Simstate, observations, termination, truncation=env.step(action=action) 
        logger.debug(f"\nSimstate: {Simstate}\nObservations: {observations}\nTermination: {termination}\nTruncation: {truncation}")
        i+=1
        cash += [observations['Cash']]
        inventory += [observations['Inventory']]
        t += [Simstate['TimeCode']]
        actions += [action[1][0]]
        print(f"ACTION DONE{i}")

    if termination:
        print("Termination condition reached.")
    elif truncation:
        print("Truncation condition reached.")    
    else:
        pass

    plt.figure(figsize=(12,8))
    plt.subplot(221)
    plt.plot(t, cash)
    plt.title('Cash')
    plt.subplot(222)
    plt.plot(t, inventory)
    plt.title('Inventory')
    plt.subplot(223)
    plt.scatter(t, actions)
    plt.yticks(np.arange(0,13), agent.actions)
    plt.title('Actions')
    plt.show()
