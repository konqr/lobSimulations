import gymnasium as gym
import numpy as np
import copy
from typing import Any, Optional
import logging
import matplotlib.pyplot as plt
from HawkesRLTrading.src.SimulationEntities.GymTradingAgent import GymTradingAgent, RandomGymTradingAgent
from HawkesRLTrading.src.SimulationEntities.MetaOrderTradingAgents import TWAPGymTradingAgent
from HawkesRLTrading.src.SimulationEntities.ImpulseControlAgent import ImpulseControlAgent
from HawkesRLTrading.src.SimulationEntities.ICRLAgent import ICRLAgent, ICRL2, ICRLSG, PPOAgent
from HawkesRLTrading.src.SimulationEntities.DiscreteSACGymAgent import DiscreteSACGymTradingAgent
from HawkesRLTrading.src.Stochastic_Processes.Arrival_Models import ArrivalModel, HawkesArrival
from HawkesRLTrading.src.SimulationEntities.Exchange import Exchange
from HawkesRLTrading.src.Kernel import Kernel
from HJBQVI.utils import TrainingLogger, ModelManager, get_gpu_specs
from gymnasium.spaces import Discrete, Dict, Box
import pickle
logging.basicConfig()
logging.getLogger(__name__).setLevel(logging.INFO)
logger = logging.getLogger(__name__)

class tradingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", "text"], "render_fps": 4}

    def __init__(self, render_mode="text", stop_time: int=100, wall_time_limit: int=300, kernel_name: str="Alpha", seed=1, log_to_file=True,**kwargs):
        """
        Initiates a trading environment. Arrivalmodel contains the simulation of the trading orders and params contains all other relevant information
        Relevant arguments:
        render_mode
        stop_time: simulation stop time in simulationseconds
        wall_time_limit: simulation wall time limit (Currently not implemented)
        kernel_name: for human readability
        seed
        log_to_file
        kwargs -- Simulation default parameters specified as following format:
            Dictionary{
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
                            "ticksize":0.01,
                            "LOBlevels": 2,
                            "numOrdersPerLevel": 10,
                            "PriceMid0": 100,
                            "spread0": 0.03},
                "Arrival_model": {"name": "Hawkes",
                                "params": {"kernelparams": None,
                                                "tod": None,
                                                "Pis": None, 
                                                "beta": 0.91,
                                                "avgSpread": 0.0101,
                                                "Pi_Q0": None,
                                                "kernelparamspath": "AAPL.OQ_ParamsInferredWCutoffEyeMu_sparseInfer_Symm_2019-01-02_2019-12-31_CLSLogLin_10",
                                                "todpath": "INTC.OQ_Params_2019-01-02_2019-03-29_dictTOD_constt"}}

            }

        """  
        super().__init__()
        self.stop_time=stop_time            
        self.wall_time_limit=wall_time_limit 
        self.log_to_file=log_to_file

        assert render_mode is None or render_mode in self.metadata["render_modes"], "Render mode must be None, human or rgb_array or text"
        self.seed=seed
        self.action_space=Discrete(13, seed=self.seed)
        #Note observation space is not the state space!!!! Agent only needs to observe the LOB, Spread, Inventory from the environment!

        #Stable baselines does not take dictionaries so need to flatten(need to check patch notes)
        self.observation_space=Dict({
            "Inventory": Box(low=-10, high=10, shape=(1,), dtype=int),
            "Spread": Box(low=-np.inf, high=np.inf, shape=(1,), dtype=float),
            "Trend_var": Box(low=-np.inf, high=np.inf, shape=(1,), dtype=float)
            #Trend var is alpha_t= intensity_(Market order, buy, inspread) + intensity_(Market order, buy, non aggressive) - intensity_(Market order, sell, inspread) - intensity_(Market order, sell, non aggressive)
        })
        self.kwargs=kwargs
        #Construct Exchange:
        ExchangeKwargs=self.kwargs.get("Exchange")
        exchange=Exchange(**ExchangeKwargs)
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
                    new_agent=RandomGymTradingAgent(seed=self.seed, log_events=True, log_to_file=log_to_file, strategy=j["strategy"], Inventory=j["Inventory"], cash=j["cash"], action_freq=j["action_freq"] , rewardpenalty=j["rewardpenalty"],  wake_on_MO=j["wake_on_MO"], wake_on_Spread=j["wake_on_Spread"], cashlimit=j["cashlimit"])
                elif j['strategy'] == 'ImpulseControl':
                    new_agent = ImpulseControlAgent(j['label'], j['epoch'], j['model_dir'], seed=self.seed, log_events=True, log_to_file=log_to_file, strategy=j["strategy"], Inventory=j["Inventory"], cash=j["cash"], action_freq=j["action_freq"],
                                                    wake_on_MO=j["wake_on_MO"], wake_on_Spread=j["wake_on_Spread"], cashlimit=j["cashlimit"])
                elif j['strategy'] == 'ICRL':
                    new_agent = j['agent_instance']
                elif j['strategy'] =="DiscreteSAC":
                    new_agent=DiscreteSACGymTradingAgent(seed=self.seed, log_events=True, log_to_file=log_to_file, strategy=j["strategy"], Inventory=j["Inventory"], cash=j["cash"], action_freq=j["action_freq"] , wake_on_MO=j["wake_on_MO"], wake_on_Spread=j["wake_on_Spread"], rewardpenalty=j["rewardpenalty"], cashlimit=j["cashlimit"], inventorylimit=j["inventorylimit"], hyperparameter_config=j['hyperparameter_config'])
                else:
                    raise Exception("Program only supports RandomGymTrading Agents for now")
                self.agents.append(new_agent)
        print("Agent IDs: " + str([agent.id for agent in self.agents]))
        #Arrival Model setup
        Arrival_model=None
        if kwargs.get("Arrival_model"):
            if kwargs["Arrival_model"].get("name") == "Hawkes":
                params = kwargs["Arrival_model"]["params"]
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
        rewards=self.calculaterewards()
        termination=self.isterminated()
        truncation=self.istruncated()
        infos=self.getinfos()
        return simstate, Observations, rewards, termination, truncation, infos
        #return observations, rewards, dones, infos    
        
        
    def reset(self, seed=None):
        """
        Reset env to default starting state + clear all simulations
        Params: Seed
        Output: Observations, info
        """
        super().reset(seed=self.seed)
        ExchangeKwargs=self.kwargs.get("Exchange")
        exchange=Exchange(**ExchangeKwargs)
        for agent in self.agents:
            agent.reset()
        del self.kernel
        #Reset the Arrival Model
        Arrival_model=None
        if kwargs.get("Arrival_model"):
            if kwargs["Arrival_model"].get("name") == "Hawkes":
                params = kwargs["Arrival_model"]["params"]
                Arrival_model=HawkesArrival(spread0=exchange.spread, seed=self.seed, **params)
            else:
                raise Exception("Program currently only supports Hawkes Arrival model")
        else:
            raise ValueError("Please specify Arrival model for exchange")
        #Make the kernel
        self.kernel=Kernel(agents=self.agents, exchange=exchange, seed=seed, stop_time=self.stop_time, wall_time_limit=self.wall_time_limit, log_to_file=self.log_to_file, Arrival_model=Arrival_model)
        self.kernel.initialize_kernel()
        np.random.seed(self.seed)
        observations=self.getobservations() #observations is inventory, spread, trend variable
        infos=self.getinfo() #contains truncation related info like cash and prices etc
        return observations, infos
    
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
    def getstate(self):
        """
        Returns kernel state
        Returns a dictionary with keys: LOB0, Cash, Inventory, Positions, lobL3, lobL3_sizes
        """
        return self.kernel.getobservations(agentID=self.agents[0].id) 
    #Wrappers
    def getobservations(self):
        #Wrapper to convert kernel observations to gym observations
        tmp=self.getstate()
        observations={
            "Inventory": np.array([tmp["Inventory"]]),
            "Spread": np.array([abs(np.round(tmp['LOB0']['Ask_L1'][0]-tmp['LOB0']['Bid_L1'][0], decimals=2))]),
            "Trend_var": np.array([self.kernel.exchange.Arrival_model.get_alpha_t(self.kernel.current_time)])
        }
        return observations

        
    def getinfos(self):
        tmp=self.getstate()
        infos={
            "LOB0": tmp["LOB0"],
            "Cash": tmp["Cash"],
            "Positions": tmp["Positions"],
            'lobL3': tmp["lobL3"]
        }
        return infos
        
    def calculaterewards(self):
        rewards={}
        for gymagent in self.kernel.gymagents:
            rewards[gymagent.id]=gymagent.calculaterewards()
        return rewards
    def istruncated(self):
        return len(self.kernel.istruncated()) > 0
    def isterminated(self):
        return self.kernel.isterminated()
    def getinfo(self):
        """
        Returns auxiliary info:
        """
        return {}
    def getAgent(self, ID):
        return self.kernel.entity_registry[ID]

def preprocessdata(kernelparams):
    """Takes in params and todpath and spits out corresponding vectorised numpy arrays

    Returns:
    tod: a [12, 13] matrix containing values of f(Q_t), the time multiplier for the 13 different 30 min bins of the trading day.
    params=[kernelparams, baselines]
        kernelparams=params[0]: an array of [12, 12] matrices consisting of mask, alpha0, beta, gamma. the item at arr[i][j] corresponds to the corresponding value from params[cols[i] + "->" + cols[j]]
        So mask=params[0][0], alpha0=params[0][1], beta=params[0][2], gamma=params[0][3]
        baselines=params[1]: a vector of dim=(num_nodes, 1) consisting of baseline intensities
    """

    cols = ["lo_deep_Ask", "co_deep_Ask", "lo_top_Ask","co_top_Ask", "mo_Ask", "lo_inspread_Ask" ,
            "lo_inspread_Bid" , "mo_Bid", "co_top_Bid", "lo_top_Bid", "co_deep_Bid","lo_deep_Bid" ]
    num_nodes = len(cols)


    baselines=np.zeros(shape=(num_nodes, 1)) #vectorising baselines
    for i in range(num_nodes):
        baselines[i]=kernelparams[cols[i]]
        #baselines[i]=data.pop(cols[i], None)


    #params=[mask, alpha, beta, gamma] where each is a 12x12 matrix
    mask, alpha, beta, gamma=[np.zeros(shape=(12, 12)) for _ in range(4)]
    for i in range(num_nodes):
        for j in range(num_nodes):
            kernelParams = kernelparams.get(cols[i] + "->" + cols[j], None)
            if kernelParams is not None:
                mask[i][j]=kernelParams[0]
                alpha[i][j]=kernelParams[1][0]
                beta[i][j]=kernelParams[1][1]
                gamma[i][j]=kernelParams[1][2]
    kernelparams=[mask, alpha, beta, gamma]
    params=[kernelparams, baselines]
    return  params

if __name__=="__main__":
    kwargs={
                "TradingAgent": [],

                # "GymTradingAgent": [{"cash": 800, 
                #                     "strategy": "Random",
                #                     "action_freq": 2,
                #                     "rewardpenalty": 0.4,
                #                     "Inventory": {"XYZ": 4},
                #                     "log_to_file": True,
                #                     "cashlimit": 1000, 
                #                     "inventorylimit": 10,
                #                     "wake_on_MO": True,
                #                     "wake_on_Spread": False}],
                "GymTradingAgent": [{"cash": 800, 
                                    "strategy": "DiscreteSAC",
                                    "action_freq": 2,
                                    "rewardpenalty": 0.4,
                                    "Inventory": {"XYZ": 4},
                                    "log_to_file": True,
                                    "cashlimit": 1000, 
                                    "inventorylimit": 10,
                                    "wake_on_MO": True,
                                    "wake_on_Spread": False,
                                    "hyperparameter_config":{
                                                            'lr':2e-5,
                                                            'minibatch_size': 512,
                                                            'discount_rate': 0.999,
                                                            "tau":0.80,
                                                            "alpha_0":1}
                                    }],
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
                            "ticksize":0.01,
                            "LOBlevels": 2,
                            "numOrdersPerLevel": 11,
                            "PriceMid0": 100,
                            "spread0": 0.04},
                "Arrival_model": {"name": "Hawkes",
                                "params": {"kernelparams": None,
                                                "tod": None,
                                                "Pis": None, 
                                                "beta": 0.94,
                                                "avgSpread": 0.0101,
                                                "Pi_Q0": None,
                                                "kernelparamspath": "HawkesRLTrading/src/AAPL.OQ_ParamsInferredWCutoffEyeMu_sparseInfer_Symm_2019-01-02_2019-12-31_CLSLogLin_10",
                                                "todpath": "HawkesRLTrading/src/INTC.OQ_Params_2019-01-02_2019-03-29_dictTOD_constt"}}

            }
    from stable_baselines3.common.env_checker import check_env
    env=tradingEnv(stop_time=500, seed=2, **kwargs)
    print("Initial observations"+ str(env.getobservations()))
    Simstate, observations, rewards, termination, truncation, infos =env.step(action=None)
    logger.debug(f"\nSimstate: {Simstate}\nObservations: {observations}\nRewards: {rewards}\nTermination: {termination}")
    i=0
    cash, inventory, t, actions = [], [], [], []
    avgEpisodicRewards, stdEpisodicRewards, finalcash =[],[],[]
    train_logger = TrainingLogger(layer_widths=layer_widths, n_layers=n_layers, log_dir=log_dir, label = label)
    # model_manager = ModelManager(model_dir = model_dir, label = label)
    for episode in range(500):
        env=tradingEnv(stop_time=20, wall_time_limit=23400, seed=1, **kwargs)
        print("Initial Observations"+ str(env.getobservations()))

        Simstate, observations, termination, truncation =env.step(action=None)
        AgentsIDs=[k for k,v in Simstate["Infos"].items() if v==True]
        agent: GymTradingAgent=env.getAgent(ID=AgentsIDs[0])
        assert isinstance(agent, GymTradingAgent), "Agent with action should be a GymTradingAgent"
        keys=["Inventory", "Spread", "Trend_var"]
        state=np.concatenate([np.ravel(observations[key][0]) for key in keys]).astype(np.float32)
        action=(agent.id, agent.get_action(state=state))   
        #print(f"Limit Order Book: {observations['LOB0']}")
        print(f"Action: {action}")
        Simstate, next_observations, rewards, termination, truncation, infos=env.step(action=action) 
        next_state=np.concatenate([np.ravel(next_observations[key][0]) for key in keys]).astype(np.float32)
        done=Simstate["Done"]
        transition=(state.copy(), action[1], rewards[AgentsIDs[0]], next_state.copy(), done)
        agent.learn(transition)
        logger.debug(f"\nSimstate: {Simstate}\nObservations: {next_observations}\nTermination: {termination}\nTruncation: {truncation}")
        i+=1
        cash += [infos['Cash']]
        inventory += [observations['Inventory']]
        t += [Simstate['TimeCode']]
        actions += [action[1][0]]
        observations=next_observations
        print(f"ACTION DONE{i}")
    print("Stepcount: ", i)
    # if termination:
    #     print("Termination condition reached.")
    # elif truncation:
    #     print("Truncation condition reached.")    
    # else:
    #     pass

    # plt.figure(figsize=(12,8))
    # plt.subplot(221)
    # plt.plot(t, cash)
    # plt.title('Cash')
    # plt.subplot(222)
    # plt.plot(t, inventory)
    # plt.title('Inventory')
    # plt.subplot(223)
    # plt.scatter(t, actions)
    # plt.yticks(np.arange(0,13), agent.actions)
    # plt.title('Actions')
    # plt.show()
