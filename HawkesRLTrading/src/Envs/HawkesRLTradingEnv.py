import gymnasium as gym
import numpy as np
import copy
from typing import Any, Optional
import logging
import matplotlib.pyplot as plt
from HawkesRLTrading.src.SimulationEntities.GymTradingAgent import GymTradingAgent, RandomGymTradingAgent
from HawkesRLTrading.src.SimulationEntities.MetaOrderTradingAgents import TWAPGymTradingAgent, POVGymTradingAgent
from HawkesRLTrading.src.SimulationEntities.ImpulseControlAgent import ImpulseControlAgent, ImpulseControlAgentPoisson
from HawkesRLTrading.src.SimulationEntities.ICRLAgent import ICRLAgent, ICRL2, ICRLSG, PPOAgent, AdversarialPPOAgent
from HawkesRLTrading.src.SimulationEntities.ProbabilisticAgent import ProbabilisticAgent
from HawkesRLTrading.src.Stochastic_Processes.Arrival_Models import ArrivalModel, HawkesArrival
from HawkesRLTrading.src.SimulationEntities.Exchange import Exchange
from HawkesRLTrading.src.Kernel import Kernel
from HJBQVI.utils import TrainingLogger, ModelManager, get_gpu_specs
from typing import Dict, List, Tuple
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
        tc = 0.0001
        #Construct agents: Right now only compatible with 1 trading agent
        # assert len(kwargs["GymTradingAgent"])==1 and len(kwargs["TradingAgent"])==0, "Kernel simulation can only take a total of 1 agent currently, and it should be a GYM agent"
        self.agents=[]
        if len(kwargs["TradingAgent"])>0:
            pass
        if len(kwargs["GymTradingAgent"])>0:
            for j in kwargs["GymTradingAgent"]:
                new_agent=None
                if j["strategy"] == "TWAP":
                     new_agent = TWAPGymTradingAgent(seed=self.seed, log_events=True, log_to_file=log_to_file, strategy=j["strategy"], Inventory=j["Inventory"], cash=j["cash"], cashlimit=j["cashlimit"], action_freq=j["action_freq"], total_order_size = j["total_order_size"], total_time = j["total_time"], window_size = j["window_size"], side = j["side"], order_target = j["order_target"], start_trading_lag=j["start_trading_lag"], wake_on_MO=j["wake_on_MO"], wake_on_Spread=j["wake_on_Spread"])
                elif j["strategy"]=="Random":
                    new_agent=RandomGymTradingAgent(seed=self.seed, log_events=True, log_to_file=log_to_file, strategy=j["strategy"], Inventory=j["Inventory"], cash=j["cash"], action_freq=j["action_freq"] , rewardpenalty=j["rewardpenalty"],  wake_on_MO=j["wake_on_MO"], wake_on_Spread=j["wake_on_Spread"], cashlimit=j["cashlimit"])
                elif j['strategy'] == 'ImpulseControl':
                    new_agent = ImpulseControlAgent(j['label'], j['epoch'], j['model_dir'], seed=self.seed, log_events=True, log_to_file=log_to_file, strategy=j["strategy"], Inventory=j["Inventory"], cash=j["cash"], action_freq=j["action_freq"],
                                                    wake_on_MO=j["wake_on_MO"], wake_on_Spread=j["wake_on_Spread"], cashlimit=j["cashlimit"])
                elif j['strategy'] == 'ICRL':
                    new_agent = j['agent_instance']
                elif j['strategy'] == 'POV':
                    new_agent = POVGymTradingAgent(seed=self.seed, log_events=True, log_to_file=log_to_file, strategy=j["strategy"], Inventory=j["Inventory"], cash=j["cash"], cashlimit=j["cashlimit"], action_freq=j["action_freq"], total_order_size = j["total_order_size"], total_time = j["total_time"], window_size = j["window_size"], side = j["side"], order_target = j["order_target"], start_trading_lag=j["start_trading_lag"], participation_rate=j["participation_rate"], wake_on_MO=j["wake_on_MO"], wake_on_Spread=j["wake_on_Spread"])
                elif j['strategy'] == 'Probabilistic':
                    new_agent = ProbabilisticAgent(seed=1, log_events=True, log_to_file=True, strategy=j["strategy"], Inventory=j["Inventory"], cash=j["cash"], action_freq=j["action_freq"],
                          wake_on_MO=j["wake_on_MO"], wake_on_Spread=j["wake_on_Spread"], cashlimit=j["cashlimit"],inventorylimit=j['inventorylimit'], 
                          rewardpenalty = 1e-4, transaction_cost=tc, start_trading_lag = j['start_trading_lag'])
                else:
                    raise Exception("Requested agent not recognised")
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
        Observations= (self.getobservations(agentID=action[0]) if action is not None else self.getobservations())
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
    def getobservations(self, agentID=None):
        """
        Returns a dictionary with keys: LOB0, Cash, Inventory, Positions
        """
        if(agentID is None):
            agentID=self.agents[0].id
        return self.kernel.getobservations(agentID=agentID)
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
        return None
    def getAgent(self, ID):
        return self.kernel.entity_registry[ID]
    def getActiveAgent(self):
        pass

    

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
    log_dir = '/SAN/fca/Konark_PhD_Experiments/icrl/logs'
    model_dir = '/SAN/fca/Konark_PhD_Experiments/icrl/models'
    label = 'PPO_ICRL'
    layer_widths=128
    n_layers=3
    # with open("D:\\PhD\\calibrated params\\INTC.OQ_ParamsInferredWCutoffEyeMu_sparseInfer_Symm_2019-01-02_2019-12-31_CLSLogLin_10", 'rb') as f: # INTC.OQ_ParamsInferredWCutoff_2019-01-02_2019-03-31_poisson
    #     kernelparams = pickle.load(f)
    # kernelparams = preprocessdata(kernelparams)
    # with open("D:\\PhD\\calibrated params\\INTC.OQ_Params_2019-01-02_2019-03-29_dictTOD_constt", 'rb') as f:
    #     tod = pickle.load(f)
    cols= ["lo_deep_Ask", "co_deep_Ask", "lo_top_Ask","co_top_Ask", "mo_Ask", "lo_inspread_Ask" ,
           "lo_inspread_Bid" , "mo_Bid", "co_top_Bid", "lo_top_Bid", "co_deep_Bid","lo_deep_Bid" ]
    # kernelparams = [[np.zeros((12,12))]*4, np.array([[kernelparams[c]] for c in cols])]
    faketod = {}
    for k in cols:
        faketod[k] = {}
        for k1 in np.arange(13):
            faketod[k][k1] = 1.0
    tod=np.zeros(shape=(len(cols), 13))
    for i in range(len(cols)):
        tod[i]=[faketod[cols[i]][k] for k in range(13)]
    Pis={'Bid_L2': [0.,
                    [(1, 1.)]],
         'Bid_inspread': [0.,
                          [(1, 1.)]],
         'Bid_L1': [0.,
                    [(1, 1.)]],
         'Bid_MO': [0.,
                    [(1, 1.)]]}
    Pis["Ask_MO"] = Pis["Bid_MO"]
    Pis["Ask_L1"] = Pis["Bid_L1"]
    Pis["Ask_inspread"] = Pis["Bid_inspread"]
    Pis["Ask_L2"] = Pis["Bid_L2"]
    Pi_Q0= {'Ask_L1': [0.,
                       [(10, 1.)]],
            'Ask_L2': [0.,
                       [(10, 1.)]],
            'Bid_L1': [0.,
                       [(10, 1.)]],
            'Bid_L2': [0.,
                       [(10, 1.)]]}
    kwargs={

                "TradingAgent": [],
                # "GymTradingAgent": [{"cash": 2000,
                #                     "strategy": 'Probabilistic',#"ICRL",

                #                     "action_freq": 1,
                #                     "rewardpenalty": 100,
                #                     "Inventory": {"INTC": 0},
                #                     "log_to_file": True,
                #                     "cashlimit": 5000000,
                #                     "inventorylimit": 1000000,
                #                     "wake_on_MO": False,
                #                     "wake_on_Spread": False}],

                "GymTradingAgent":
                                #    [{"cash":10000000,
                                #     "cashlimit": 1000000000,
                                #      "strategy": "POV",
                                #      "on_trade":False,
                                #      "total_order_size":500,
                                #      "order_target":"XYZ",
                                #      "total_time":200, 
                                #      "window_size":20, #window size, measured in seconds
                                #      "participation_rate":0.05, # in %age
                                #      "side":"buy", #buy or sell
                                #      "action_freq":0.2,
                                #      "Inventory": {"XYZ":1}, #inventory cant be 0
                                #      "wake_on_MO": False,
                                #      "wake_on_Spread": False}],
                                    [
                                    # {"cash": 1000000,
                                    # "strategy": "Random",
                                    # "action_freq": 1.333333,
                                    # "rewardpenalty": 0.4,
                                    # "Inventory": {"XYZ": 1000},
                                    # "wake_on_MO": False,
                                    # "wake_on_Spread": False,
                                    # "log_to_file": True,
                                    # "cashlimit": 100000000},
                                    # ],
                                    {"cash":1000,
                                    "cashlimit": 1000000000,
                                     "strategy": "TWAP",
                                     "on_trade":False,
                                     "total_order_size":200,
                                     "order_target":"XYZ",
                                     "total_time":200,
                                     "window_size":20, #window size, measured in seconds
                                     "side":"sell", #buy or sell
                                     "action_freq":1,
                                     "Inventory": {"XYZ":200}, #inventory cant be 0
                                     "start_trading_lag": 100,
                                     "wake_on_MO": False,
                                     "wake_on_Spread": False}],
                #"GymTradingAgent": [{"cash": 1000000,
                #                    "strategy": "ImpulseControl",x
                #                     'on_trade':True,
                #                    "action_freq": .2,
                #                    "rewardpenalty": 0.4,
                #                    "Inventory": {"INTC":5},
                #                    "log_to_file": True,
                #                    "cashlimit": 100000000,
                #                     'label' : '20250325_160949_INTC_SIMESPPL',
                #                     'epoch':2180,
                #                     'model_dir' : 'D:\\PhD\\calibrated params\\'}],

                "Exchange": {"symbol": "XYZ",
                "ticksize":0.01,
                "LOBlevels": 2,
                "numOrdersPerLevel": 10,
                "PriceMid0": 100,
                "spread0": 0.03},
                "Arrival_model": {"name": "Hawkes",
                                  "parameters": {"kernelparams": None,
                                            "tod": tod,
                                            "Pis": Pis,
                                            "beta": 0.941,
                                            "avgSpread": 0.0101,
                                            "Pi_Q0": Pi_Q0}}

            }
    agents = kwargs['GymTradingAgent']
    # agentInstance = PPOAgent( seed=1, log_events=True, log_to_file=True, strategy=j["strategy"], Inventory=j["Inventory"], cash=j["cash"], action_freq=j["action_freq"],
    #            wake_on_MO=j["wake_on_MO"], wake_on_Spread=j["wake_on_Spread"], cashlimit=j["cashlimit"], batch_size=256, layer_widths=layer_widths, n_layers =n_layers)
    # j['agent_instance'] = agentInstance
    # kwargs['GymTradingAgent'] = [j]
    i=0
    # cash, inventory, t, actions = [], [], [], []
    t = []

    cashs:Dict[int, List] = {}
    inventories:Dict[int, List] = {}
    actionss:Dict[int, List] = {}

    avgEpisodicRewards, stdEpisodicRewards, finalcash =[],[],[]
    # train_logger = TrainingLogger(layer_widths=layer_widths, n_layers=n_layers, log_dir=log_dir, label = label)
    # model_manager = ModelManager(model_dir = model_dir, label = label)

    # for episode in range(10):
    env=tradingEnv(stop_time=300, wall_time_limit=23400, seed=1, **kwargs)
    print("Initial Observations"+ str(env.getobservations()))



    Simstate, observations, termination, truncation =env.step(action=None) 
    AgentsIDs=[k for k,v in Simstate["Infos"].items() if v==True]
    agents:List[GymTradingAgent] = [env.getAgent(ID=agentid) for agentid in AgentsIDs]
    observationsDict:Dict[int, Dict] = {agentid: {"Inventory": agent.Inventory, "Positions": []} for agent, agentid in zip(agents, AgentsIDs)}
    # agent: GymTradingAgent=env.getAgent(ID=AgentsIDs[0])
    # agent.setupNNs(observations)
    logger.debug(f"\nSimstate: {Simstate}\nObservations: {observations}\nTermination: {termination}")

    while Simstate["Done"]==False and termination!=True:
        logger.debug(f"ENV TERMINATION: {termination}")
        AgentsIDs=[k for k,v in Simstate["Infos"].items() if v==True]
        print(f"Agents with IDs {AgentsIDs} have an action available")
        agents:List[GymTradingAgent] = [env.getAgent(ID=agentid) for agentid in AgentsIDs]
        action:list[Tuple] = []
        for agent in agents:
            assert isinstance(agent, GymTradingAgent), "Agent with action should be a GymTradingAgent"

            agentAction:Tuple[int, int] = agent.get_action(data=env.getobservations(agentID=agent.id))
            action = (agent.id, agentAction)
            #print(f"Action: {action}")
            observations_prev = copy.deepcopy(observationsDict.get(agent.id, {}))
            # print(f"Limit Order Book: {observationsDict.get(agent.id, {}).get('LOB0', '')}")
            print(f"Inventory: {observationsDict.get(agent.id).get('Inventory')}")
            Simstate, observations, termination, truncation=env.step(action=action) #do not try and use this data before this line in the loop
            observationsDict.update({agent.id:observations})
            logger.debug(f"\n Agent: {agent.id}\n Simstate: {Simstate}\nObservations: {observations}\nTermination: {termination}\nTruncation: {truncation}")
            # cash += [observations['Cash']]
            cashs.update({agent.id:cashs.get(agent.id, [])+[observations['Cash']]})
            # inventory += [observations['Inventory']]
            inventories.update({agent.id:inventories.get(agent.id, []) + [observations['Inventory']]})
            # actions += [action[1][0]]
            actionss.update({agent.id: actionss.get(agent.id, []) + [action[1][0]]})
            print(f"ACTION DONE{i}")
            t += [Simstate['TimeCode']]
            i+=1
    if termination:
        print("Termination condition reached.")
    elif truncation:
        print("Truncation condition reached.")
    else:
        pass

