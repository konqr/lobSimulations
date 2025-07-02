import sys
import os
sys.path.append(os.path.abspath('/Users/alirazajafree/Documents/GitHub/lobSimulations/'))
from HawkesRLTrading.src.Envs.HawkesRLTradingEnv import *

log_dir = '/SAN/fca/Konark_PhD_Experiments/icrl/logs'
model_dir = '/SAN/fca/Konark_PhD_Experiments/icrl/models'
label = 'PPO_ICRL'
layer_widths=128
n_layers=3
with open("/Users/alirazajafree/researchprojects/otherdata/INTC.OQ_ParamsInferredWCutoffEyeMu_sparseInfer_Symm_2019-01-02_2019-12-31_CLSLogLin_10", 'rb') as f: # INTC.OQ_ParamsInferredWCutoff_2019-01-02_2019-03-31_poisson
    kernelparams = pickle.load(f)
kernelparams = preprocessdata(kernelparams)
cols= ["lo_deep_Ask", "co_deep_Ask", "lo_top_Ask","co_top_Ask", "mo_Ask", "lo_inspread_Ask" ,
        "lo_inspread_Bid" , "mo_Bid", "co_top_Bid", "lo_top_Bid", "co_deep_Bid","lo_deep_Bid" ]
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
                            #      "total_order_size":100,
                            #      "order_target":"INTC",
                            #      "total_time":500, 
                            #      "window_size":20, #window size, measured in seconds
                            #      "participation_rate":0.1,
                            #      "side":"buy", #buy or sell
                            #      "action_freq":2,
                            #      "Inventory": {"INTC":1}, #inventory cant be 0
                            #      "wake_on_MO": False,
                            #      "wake_on_Spread": False}],
                                [
                                #     {"cash": 1000000,
                                # "strategy": "Random",
                                #     'on_trade':False,
                                # "action_freq": 1.3,
                                # "rewardpenalty": 0.4,
                                # "Inventory": {"INTC": 1000},
                                # "wake_on_MO": False,
                                # "wake_on_Spread": False,
                                # "log_to_file": True,
                                # "cashlimit": 100000000},
                                {"cash":10000000,
                                "cashlimit": 1000000000,
                                "strategy": "POV",
                                "on_trade":False,
                                "total_order_size":300,
                                "order_target":"INTC",
                                "participation_rate":0.1,
                                "total_time":400,
                                "window_size":50, #window size, measured in seconds
                                "side":"buy", #buy or sell
                                "action_freq":1,
                                "Inventory": {"INTC":0},
                                'start_trading_lag': 100,
                                "wake_on_MO": False,
                                "wake_on_Spread": False}],
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

            "Exchange": {"symbol": "INTC",
            "ticksize":0.01,
            "LOBlevels": 2,
            "numOrdersPerLevel": 10,
            "PriceMid0": 100,
            "spread0": 0.03},
            "Arrival_model": {"name": "Hawkes",
                                "parameters": {"kernelparams": kernelparams,
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
env=tradingEnv(stop_time=400, wall_time_limit=23400, seed=1, **kwargs)
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
        print(f"Limit Order Book: {observationsDict.get(agent.id, {}).get('LOB0', '')}")
        print(f"Inventory: {observationsDict.get(agent.id, {}).get('Inventory', '')}")
        
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
        print(f"Final inventory: {agent.Inventory}")

if termination:
    print("Termination condition reached.")
elif truncation:
    print("Truncation condition reached.")
else:
    pass
