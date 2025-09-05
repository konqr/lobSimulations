import sys
import os
sys.path.append(os.path.abspath('/home/ajafree/lobSimulations'))
# sys.path.append(os.path.abspath('/Users/alirazajafree/Documents/GitHub/lobSimulations/'))
from HawkesRLTrading.src.Envs.HawkesRLTradingEnv import *

log_dir = '/home/ajafree/twap_testing_final/vs_noRL/logs'

label = 'test_no_RL,TWAP'


with open("/home/ajafree/researchprojects/otherdata/Symmetric_INTC.OQ_ParamsInferredWCutoffEyeMu_sparseInfer_2019-01-02_2019-12-31_CLSLogLin_10", 'rb') as f: # INTC.OQ_ParamsInferredWCutoff_2019-01-02_2019-03-31_poisson
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

tc = 0.0001

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

            "GymTradingAgent":[
                                # {"cash": 1000000,
                                # "strategy": "Random",
                                # 'on_trade':False,
                                # "action_freq": 1.3,
                                # "rewardpenalty": 0.4,
                                # "Inventory": {"INTC": 1000},
                                # "wake_on_MO": False,
                                # "wake_on_Spread": False,
                                # "log_to_file": True,
                                # "cashlimit": 100000000,
                                # 'start_trading_lag': 0}, 
                                {"cash":100,
                                "cashlimit": 1000000000,
                                "strategy": "TWAP",
                                "on_trade":False,
                                "total_order_size":300,
                                "order_target":"INTC",
                                "total_time":400,
                                "window_size":50, #window size, measured in seconds
                                "action_freq":1,
                                "Inventory": {"INTC":500},
                                'start_trading_lag': 100,
                                "wake_on_MO": False,
                                "wake_on_Spread": False}
                                ],
                                # {"cash": 2500,
                                # "strategy": "Probabilistic",
                                # "action_freq": 0.213,
                                # "rewardpenalty": 0.5,
                                # "Inventory": {"INTC": 0},
                                # "log_to_file": True,
                                # "cashlimit": 5000000,
                                # "inventorylimit": 25,
                                # 'start_trading_lag': 100,
                                # "wake_on_MO": True,
                                # "wake_on_Spread": True}
                                
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
                                     "Pi_Q0": Pi_Q0,
                                     'expapprox' : True}}

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

avgEpisodicRewards, stdEpisodicRewards, finalcash, finalcash2 =[],[],[], []
# train_logger = TrainingLogger(layer_widths=layer_widths, n_layers=n_layers, log_dir=log_dir, label = label)
# model_manager = ModelManager(model_dir = model_dir, label = label)

# for episode in range(10):
twap_time = int(np.clip(np.random.normal(150, 50), 1, 300)) + 100
kwargs["GymTradingAgent"][1]["start_trading_lag"] = twap_time
#randomise buy or sell
side = np.random.choice(["buy", "sell"])
kwargs["GymTradingAgent"][1]["side"] = side

print(f"Twap time: {twap_time} and side: {side}")
env=tradingEnv(stop_time=400, wall_time_limit=23400, seed=1, **kwargs)
print("Initial Observations"+ str(env.getobservations()))


Simstate, observations, termination, truncation =env.step(action=None) 
AgentsIDs=[k for k,v in Simstate["Infos"].items() if v==True]
agents:List[GymTradingAgent] = [env.getAgent(ID=agentid) for agentid in AgentsIDs]
observationsDict:Dict[int, Dict] = {agentid: {"Inventory": agent.Inventory, "Positions": []} for agent, agentid in zip(agents, AgentsIDs)}
# agent: GymTradingAgent=env.getAgent(ID=AgentsIDs[0])
# agent.setupNNs(observations)
logger.debug(f"\nSimstate: {Simstate}\nObservations: {observations}\nTermination: {termination}")
start_midprices = []
twap_agent_executions_by_episode:Dict[int, List] = {}

for episode in range(61):
    kwargs["GymTradingAgent"][0]["Inventory"] = {"INTC": 500}
    kwargs["GymTradingAgent"][0]["cash"] = 1000000
    twap_side = np.random.choice(["buy", "sell"])
    kwargs["GymTradingAgent"][0]["side"] = twap_side
    twap_agent_executions_by_episode[episode] = []
    i = 0
    action_num = 0
    env=tradingEnv(stop_time=400, wall_time_limit=23400, **kwargs)
    print("Initial Observations"+ str(env.getobservations()))
    Simstate, observations, termination, truncation =env.step(action=None) 
    AgentsIDs=[k for k,v in Simstate["Infos"].items() if v==True]
    agents:List[GymTradingAgent] = [env.getAgent(ID=agentid) for agentid in AgentsIDs]
    observationsDict:Dict[int, Dict] = {agentid: {"Inventory": agent.Inventory, "Positions": []} for agent, agentid in zip(agents, AgentsIDs)}
    start_midprices.append(float((observations.get('LOB0').get('Ask_L1')[0] + observations.get('LOB0').get('Bid_L1')[0])/2))
    if episode == 0:
        for agent in agents:
            if isinstance(agent, PPOAgent):
                agent.setupNNs(observations)
        
    logger.debug(f"\nSimstate: {Simstate}\nObservations: {observations}\nTermination: {termination}")
    prev_inventory = 500
    while Simstate["Done"]==False and termination!=True:
        logger.debug(f"ENV TERMINATION: {termination}")
        AgentsIDs=[k for k,v in Simstate["Infos"].items() if v==True]
        print(f"Agents with IDs {AgentsIDs} have an action available")
        agents:List[GymTradingAgent] = [env.getAgent(ID=agentid) for agentid in AgentsIDs]

        for agent in agents: 
            action_num+=1
            agentAction:Tuple[int, int] = agent.get_action(data=env.getobservations(agentID=agent.id))
            action = (agent.id, agentAction)
            print(f"Action: {action}")
            
            print(f"Limit Order Book: {observationsDict.get(agent.id, {}).get('LOB0', '')}")
            print(f"Inventory: {observationsDict.get(agent.id, {}).get('Inventory', '')}")
            
            Simstate, observations, termination, truncation=env.step(action=action) #do not try and use this data before this line in the loop
            observationsDict.update({agent.id:observations})
            logger.debug(f"\n Agent: {agent.id}\n Simstate: {Simstate}\nObservations: {observations}\nTermination: {termination}\nTruncation: {truncation}")
            cashs.update({agent.id:cashs.get(agent.id, [])+[observations['Cash']]})
            inventories.update({agent.id:inventories.get(agent.id, []) + [observations['Inventory']]})
            actionss.update({agent.id: actionss.get(agent.id, []) + [action[1][0]]})

            diff = abs(inventories[agent.id][-1] - prev_inventory)

            if(diff != 0):
                #inventory has changed, order has gone through
                if twap_side == 'sell':
                    twap_agent_executions_by_episode[episode].append((observationsDict.get(agent.id, {}).get('LOB0', '').get('Bid_L1')[0], diff, twap_side))
                else:
                    twap_agent_executions_by_episode[episode].append((observationsDict.get(agent.id, {}).get('LOB0', '').get('Ask_L1')[0], diff, twap_side))


                prev_inventory = observations['Inventory']

    if termination:
        print("Termination condition reached.")
    elif truncation:
        print("Truncation condition reached.")
    else:
        pass

start_midprices_array = np.array(start_midprices)

executions_data = {}
for episode, executions in twap_agent_executions_by_episode.items():
    if executions:
        executions_data[f"episode_{episode}"] = np.array(executions, dtype=[('price', 'f8'), ('quantity', 'f8'), ('side', 'U4')])

np.save(log_dir + label + '_start_midprices.npy', start_midprices_array)
np.savez(log_dir + label + '_twap_executions.npz', **executions_data)