import sys
import os
sys.path.append(os.path.abspath('/home/ajafree/lobSimulations'))
# sys.path.append(os.path.abspath('/Users/alirazajafree/Documents/GitHub/lobSimulations/'))
from HawkesRLTrading.src.Envs.HawkesRLTradingEnv import *
import matplotlib.pyplot as plt

# with open("/Users/alirazajafree/researchprojects/otherdata/INTC.OQ_ParamsInferredWCutoffEyeMu_sparseInfer_Symm_2019-01-02_2019-12-31_CLSLogLin_10", 'rb') as f: # INTC.OQ_ParamsInferredWCutoff_2019-01-02_2019-03-31_poisson
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
kwargs={
            "TradingAgent": [],
            "GymTradingAgent":
                                [{"cash":10000000,
                                "cashlimit": 1000000000,
                                "strategy": "TWAP",
                                "on_trade":False,
                                "total_order_size":1200,
                                "order_target":"INTC",
                                # "participation_rate":0.1,
                                "total_time":1300,
                                "window_size":50, #window size, measured in seconds
                                "side":"buy", #buy or sell
                                "action_freq":1,
                                "Inventory": {"INTC":0},
                                'start_trading_lag': 100,
                                "wake_on_MO": False,
                                "wake_on_Spread": False}],
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
i=0
t = []
inventoryhistories: Dict[int, Dict[int, list]] = {}  # episode -> agent_id -> list of (time, inventory)
cashs:Dict[int, List] = {}
inventories:Dict[int, List] = {}
actionss:Dict[int, List] = {}

starting_midprice:int = 0
price_paths = []
execution_history:List[Tuple] = []
price_paths_non_agent = []
times = []

cash_differences = 0

for episode in range(1):
    env=tradingEnv(stop_time=5000, wall_time_limit=23400, seed=1, **kwargs)
    prev_inventory = 0

    Simstate, observations, termination, truncation =env.step(action=None) 
    AgentsIDs=[k for k,v in Simstate["Infos"].items() if v==True]
    agents:List[GymTradingAgent] = [env.getAgent(ID=agentid) for agentid in AgentsIDs]
    observationsDict:Dict[int, Dict] = {agentid: {"Inventory": agent.Inventory, "Positions": []} for agent, agentid in zip(agents, AgentsIDs)}
    while Simstate["Done"]==False and termination!=True:
        logger.debug(f"ENV TERMINATION: {termination}")
        AgentsIDs=[k for k,v in Simstate["Infos"].items() if v==True]
        print(f"Agents with IDs {AgentsIDs} have an action available")
        agents:List[GymTradingAgent] = [env.getAgent(ID=agentid) for agentid in AgentsIDs]
        action:list[Tuple] = []
        price_paths_non_agent.append(float(observations.get('LOB0').get('Ask_L1')[0] + observations.get('LOB0').get('Bid_L1')[0])/2)
        times.append(Simstate["TimeCode"])
        
        for agent in agents:
            assert isinstance(agent, GymTradingAgent), "Agent with action should be a GymTradingAgent"
            agentAction:Tuple[int, int] = agent.get_action(data=env.getobservations(agentID=agent.id))
            action = (agent.id, agentAction)
            observations_prev = copy.deepcopy(observationsDict.get(agent.id, {}))
            print(f"Limit Order Book: {observationsDict.get(agent.id, {}).get('LOB0', '')}")
            print(f"Inventory: {observationsDict.get(agent.id, {}).get('Inventory', '')}")

            Simstate, observations, termination, truncation=env.step(action=action) #do not try and use this data before this line in the loop

            price_paths.append(float(observations.get('LOB0').get('Ask_L1')[0] + observations.get('LOB0').get('Bid_L1')[0])/2)

            if(i==0):
                starting_midprice = float(observations.get('LOB0').get('Ask_L1')[0] + observations.get('LOB0').get('Bid_L1')[0])/2

            observationsDict.update({agent.id:observations})
            logger.debug(f"\n Agent: {agent.id}\n Simstate: {Simstate}\nObservations: {observations}\nTermination: {termination}\nTruncation: {truncation}")

            cashs.update({agent.id:cashs.get(agent.id, [])+[observations['Cash']]})


            inventories.update({agent.id:inventories.get(agent.id, []) + [observations['Inventory']]})

            diff = abs(inventories[agent.id][-1] - prev_inventory)

            if(diff != 0):
                #inventory has changed, order has gone through
                if kwargs['GymTradingAgent'][agent.id-1]["side"] == 'sell':
                    execution_history.append((observationsDict.get(agent.id, {}).get('LOB0', '').get('Bid_L1'), diff))  
                    cash_differences += (observationsDict.get(agent.id, {}).get('LOB0', '').get('Bid_L1')[0])*diff
                else:
                    execution_history.append((observationsDict.get(agent.id, {}).get('LOB0', '').get('Ask_L1'), diff)) 
                    cash_differences += (observationsDict.get(agent.id, {}).get('LOB0', '').get('Ask_L1')[0])*diff

            prev_inventory = observations['Inventory']
            actionss.update({agent.id: actionss.get(agent.id, []) + [action[1][0]]})
            print(f"ACTION DONE{i}")
            t += [Simstate['TimeCode']]
            i+=1
            if episode not in inventoryhistories:
                inventoryhistories[episode] = {}
            if agent.id not in inventoryhistories[episode]:
                inventoryhistories[episode][agent.id] = []
            inventoryhistories[episode][agent.id].append((Simstate['TimeCode'], observations['Inventory']))

agent_ids = set()
for ep in inventoryhistories:
    agent_ids.update(inventoryhistories[ep].keys())

# final_cash_diff = abs(kwargs["GymTradingAgent"][0]["cash"] - cashs[1][-1])
# print(f"Calculated diff: {cash_differences}. Actual difference: {final_cash_diff}")
price_paths = [p - price_paths[0] for p in price_paths]

# plt.plot(inventories[1], price_paths, alpha=0.5)
# plt.xlabel("Cumulative executed volume")
# plt.ylabel("priceq - pricestart")
# plt.title("Checking SQL for impact of TWAP agent")
# plt.legend(fontsize='small')
# plt.show()

percentage_change_price = [(p - price_paths_non_agent[0])/price_paths_non_agent[0] for p in price_paths_non_agent]

plt.figure()
plt.plot(times, percentage_change_price, alpha=0.5)
plt.xlabel("Time step")
plt.ylabel("Midprice")
plt.title("Price Path Tracking")
plt.savefig("p_change_price_path_tracking_10.png", dpi=300, bbox_inches='tight')
plt.close()

np.save("p_change_price_path_tracking_10.npy", np.array(percentage_change_price))


# for agent_id in agent_ids:
#     plt.figure()
#     for ep in inventoryhistories:
#         if agent_id in inventoryhistories[ep]:
#             history = inventoryhistories[ep][agent_id]
#             times, invs = zip(*history)
#             plt.plot(times, invs, alpha=0.5, label=f"Ep {ep+1}")
#     plt.xlabel("Time")
#     plt.ylabel("Inventory")
#     plt.title(f"Inventory Trajectories Across All Episodes for agent {agent_id}")
#     plt.legend(fontsize='small', ncol=2)
#     plt.show()

            
       
for agent in agents:
    goal = kwargs["GymTradingAgent"][agent.id-1]["total_order_size"]
    final_inventory = inventories[agent.id]
        

            
