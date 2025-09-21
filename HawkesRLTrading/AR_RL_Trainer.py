import sys
import os
sys.path.append(os.path.abspath('/home/ajafree/lobSimulations'))
# sys.path.append(os.path.abspath('/Users/alirazajafree/Documents/GitHub/lobSimulations'))
from HawkesRLTrading.src.Envs.HawkesRLTradingEnv import *
from HawkesRLTrading.src.SimulationEntities.MetaOrderTradingAgents import TWAPGymTradingAgent

import torch

log_dir = '/home/ajafree/TRAINING/logs/'
model_dir = '/home/ajafree/TRAINING/icrl_ppo_model_symmetric'
# log_dir = '/Users/alirazajafree/researchprojects/logs'
# model_dir = '/Users/alirazajafree/researchprojects/models/icrl_ppo_model_symmetric'

start_trading_lag = 100

label = 'train_RLAgent_vs_TWAP_standardised_starttime_repeated_withslippagegraphs'
layer_widths=512
n_layers=1

checkpoint_params = ('20250919_071428_train_RLAgent_vs_TWAP_standardised_starttime', 36)

def graphInventories(beforetwap, withtwap_buy, withtwap_sell, episode_num):
    plt.figure(figsize=(12, 8))
    
    # Flatten the lists of lists to get all inventory values
    all_before = []
    all_buy = []
    all_sell = []
    
    # Flatten beforetwap (list of lists across episodes)
    for episode_inventories in beforetwap:
        all_before.extend(episode_inventories)
    
    # Flatten withtwap_buy (list of lists across episodes) 
    for episode_inventories in withtwap_buy:
        all_buy.extend(episode_inventories)
        
    # Flatten withtwap_sell (list of lists across episodes)
    for episode_inventories in withtwap_sell:
        all_sell.extend(episode_inventories)
    
    # Create normalized histograms (density=True gives probability density)
    # weights parameter normalizes to show ratios/proportions that sum to 1
    if all_before:
        weights_before = np.ones(len(all_before)) / len(all_before)
        plt.hist(all_before, bins=30, alpha=0.7, label=f'Before TWAP (n={len(all_before)})', 
                 color='blue', edgecolor='black', weights=weights_before)
    
    if all_buy:
        weights_buy = np.ones(len(all_buy)) / len(all_buy)
        plt.hist(all_buy, bins=30, alpha=0.7, label=f'With TWAP Buy (n={len(all_buy)})', 
                 color='green', edgecolor='black', weights=weights_buy)
    
    if all_sell:
        weights_sell = np.ones(len(all_sell)) / len(all_sell)
        plt.hist(all_sell, bins=30, alpha=0.7, label=f'With TWAP Sell (n={len(all_sell)})', 
                 color='red', edgecolor='black', weights=weights_sell)
    
    # Add median lines
    if all_before:
        plt.axvline(np.median(all_before), color='blue', linestyle='--', linewidth=2, 
                   label=f'Before Median: {np.median(all_before):.1f}')
    if all_buy:
        plt.axvline(np.median(all_buy), color='green', linestyle='--', linewidth=2,
                   label=f'Buy Median: {np.median(all_buy):.1f}')
    if all_sell:
        plt.axvline(np.median(all_sell), color='red', linestyle='--', linewidth=2,
                   label=f'Sell Median: {np.median(all_sell):.1f}')
    
    plt.xlabel('RL Agent Inventory')
    plt.ylabel('Proportion') 
    plt.title('RL Agent Inventory Distribution: Before vs With TWAP (Normalized)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(log_dir + f'_all_inventory_distributions_episode_{episode_num}.png', dpi=300, bbox_inches='tight')
    plt.close()



# with open("/Users/alirazajafree/researchprojects/otherdata/Symmetric_INTC.OQ_ParamsInferredWCutoffEyeMu_sparseInfer_2019-01-02_2019-12-31_CLSLogLin_10", 'rb') as f: # INTC.OQ_ParamsInferredWCutoff_2019-01-02_2019-03-31_poisson
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
                [(40, 1.)]],
     'Bid_inspread': [0.,
                      [(40, 1.)]],
     'Bid_L1': [0.,
                [(40, 1.)]],
     'Bid_MO': [0.,
                [(40, 1.)]]}
Pis["Ask_MO"] = Pis["Bid_MO"]
Pis["Ask_L1"] = Pis["Bid_L1"]
Pis["Ask_inspread"] = Pis["Bid_inspread"]
Pis["Ask_L2"] = Pis["Bid_L2"]
Pi_Q0= {'Ask_L1': [0.,
                   [(200, 1.)]],
        'Ask_L2': [0.,
                   [(200, 1.)]],
        'Bid_L1': [0.,
                   [(200, 1.)]],
        'Bid_L2': [0.,
                   [(200, 1.)]]}

kwargs={
    "TradingAgent": [],
    "GymTradingAgent": [
                        {"cash": 2500,
                         "strategy": "ICRL",
                         "action_freq": 0.213,
                         "rewardpenalty": 1,
                         "Inventory": {"INTC": 0},
                         "log_to_file": True,
                         "cashlimit": 5000000,
                         "inventorylimit": 25,
                         'start_trading_lag': start_trading_lag,
                         "wake_on_MO": True,
                         "wake_on_Spread": True}
                        # {"cash": 2500,
                        #  "strategy": "Probabilistic",
                        #  "action_freq": 0.213,
                        #  "rewardpenalty": 0.5,
                        #  "Inventory": {"INTC": 0},
                        #  "log_to_file": True,
                        #  "cashlimit": 5000000,
                        #  "inventorylimit": 25,
                        #  'start_trading_lag': 100,
                        #  "wake_on_MO": True,
                        #  "wake_on_Spread": True}
                         ,
                         {"cash":1000000,
                          "cashlimit": 100000000000,
                          "strategy": "TWAP",
                          "on_trade":False,
                          "total_order_size":300,
                          "order_target":"INTC",
                          "total_time":400,
                          "window_size":50, #window size, measured in seconds
                          "action_freq":1,
                          "Inventory": {"INTC":500},
                          'start_trading_lag': start_trading_lag,
                          "wake_on_MO": False,
                          "wake_on_Spread": False}
                          ],
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
j = agents[0]
tc = 0.0001
RLagentInstance = AdversarialPPOAgent( seed=1, log_events=True, log_to_file=True, strategy=j["strategy"], Inventory=j["Inventory"], cash=j["cash"], action_freq=j["action_freq"],
                          wake_on_MO=j["wake_on_MO"], wake_on_Spread=j["wake_on_Spread"], cashlimit=j["cashlimit"],inventorylimit=j['inventorylimit'], batch_size=512,
                          layer_widths=layer_widths, n_layers =n_layers, buffer_capacity = 100000, rewardpenalty = j["rewardpenalty"], epochs = 5, transaction_cost=1e-4, start_trading_lag = j['start_trading_lag'],
                          gae_lambda=0.5, truncation_enabled=False, action_space_config = 1, alt_state=True, enhance_state=False, include_time=False, optim_type='ADAM',entropy_coef=0, exploration_bonus = 0, TWAPPresent=0 , hidden_activation='sigmoid')
# RLagentInstance = ProbabilisticAgent(seed=1, log_events=True, log_to_file=True, strategy=j["strategy"], Inventory=j["Inventory"], cash=j["cash"], action_freq=j["action_freq"],
#                           wake_on_MO=j["wake_on_MO"], wake_on_Spread=j["wake_on_Spread"], cashlimit=j["cashlimit"],inventorylimit=j['inventorylimit'], 
#                           rewardpenalty = 1e-4, transaction_cost=tc, start_trading_lag = j['start_trading_lag'])

inventories_with_twap_buy = []
inventories_with_twap_sell = []
inventories_without_twap = []

j['agent_instance'] = RLagentInstance
kwargs['GymTradingAgent'] = agents
i_eps=0
# cash, inventory, t, actions = [], [], [], []
t = []
avgEpisodicRewards, stdEpisodicRewards, finalcash, finalcash2 = [], [], [], []
train_logger = TrainingLogger(layer_widths=layer_widths, n_layers=n_layers, log_dir=log_dir, label = label)
model_manager = ModelManager(model_dir = model_dir, label = label)
counter_profit = 0
episode_boundaries = [0]
cashs:Dict[int, List] = {}
inventories:Dict[int, List] = {}
actionss:Dict[int, List] = {}
RLagentID = 1
twap_sell_slippages = []
twap_buy_slippages = []

for episode in range(100):
    inventory_with_twap_buy = []
    inventory_with_twap_sell = []
    inventory_without_twap = []
    twap_diff = 0
    starting_midprice = 0
    new_midprice = True
    kwargs["GymTradingAgent"][1]["Inventory"] = {"INTC": 500}
    kwargs["GymTradingAgent"][1]["cash"] = 1000000
    #the time that the TWAP agent will kick in:
    twap_time = 150 + start_trading_lag #int(np.clip(np.random.normal(150, 50), 1, 300)) + start_trading_lag
    RLagentInstance.TWAPPresent = False
    twap_side = np.random.choice(["buy", "sell"])
    kwargs["GymTradingAgent"][1]["start_trading_lag"] = twap_time
    #randomise buy or sell
    kwargs["GymTradingAgent"][1]["side"] = twap_side
    i = 0
    action_num = 0
    env=tradingEnv(stop_time=400, wall_time_limit=23400, **kwargs)
    print(f"Start of episode {episode}. TWAP Time is {twap_time} and side is {twap_side}")
    print("Initial Observations"+ str(env.getobservations()))
    Simstate, observations, termination, truncation =env.step(action=None) 
    AgentsIDs=[k for k,v in Simstate["Infos"].items() if v==True]
    agents:List[GymTradingAgent] = [env.getAgent(ID=agentid) for agentid in AgentsIDs]
    observationsDict:Dict[int, Dict] = {agentid: {"Inventory": agent.Inventory, "Positions": []} for agent, agentid in zip(agents, AgentsIDs)}
    if episode == 0:
        for agent in agents:
            if isinstance(agent, PPOAgent):
                agent.setupNNs(observations)
        
    if checkpoint_params is not None:
        loaded_models = model_manager.load_models(timestamp=checkpoint_params[0], epoch = checkpoint_params[1], d = agent.Actor_Critic_d, u = agent.Actor_Critic_u)
        agent.Actor_Critic_d = loaded_models['d']
        agent.Actor_Critic_u = loaded_models['u']
    logger.debug(f"\nSimstate: {Simstate}\nObservations: {observations}\nTermination: {termination}")
    while Simstate["Done"]==False and termination!=True:
        counter_profit +=1
        logger.debug(f"ENV TERMINATION: {termination}")
        AgentsIDs=[k for k,v in Simstate["Infos"].items() if v==True]
        print(f"Agents with IDs {AgentsIDs} have an action available")
        agents:List[GymTradingAgent] = [env.getAgent(ID=agentid) for agentid in AgentsIDs]
        # action:list[Tuple] = []
        if(Simstate['TimeCode'] > twap_time) and not RLagentInstance.TWAPPresent:
            RLagentInstance.TWAPPresent = -1 if twap_side == 'sell' else 1

        for agent in agents:
            assert isinstance(agent, GymTradingAgent), "Agent with action should be a GymTradingAgent"
            #check if agent is an RL agent or not
            
            if not isinstance(agent, PPOAgent):
                if(new_midprice):
                    starting_midprice = float((observations.get('LOB0').get('Ask_L1')[0] + observations.get('LOB0').get('Bid_L1')[0])/2)
                    new_midprice = False
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
                twap_agent = agent if isinstance(agent, TWAPGymTradingAgent) else None
                if 399 <= Simstate['TimeCode'] <= 400:
                    if twap_agent.side == "sell":
                        # TWAP started with 500 shares, calculate how many were sold
                        total_executed = 500 - twap_agent.Inventory["INTC"]
                        if total_executed > 0:
                            # Calculate cash earned from selling
                            total_earned = twap_agent.cash - 1000000  # Started with 1M cash
                            # Benchmark: what they would have earned selling at starting mid-price
                            benchmark_earned = total_executed * starting_midprice
                            # Slippage: (actual - benchmark) / benchmark
                            slippage = (total_earned - benchmark_earned) / benchmark_earned
                            
                            # Overwrite if we already have data for this episode (prevent duplicates)
                            if len(twap_sell_slippages) > episode:
                                twap_sell_slippages[episode] = slippage
                                print(f"SELL - Overwriting episode {episode} slippage data")
                            else:
                                # Pad with None if needed and add new data
                                while len(twap_sell_slippages) < episode:
                                    twap_sell_slippages.append(None)
                                twap_sell_slippages.append(slippage)
                            
                            print(f"SELL - Executed: {total_executed}, Earned: {total_earned}, Benchmark: {benchmark_earned}, Slippage: {slippage}")
                        else:
                            print("SELL - No executions this episode")
                            
                    elif twap_agent.side == "buy":
                        # TWAP started with 500 shares, calculate how many were bought
                        total_executed = twap_agent.Inventory["INTC"] - 500
                        if total_executed > 0:
                            # Calculate cash spent on buying
                            total_paid = 1000000 - twap_agent.cash  # Started with 1M cash
                            # Benchmark: what they would have paid buying at starting mid-price
                            benchmark_paid = total_executed * starting_midprice
                            # Slippage: (actual - benchmark) / benchmark
                            slippage = (total_paid - benchmark_paid) / benchmark_paid
                            
                            # Overwrite if we already have data for this episode (prevent duplicates)
                            if len(twap_buy_slippages) > episode:
                                twap_buy_slippages[episode] = slippage
                                print(f"BUY - Overwriting episode {episode} slippage data")
                            else:
                                # Pad with None if needed and add new data
                                while len(twap_buy_slippages) < episode:
                                    twap_buy_slippages.append(None)
                                twap_buy_slippages.append(slippage)
                            
                            print(f"BUY - Executed: {total_executed}, Paid: {total_paid}, Benchmark: {benchmark_paid}, Slippage: {slippage}")
                        else:
                            print("BUY - No executions this episode")
                    
                    # Debug output
                    print(f"Total slippages recorded so far - Sell: {len([s for s in twap_sell_slippages if s is not None])}, Buy: {len([s for s in twap_buy_slippages if s is not None])}")
                    
            else:
                action_num+=1
                RLagentID = agent.id
                agentAction:Tuple[int, int] = agent.get_action(data=env.getobservations(agentID=agent.id), epsilon = 0.5 if i_eps < 100 else 0.1)
                action = (agent.id, (agentAction[0],1))
                observations_prev = observationsDict[agent.id].copy() if i != 0 else observations.copy()
                Simstate, observations, termination, truncation=env.step(action=action) #do not try and use this data before this line in the loop
                observationsDict.update({agent.id:observations})
                logger.debug(f"\n Agent: {agent.id}\n Simstate: {Simstate}\nObservations: {observations}\nTermination: {termination}\nTruncation: {truncation}")
                cashs.update({agent.id:cashs.get(agent.id, [])+[observations['Cash']]})
                inventories.update({agent.id:inventories.get(agent.id, []) + [observations['Inventory']]})
                actionss.update({agent.id: actionss.get(agent.id, []) + [action[1][0]]})
                if(Simstate["TimeCode"] > twap_time):
                    if twap_side == "sell":
                        inventory_with_twap_sell.append(observations["Inventory"])
                    else:
                        inventory_with_twap_buy.append(observations["Inventory"])
                else:
                    inventory_without_twap.append(observations["Inventory"])
                t += [Simstate['TimeCode']]
                if 'test' in label:
                    observations['current_time'] = 100+((observations['current_time'] - 100)%300)
                # agent.appendER((agent.readData(observations_prev), agentAction, agent.calculaterewards(termination), agent.readData(observations_prev), (termination or truncation)))
                agent.store_transition(episode, agent.readData(observations_prev), agentAction[1], agent.calculaterewards(termination), agent.readData(observations), (termination or truncation))
                print(f'Current reward: {agent.calculaterewards(termination):0.4f}')
                # print(f'Prev avg reward: {np.mean([r[2] for r in agent.experience_replay[-100:]]):0.4f}')
                i_eps+=1
                logger.debug(f"\nSimstate: {Simstate}\nObservations: {observations}\nTermination: {termination}\nTruncation: {truncation}")
                if len(t) > 0 and Simstate['TimeCode'] < t[-1]:
                    # Episode has reset - mark boundary
                    episode_boundaries.append(len(cashs[agent.id]))
                # Calculate current PnL (cash + inventory value)
                current_pnl = cashs[agent.id][-1] + inventories[agent.id][-1] * agent.mid * (1 - tc*np.sign(inventories[agent.id][-1]))
                finalcash2.append(current_pnl)
                
                #Sharpe ratio
                all_log_returns = []
                pft = np.array(finalcash2)
                # Calculate log returns for each episode separately
                for i in range(len(episode_boundaries)):
                    start_idx = episode_boundaries[i]
                    end_idx = episode_boundaries[i + 1] if i + 1 < len(episode_boundaries) else len(pft)

                    if end_idx - start_idx > 1:  # Need at least 2 points for log returns
                        episode_pnl = pft[start_idx:end_idx]
                        episode_log_returns = np.diff(np.log(episode_pnl))
                        all_log_returns.extend(episode_log_returns)

                # Calculate Sharpe on concatenated log returns from all episodes
                if len(all_log_returns) > 0:
                    all_log_returns = np.array(all_log_returns)
                    sr = np.mean(all_log_returns) / np.std(all_log_returns) if np.std(all_log_returns) > 0 else 0
                else:
                    sr = 0

                # Plotting logic
                if (counter_profit % 100 == 0):
                    plt.figure(figsize=(12, 8))

                    # Create episode-aware plotting
                    pft = np.array(finalcash2)
                    t_array = np.array(t)


                    # Plot each episode as a separate line
                    for i in range(len(episode_boundaries)):
                        start_idx = episode_boundaries[i]
                        end_idx = episode_boundaries[i + 1] if i + 1 < len(episode_boundaries) else len(pft)

                        if end_idx > start_idx:  # Valid episode
                            episode_t = t_array[start_idx:end_idx]
                            episode_pnl = pft[start_idx:end_idx]
                            episode_profit = episode_pnl - j["cash"] # Profit relative to starting capital

                            # Plot this episode
                            if i == len(episode_boundaries) - 1:
                                plt.plot(episode_t, episode_profit, alpha=0.7, label=f'Sharpe:{sr:0.4f}', marker = 'X')
                            else:
                                plt.plot(episode_t, episode_profit, alpha=0.7)

                plt.legend()
                plt.ticklabel_format(useOffset=False, style='plain')
                plt.xlabel('Time in seconds')
                plt.ylabel('Profit in Dollars')
                plt.title('Final Profit - All Episodes Overlaid')
                plt.savefig(log_dir + label + '_profit.png')
                np.save(log_dir + label + '_profit', np.array([t, finalcash2]))
            
            print(agent.current_time)
            print(f"ACTION DONE{action_num}")
            
    if(len(inventory_with_twap_buy) > 0):
        inventories_with_twap_buy.append(inventory_with_twap_buy)
    else:
        inventories_with_twap_sell.append(inventory_with_twap_sell)
    inventories_without_twap.append(inventory_without_twap)
    graphInventories(withtwap_buy = inventories_with_twap_buy, withtwap_sell=inventories_with_twap_sell, beforetwap=inventories_without_twap, episode_num=episode)

    # for agent in agents:
    #     if isinstance(agent, TWAPGymTradingAgent):
    #         if agent.side == "sell":
    #             total_executed = 500 - agent.Inventory["INTC"] 
    #             assert total_executed > 0
    #             total_earned = agent.cash - 1000000
    #             twap_sell_slippages.append((total_earned - total_executed*starting_midprice)/(total_executed*starting_midprice))
    #         else:
    #             total_executed = agent.Inventory["INTC"] - 500
    #             assert total_executed > 0
    #             total_paid = 1000000 - agent.cash
    #             twap_buy_slippages.append((total_paid - total_executed*starting_midprice)/(total_executed*starting_midprice))

# Plot slippages (filter out None values)
    plt.figure(figsize=(10, 6))
    
    valid_sell_slippages = [s for s in twap_sell_slippages if s is not None]
    valid_buy_slippages = [s for s in twap_buy_slippages if s is not None]
    
    if valid_sell_slippages:
        sell_episodes = [i for i, s in enumerate(twap_sell_slippages) if s is not None]
        plt.plot(sell_episodes, [-s for s in valid_sell_slippages], label=f'TWAP Sell Slippage (negated) n={len(valid_sell_slippages)}', marker='o')
    
    if valid_buy_slippages:
        buy_episodes = [i for i, s in enumerate(twap_buy_slippages) if s is not None]
        plt.plot(buy_episodes, valid_buy_slippages, label=f'TWAP Buy Slippage n={len(valid_buy_slippages)}', marker='x')
    
    plt.xlabel('Episode')
    plt.ylabel('Slippage')
    plt.title('TWAP Slippages (Buy vs Sell)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(log_dir + label + '_twap_slippages.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    if termination:
        print("Termination condition reached.")
    elif truncation:
        print("Truncation condition reached.")
    else:
        pass

    if ((episode) % 4 == 0):
        if ('test' not in label) and ((checkpoint_params is None) or (episode >= 0)):
            for epoch in range(1):
                d_policy_loss, d_value_loss, d_entropy_loss, u_policy_loss, u_value_loss, u_entropy_loss = agent.train(train_logger) #, use_CEM = bool((episode+1) % 4))
                train_logger.save_logs()
            train_logger.plot_losses(show=False, save=True)

        model_manager.save_models(epoch = episode, u = agent.Actor_Critic_u, d= agent.Actor_Critic_d)
    for agent in agents:
        if isinstance(agent, PPOAgent):
            agent.current_time = 0
            agent.istruncated = False
            agent.cash = j['cash']
            agent.Inventory = {"INTC": 0}
            agent.positions = {'INTC':{}}
            j['agent_instance'] = agent
            kwargs['GymTradingAgent'][0] = j



    plt.figure(figsize=(12,8))
    plt.subplot(221)
    plt.plot(np.arange(len(cashs[(RLagentID)])), cashs[(RLagentID)])
    plt.title('Cash')
    plt.subplot(222)
    plt.plot(np.arange(len(cashs[(RLagentID)])), inventories[(RLagentID)])
    plt.title('Inventory')
    plt.subplot(223)
    plt.scatter(np.arange(len(cashs[(RLagentID)])), actionss[(RLagentID)])
    plt.yticks(np.arange(0,13), agent.actions)
    plt.title('Actions')
    plt.savefig(log_dir + label+'_policy.png')
    episodic_rewards = []
    r=0
    tmp = agent.trajectory_buffer[0][0]
    for ij in agent.trajectory_buffer:
        if ij[0] == tmp:
            r+=ij[1][3]
        else:
            episodic_rewards.append(r)
            r = ij[1][3]
            tmp=ij[0]
    avgEpisodicRewards.append(np.mean(episodic_rewards[-4:]))
    stdEpisodicRewards.append(np.std(episodic_rewards[-4:]))
    finalcash.append(cashs[(RLagentID)][-1] + inventories[(RLagentID)][-1]*agent.mid )
    pft = np.array(finalcash) - j["cash"]
    ma = np.convolve(pft, np.ones(5)/5, mode='valid')
    plt.figure(figsize=(12,8))
    plt.subplot(311)
    plt.plot(np.arange(len(avgEpisodicRewards)),avgEpisodicRewards)
    plt.fill_between(np.arange(len(avgEpisodicRewards)),np.array(avgEpisodicRewards) - np.array(stdEpisodicRewards),np.array(avgEpisodicRewards) + np.array(stdEpisodicRewards), alpha=0.3  )
    plt.title('Moving Avg Episodic Rewards')
    plt.subplot(312)
    plt.plot(np.arange(len(ma)), ma)
    plt.ticklabel_format(useOffset=False, style='plain')
    plt.title('Final Profit MA')
    plt.subplot(313)
    plt.plot(np.arange(len(pft)), pft)
    plt.ticklabel_format(useOffset=False, style='plain')
    plt.title('Final Profit Raw')

    plt.savefig(log_dir + label+'_avgepisodicreward.png')
    torch.cuda.empty_cache()
    # torch.mps.empty_cache()