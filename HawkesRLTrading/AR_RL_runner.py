import sys
import os
sys.path.append(os.path.abspath('/home/ajafree/lobSimulations'))
# sys.path.append(os.path.abspath('/Users/alirazajafree/Documents/GitHub/lobSimulations'))
from HawkesRLTrading.src.Envs.HawkesRLTradingEnv import *
import torch

log_dir = '/home/ajafree/researchprojects/logs'
model_dir = '/home/ajafree/researchprojects/models/icrl_ppo_model_symmetric'
# log_dir = '/Users/alirazajafree/researchprojects/logs'
# model_dir = '/Users/alirazajafree/researchprojects/models/icrl_ppo_model_symmetric'

label = 'test_RLAgent_vs_SELL_TWAP_10200q_1s'
layer_widths=128
n_layers=3
checkpoint_params = ('20250618_115039_inv10_symmHP_lowEpochs_standard', 52)

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
    "GymTradingAgent": [
                        {"cash": 2500,
                         "strategy": "ICRL",
                         "action_freq": 0.213,
                         "rewardpenalty": 0.5,
                         "Inventory": {"INTC": 0},
                         "log_to_file": True,
                         "cashlimit": 5000000,
                         "inventorylimit": 25,
                         'start_trading_lag': 100,
                         "wake_on_MO": True,
                         "wake_on_Spread": True}
                         ,
                         {"cash":10000000,
                          "cashlimit": 1000000000,
                          "strategy": "TWAP",
                          "on_trade":False,
                          "total_order_size":10200,
                          "order_target":"INTC",
                          "total_time":400,
                          "window_size":50, #window size, measured in seconds
                          "side":"sell", #buy or sell
                          "action_freq":1,
                          "Inventory": {"INTC":10500},
                          'start_trading_lag': 100,
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
RLagentInstance = PPOAgent( seed=1, log_events=True, log_to_file=True, strategy=j["strategy"], Inventory=j["Inventory"], cash=j["cash"], action_freq=j["action_freq"],
                          wake_on_MO=j["wake_on_MO"], wake_on_Spread=j["wake_on_Spread"], cashlimit=j["cashlimit"],inventorylimit=j['inventorylimit'], batch_size=512,
                          layer_widths=layer_widths, n_layers =n_layers, buffer_capacity = 100000, rewardpenalty = 1e-4, epochs = 1000, transaction_cost=tc, start_trading_lag = j['start_trading_lag'],
                          gae_lambda=0.5, truncation_enabled=False, action_space_config = 1, alt_state=True, include_time=False, optim_type='ADAM',entropy_coef=0,lr=1e-5) #, hidden_activation='sigmoid'
j['agent_instance'] = RLagentInstance
kwargs['GymTradingAgent'] = agents
i_eps=0
# cash, inventory, t, actions = [], [], [], []
t = []
avgEpisodicRewards, stdEpisodicRewards , finalcash, finalcash2= [], [],[],[]
train_logger = TrainingLogger(layer_widths=layer_widths, n_layers=n_layers, log_dir=log_dir, label = label)
model_manager = ModelManager(model_dir = model_dir, label = label)
counter_profit = 0
episode_boundaries = [0]
cashs:Dict[int, List] = {}
inventories:Dict[int, List] = {}
actionss:Dict[int, List] = {}
RLagentID = 1

for episode in range(61):
    i = 0
    action_num = 0
    env=tradingEnv(stop_time=400, wall_time_limit=23400, **kwargs)
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
        for agent in agents:
            assert isinstance(agent, GymTradingAgent), "Agent with action should be a GymTradingAgent"
            #check if agent is an RL agent or not
            
            if not isinstance(agent, PPOAgent):
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