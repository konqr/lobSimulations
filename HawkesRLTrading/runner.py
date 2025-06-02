import sys
sys.path.append("/home/konajain/code/lobSimulations")
from HawkesRLTrading.src.Envs.HawkesRLTradingEnv import *
from HJBQVI.utils import get_gpu_specs
import torch
get_gpu_specs()

log_dir = '/SAN/fca/Konark_PhD_Experiments/icrl/logs'
model_dir = '/SAN/fca/Konark_PhD_Experiments/icrl/models'
label = 'PPO_ctstrain_tc'
layer_widths=128
n_layers=3
checkpoint_params = None #('20250525_083145_PPO_weitlgtp_0.5_ICRL_2side_lr3', 231)
with open("/SAN/fca/Konark_PhD_Experiments/extracted/INTC.OQ_ParamsInferredWCutoffEyeMu_sparseInfer_Symm_2019-01-02_2019-12-31_CLSLogLin_10", 'rb') as f: # INTC.OQ_ParamsInferredWCutoff_2019-01-02_2019-03-31_poisson
    kernelparams = pickle.load(f)
kernelparams = preprocessdata(kernelparams)
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

    "GymTradingAgent": [{"cash": 2500,
                         "strategy": "ICRL",

                         "action_freq": 0.2,
                         "rewardpenalty": 100,
                         "Inventory": {"INTC": 0},
                         "log_to_file": True,
                         "cashlimit": 5000000,
                         "inventorylimit": 25,
                         "wake_on_MO": True,
                         "wake_on_Spread": True}],
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
    "Exchange": {"symbol": "INTC",
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
                                     "Pis": Pis,
                                     "beta": 0.941,
                                     "avgSpread": 0.0101,
                                     "Pi_Q0": Pi_Q0,
                                     'expapprox': False}}

}
j = kwargs['GymTradingAgent'][0]
tc = 0.01
agentInstance = PPOAgent( seed=1, log_events=True, log_to_file=True, strategy=j["strategy"], Inventory=j["Inventory"], cash=j["cash"], action_freq=j["action_freq"],
                          wake_on_MO=j["wake_on_MO"], wake_on_Spread=j["wake_on_Spread"], cashlimit=j["cashlimit"],inventorylimit=j['inventorylimit'], batch_size=512,
                          layer_widths=layer_widths, n_layers =n_layers, buffer_capacity = 100000, rewardpenalty = .5, epochs = 1000, transaction_cost=tc, truncation_enabled=False) #, hidden_activation='sigmoid'
j['agent_instance'] = agentInstance
kwargs['GymTradingAgent'] = [j]
i=0
cash, inventory, t, actions = [], [], [], []
avgEpisodicRewards, stdEpisodicRewards , finalcash, finalcash2= [], [],[],[]
train_logger = TrainingLogger(layer_widths=layer_widths, n_layers=n_layers, log_dir=log_dir, label = label)
model_manager = ModelManager(model_dir = model_dir, label = label)
counter_profit = 0
episode_boundaries = [0]
for episode in range(500):
    env=tradingEnv(stop_time=23400, wall_time_limit=23400, **kwargs)
    print("Initial Observations"+ str(env.getobservations()))

    Simstate, observations, termination, truncation =env.step(action=None)
    AgentsIDs=[k for k,v in Simstate["Infos"].items() if v==True]
    agent: GymTradingAgent=env.getAgent(ID=AgentsIDs[0])
    if episode == 0:
        agent.setupNNs(observations,type='separate')
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
        if len(AgentsIDs)>1:
            raise Exception("Code should be unreachable: Multiple gym agents are not yet implemented")
        agent: GymTradingAgent=env.getAgent(ID=AgentsIDs[0])
        assert isinstance(agent, GymTradingAgent), "Agent with action should be a GymTradingAgent"
        agentAction = agent.get_action(data=observations, epsilon = 0.5 if i < 100 else 0.1)
        action=(agent.id, (agentAction[0],1))
        print(f"Limit Order Book: {observations['LOB0']}")
        print(f"Action: {action}")
        observations_prev = observations.copy()
        Simstate, observations, termination, truncation=env.step(action=action)
        # agent.appendER((agent.readData(observations_prev), agentAction, agent.calculaterewards(termination), agent.readData(observations_prev), (termination or truncation)))
        agent.store_transition(episode, agent.readData(observations_prev), agentAction[1], agent.calculaterewards(termination), agent.readData(observations), (termination or truncation))
        print(f'Current reward: {agent.calculaterewards(termination):0.4f}')
        # print(f'Prev avg reward: {np.mean([r[2] for r in agent.experience_replay[-100:]]):0.4f}')
        i+=1
        logger.debug(f"\nSimstate: {Simstate}\nObservations: {observations}\nTermination: {termination}\nTruncation: {truncation}")


        if len(t) > 0 and Simstate['TimeCode'] < t[-1]:
            # Episode has reset - mark boundary
            episode_boundaries.append(len(cash))

        cash += [observations['Cash']]
        inventory += [observations['Inventory']]
        t += [Simstate['TimeCode']]
        actions += [action[1][0]]

        # Calculate current PnL (cash + inventory value)
        current_pnl = cash[-1] + inventory[-1] * agent.mid * (1 - tc*np.sign(inventory[-1]))
        finalcash2.append(current_pnl)

        print(f"ACTION DONE{i}")
        if (counter_profit % 1000 == 0):

            if ('test' not in label) and ((checkpoint_params is None) or (episode >= 0)):
                for epoch in range(1):
                    d_policy_loss, d_value_loss, d_entropy_loss, u_policy_loss, u_value_loss, u_entropy_loss = agent.train(train_logger)
                    train_logger.save_logs()
                train_logger.plot_losses(show=False, save=True)

            model_manager.save_models(epoch = episode, u = agent.Actor_Critic_u, d= agent.Actor_Critic_d)
        # Calculate Sharpe ratio on log returns within episodes
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
                    episode_profit = episode_pnl - 2500  # Profit relative to starting capital

                    # Plot this episode
                    if i == 0:
                        plt.plot(episode_t, episode_profit, alpha=0.7, label=f'Sharpe:{sr:0.4f}')
                    else:
                        plt.plot(episode_t, episode_profit, alpha=0.7)

            plt.legend()
            plt.ticklabel_format(useOffset=False, style='plain')
            plt.xlabel('Time in seconds')
            plt.ylabel('Profit in Dollars')
            plt.title('Final Profit - All Episodes Overlaid')
            plt.savefig(log_dir + label + '_profit.png')
            np.save(log_dir + label + '_profit', np.array([t, finalcash2]))

    if termination:
        print("Termination condition reached.")
    elif truncation:
        print("Truncation condition reached.")
    else:
        pass


    # ER = agent.experience_replay
    agent.current_time = 0
    agent.istruncated = False
    agent.cash = j['cash']
    agent.Inventory = {"INTC": 0}
    agent.positions = {'INTC':{}}
    j['agent_instance'] = agent
    kwargs['GymTradingAgent'] = [j]

    plt.figure(figsize=(12,8))
    plt.subplot(221)
    plt.plot(np.arange(len(cash)), cash)
    plt.title('Cash')
    plt.subplot(222)
    plt.plot(np.arange(len(cash)), inventory)
    plt.title('Inventory')
    plt.subplot(223)
    plt.scatter(np.arange(len(cash)), actions)
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
    avgEpisodicRewards.append(np.mean(episodic_rewards))
    stdEpisodicRewards.append(np.std(episodic_rewards))
    finalcash.append(cash[-1] + inventory[-1]*agent.mid )
    pft = np.array(finalcash) - 2500
    ma = np.convolve(pft, np.ones(5)/5, mode='valid')
    plt.figure(figsize=(12,8))
    plt.subplot(211)
    plt.plot(np.arange(len(avgEpisodicRewards)),avgEpisodicRewards)
    plt.fill_between(np.arange(len(avgEpisodicRewards)),np.array(avgEpisodicRewards) - np.array(stdEpisodicRewards),np.array(avgEpisodicRewards) + np.array(stdEpisodicRewards), alpha=0.3  )
    plt.title('Avg Episodic Rewards')
    plt.subplot(212)
    plt.plot(np.arange(len(ma)), ma)
    plt.ticklabel_format(useOffset=False, style='plain')
    plt.title('Final Profit MA')


    plt.savefig(log_dir + label+'_avgepisodicreward.png')
    torch.cuda.empty_cache()