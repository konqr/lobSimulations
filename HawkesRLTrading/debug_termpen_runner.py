"""
Debug runner: 8 short episodes, sell side, terminal_invpenalty=5*eta.
Purpose: verify that CEM top-5 episode selection works correctly.
"""
import sys
import os
sys.path.append(os.path.abspath('/Users/alirazajafree/lobSimulations-1'))
from HawkesRLTrading.src.Envs.HawkesRLTradingEnv import *

import torch
import random
import logging
logging.disable(logging.CRITICAL)

log_dir = '/Users/alirazajafree/researchprojects/LSTM_fRL/without_expo/testing/self_imitation/onesided/sell/termpen/logs/'
model_dir = '/Users/alirazajafree/researchprojects/LSTM_fRL/without_expo/testing/self_imitation/onesided/sell/termpen/model'

start_trading_lag = 0
twap_off_time = 40
twap_starting_inventory = 30

# Network architecture parameters - MUST match training
layer_widths=100
n_layers=3
eta = 50

label = f'debug_termpensell'

# Sell + terminal penalty checkpoint
checkpoint_params = ("20260323_111256_train_termpen_sell", 44)

with open("/Users/alirazajafree/researchprojects/otherdata/Symmetric_INTC.OQ_ParamsInferredWCutoffEyeMu_sparseInfer_2019-01-02_2019-12-31_CLSLogLin_10", 'rb') as f:
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
                   [(400, 1.)]],
        'Ask_L2': [0.,
                   [(400, 1.)]],
        'Bid_L1': [0.,
                   [(400, 1.)]],
        'Bid_L2': [0.,
                   [(400, 1.)]]}

kwargs={
    "TradingAgent": [],
    "GymTradingAgent": [
                        {"cash": 2500,
                         "strategy": "ICRL",
                         "action_freq": 0.213,
                         "rewardpenalty": 50,
                         "Inventory": {"INTC": 0},
                         "log_to_file": True,
                         "cashlimit": 5000000,
                         "inventorylimit": 25,
                         'start_trading_lag': start_trading_lag,
                         "wake_on_MO": True,
                         "wake_on_Spread": True}
                         ,
                         {"cash":1000000,
                          "cashlimit": 100000000000,
                          "strategy": "TWAP",
                          "on_trade":False,
                          "total_order_size":30,
                          "order_target":"INTC",
                          "total_time":30,
                          "window_size":10,
                          "action_freq":1,
                          "Inventory": {"INTC":500},
                          'start_trading_lag': start_trading_lag,
                          "wake_on_MO": False,
                          "wake_on_Spread": False,
                          "off_time": twap_off_time}
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
                                     'expApprox' : False}}
}

agents = kwargs['GymTradingAgent']
j = agents[0]
tc = 0.0001
RLagentInstance = AdversarialPPOAgent( seed=1, log_events=False, log_to_file=False, strategy=j["strategy"], Inventory=j["Inventory"], cash=j["cash"], action_freq=j["action_freq"],
                          wake_on_MO=j["wake_on_MO"], wake_on_Spread=j["wake_on_Spread"], cashlimit=j["cashlimit"],inventorylimit=j['inventorylimit'], batch_size=512,
                          layer_widths=layer_widths, n_layers =n_layers, buffer_capacity = 100000, rewardpenalty = j["rewardpenalty"], epochs = 100, transaction_cost=1e-4, start_trading_lag = j['start_trading_lag'],
                          gae_lambda=0.5, truncation_enabled=False, action_space_config = 1, alt_state=True, enhance_state=True, include_time=True, optim_type='ADAM',entropy_coef=0, exploration_bonus = 0, hidden_activation='sigmoid',
                          typeNN = "LSTM", lr = 3e-4, chunk_length=64, TWAPPresent=0, cem_full_episode=True, terminal_invpenalty=5*eta)

j['agent_instance'] = RLagentInstance
kwargs['GymTradingAgent'] = agents

model_manager = ModelManager(model_dir = model_dir, label = label)

num_episodes = 8
stop_time = 50
twap_time = 10

nns_setup = False

_real_stdout = sys.stdout
_devnull = open(os.devnull, 'w')
import time as _time
ep0_setup_time = None

for episode in range(num_episodes):
    twap_side = "buy" if episode % 2 == 0 else "sell"
    _real_stdout.write(f"\nEPISODE {episode} (twap_side={twap_side})\n")
    _real_stdout.flush()

    # Reset agent state for new episode (matches AR_RL_Trainer.py lines 497-506)
    RLagentInstance.current_time = 0
    RLagentInstance.istruncated = False
    RLagentInstance.cash = 2500
    RLagentInstance.Inventory = {"INTC": 0}
    RLagentInstance.positions = {'INTC': {}}
    RLagentInstance.last_state = None
    RLagentInstance.last_action = None

    kwargs["GymTradingAgent"][1]["Inventory"] = {"INTC": 30}
    kwargs["GymTradingAgent"][1]["cash"] = 1000000
    kwargs["GymTradingAgent"][1]["side"] = twap_side
    kwargs["GymTradingAgent"][1]["start_trading_lag"] = twap_time

    prev_readData = None
    _t0 = _time.time()
    sys.stdout = _devnull
    env = tradingEnv(stop_time=stop_time, wall_time_limit=23400, **kwargs)
    Simstate, observations, termination, truncation = env.step(action=None)
    sys.stdout = _real_stdout
    _setup_elapsed = _time.time() - _t0
    if episode == 0:
        ep0_setup_time = _setup_elapsed
    _real_stdout.write(f"  env setup: {_setup_elapsed:.1f}s (ep0 was {ep0_setup_time:.1f}s)\n")
    _real_stdout.flush()
    AgentsIDs = [k for k,v in Simstate["Infos"].items() if v==True]
    agents_list: List[GymTradingAgent] = [env.getAgent(ID=agentid) for agentid in AgentsIDs]

    if not nns_setup:
        for agent in agents_list:
            if isinstance(agent, PPOAgent):
                agent.setupNNs(observations)
        if checkpoint_params is not None:
            loaded_models = model_manager.load_models(timestamp=checkpoint_params[0], epoch=checkpoint_params[1], d=agent.Actor_Critic_d, u=agent.Actor_Critic_u)
            agent.Actor_Critic_d = loaded_models['d']
            agent.Actor_Critic_u = loaded_models['u']
        nns_setup = True

    action_num = 0
    _t_loop = _time.time()

    while Simstate["Done"]==False and termination!=True:
        AgentsIDs = [k for k,v in Simstate["Infos"].items() if v==True]
        agents_list = [env.getAgent(ID=agentid) for agentid in AgentsIDs]

        if isinstance(RLagentInstance, AdversarialPPOAgent):
            if twap_off_time >= Simstate['TimeCode'] >= twap_time:
                if not RLagentInstance.TWAPPresent:
                    RLagentInstance.TWAPPresent = -1 if twap_side == 'sell' else 1
            else:
                RLagentInstance.TWAPPresent = 0

        for agent in agents_list:
            assert isinstance(agent, GymTradingAgent)

            if not isinstance(agent, PPOAgent):
                agentAction = agent.get_action(data=env.getobservations(agentID=agent.id))
                action = (agent.id, agentAction)
                sys.stdout = _devnull
                Simstate, observations, termination, truncation = env.step(action=action)
                sys.stdout = _real_stdout
            else:
                action_num += 1
                agentAction = agent.get_action(data=env.getobservations(agentID=agent.id), epsilon=0.1)
                action = (agent.id, (agentAction[0], 1))
                sys.stdout = _devnull
                Simstate, observations, termination, truncation = env.step(action=action)
                sys.stdout = _real_stdout
                current_readData = agent.readData(observations)
                if prev_readData is not None:
                    agent.store_transition(episode, prev_readData, agentAction[1], agent.calculaterewards(termination), current_readData, (termination or truncation))
                prev_readData = current_readData
                _real_stdout.write(f"    t={Simstate['TimeCode']:.1f} act={agentAction[0]} inv={RLagentInstance.countInventory()} r={agent.calculaterewards(termination):.2f}\n")
                _real_stdout.flush()

    _loop_elapsed = _time.time() - _t_loop
    # Episode summary
    ep_transitions = [tr[1] for tr in RLagentInstance.trajectory_buffer if tr[0] == episode]
    ep_total_reward = sum(t[3] for t in ep_transitions)
    print(f"  Episode {episode}: {len(ep_transitions)} transitions, total_reward={ep_total_reward:.4f}, final_inv={RLagentInstance.countInventory()}, loop={_loop_elapsed:.1f}s")

# ---- DEBUG: CEM elite episode selection ----
print("\n" + "="*80)
print("DEBUG: CEM top-5 episode selection")
print("="*80)

buf = RLagentInstance.trajectory_buffer
print(f"Total transitions in buffer: {len(buf)}")

# Show per-episode totals
episodes_in_buf = {}
for ep, tr in buf:
    if ep not in episodes_in_buf:
        episodes_in_buf[ep] = []
    episodes_in_buf[ep].append(tr)

print("\nPer-episode total rewards:")
for ep in sorted(episodes_in_buf.keys()):
    total = sum(t[3] for t in episodes_in_buf[ep])
    print(f"  Episode {ep}: {len(episodes_in_buf[ep])} transitions, total_reward={total:.4f}")

elite_segments = RLagentInstance.get_max_contiguous_rewards(K=10)
print(f"\nCEM selection results:")
for ep_id in sorted(elite_segments.keys()):
    seg = elite_segments[ep_id]
    if seg is not None:
        start, end, total = seg
        print(f"  Episode {ep_id}: SELECTED (full episode [{start}, {end}], total={total:.4f})")
    else:
        print(f"  Episode {ep_id}: SKIPPED")

selected = sum(1 for s in elite_segments.values() if s is not None)
skipped = sum(1 for s in elite_segments.values() if s is None)
print(f"\nSelected: {selected}, Skipped: {skipped}")

# ---- DEBUG: CEM training pass ----
print("\n" + "="*80)
print("DEBUG: Running CEM training pass")
print("="*80)

train_logger = TrainingLogger(layer_widths=layer_widths, n_layers=n_layers, log_dir=log_dir, label=label)

old_epochs = RLagentInstance.epochs
RLagentInstance.epochs = 1
sys.stdout = _devnull
RLagentInstance.train(train_logger, use_CEM=True)
sys.stdout = _real_stdout
RLagentInstance.epochs = old_epochs

print("\n" + "="*80)
print("DEBUG: CEM training pass complete")
print("="*80)
