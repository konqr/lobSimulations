import sys
sys.path.append("/home/konajain/code/lobSimulations")
from HawkesRLTrading.src.Envs.HawkesRLTradingEnv import *
from HawkesRLTrading.src.Envs.ParallelTrainer import *
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
                         "wake_on_Spread": True,
                         'start_trading_lag' : 100,
                         'agent_instance': None}],
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

trainer = ParallelPPOTrainer(
    n_actors=8,           # Number of parallel actors
    rollout_steps=400,    # T stop_time for each rollout
    ppo_epochs=1,         # Number of PPO epochs per training step
    batch_size=512,
    agent_config=kwargs['GymTradingAgent'][0],
    env_kwargs=kwargs,
    log_dir=log_dir,
    model_dir=model_dir,
    label='parallel_ppo',
    transaction_cost=0.0001
)

# Start training
trainer.train(num_iterations=125)  # 125 * 4 actors = 500 total episodes equivalent

# Get statistics
stats = trainer.get_training_statistics()
print(f"Final average reward: {stats['avg_reward']:.4f} ± {stats['std_reward']:.4f}")