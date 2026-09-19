"""Phase A1 drift probe: is the background market's price drift a property of
seed 1, or of the simulator?

The fitted Hawkes generator is exactly symmetric under bid/ask reflection (all
12 baselines and all 74 kernels mirror exactly), and the initial book is
symmetric, so the *expected* drift of this market is zero by construction. Any
persistent drift is therefore one of:

  (a) a realisation effect  -- seed 1's particular path happens to drift, in
      which case the ensemble mean over independent seeds is ~0; or
  (b) a simulator asymmetry -- some code path treats bid and ask differently,
      in which case the drift survives averaging over seeds.

This probe measures the agent-free midprice path for one seed per invocation.
Run it over many seeds and test whether the ensemble mean drift is zero: that
is the (a) vs (b) discriminator.

The market is made agent-free by neutering the TWAP agent with off_time=0 --
get_action then returns (12,0) at every wake (MetaOrderTradingAgents.py:54),
so the agent still wakes on schedule (giving a clean 1 Hz sampling clock) but
never sends an order. Agent count, entity IDs and code paths are unchanged
relative to a real run, which keeps the RNG consumption comparable.

Config matches AR_RL_Trainer.py: Pis dirac 40, Pi_Q0 400, spread0 0.03,
mid 100, expApprox=False.

Env vars:
    DP_SEED       explicit seed; otherwise SGE_TASK_ID + DP_SEED_BASE
    DP_SEED_BASE  offset for array jobs (default 0)
    DP_STOP       sim stop time in seconds (default 550, matching the trainer)
    DP_OUT        output directory
    DP_NULL_KERNELS  "1" zeroes the Hawkes excitation mask, reducing the process
                  to pure Poisson at the (exactly symmetric) baselines. Phase A4
                  bisection step 1: if drift survives this, the asymmetry is in
                  the exchange/matching layer and the Hawkes code is exonerated;
                  if it vanishes, the asymmetry is in the excitation path.
"""
import os
import sys
import pickle
import time
import logging

import numpy as np

# --- paths: cluster first, fall back to local Mac ---------------------------
_CLUSTER_REPO = "/home/ajafree/lobSimulations"
_LOCAL_REPO = "/Users/alirazajafree/lobSimulations-1"
REPO = _CLUSTER_REPO if os.path.isdir(_CLUSTER_REPO) else _LOCAL_REPO
sys.path.append(os.path.abspath(REPO))

_PARAM_NAME = ("Symmetric_INTC.OQ_ParamsInferredWCutoffEyeMu_sparseInfer_"
               "2019-01-02_2019-12-31_CLSLogLin_10")
_CLUSTER_PARAMS = os.path.join("/home/ajafree/researchprojects/otherdata", _PARAM_NAME)
_LOCAL_PARAMS = os.path.join("/Users/alirazajafree/researchprojects/otherdata", _PARAM_NAME)
PARAM_FILE = _CLUSTER_PARAMS if os.path.exists(_CLUSTER_PARAMS) else _LOCAL_PARAMS

logging.disable(logging.CRITICAL)
from HawkesRLTrading.src.Envs.HawkesRLTradingEnv import *  # noqa: E402,F401,F403

STOP = float(os.environ.get("DP_STOP", 550))
SEED_BASE = int(os.environ.get("DP_SEED_BASE", 0))
if os.environ.get("DP_SEED"):
    SEED = int(os.environ["DP_SEED"])
else:
    SEED = SEED_BASE + int(os.environ.get("SGE_TASK_ID", 1))
OUT = os.environ.get("DP_OUT", os.path.join(REPO, "HawkesRLTrading", "drift_probe_out"))
os.makedirs(OUT, exist_ok=True)

NULL_KERNELS = os.environ.get("DP_NULL_KERNELS", "0") == "1"

with open(PARAM_FILE, "rb") as f:
    kernelparams = pickle.load(f)
kernelparams = preprocessdata(kernelparams)

if NULL_KERNELS:
    # kernelparams = [[mask, alpha, beta, gamma], baselines]; the excitation
    # enters only as mask*alpha (Arrival_Models.py:324-327 and :356), so zeroing
    # the mask makes every kernel identically zero and leaves a Poisson process
    # at the baselines.
    kernelparams[0][0] = np.zeros_like(kernelparams[0][0])

cols = ["lo_deep_Ask", "co_deep_Ask", "lo_top_Ask", "co_top_Ask", "mo_Ask", "lo_inspread_Ask",
        "lo_inspread_Bid", "mo_Bid", "co_top_Bid", "lo_top_Bid", "co_deep_Bid", "lo_deep_Bid"]
tod = np.ones((len(cols), 13))

Pis = {'Bid_L2': [0., [(40, 1.)]], 'Bid_inspread': [0., [(40, 1.)]],
       'Bid_L1': [0., [(40, 1.)]], 'Bid_MO': [0., [(40, 1.)]]}
Pis["Ask_MO"] = Pis["Bid_MO"]
Pis["Ask_L1"] = Pis["Bid_L1"]
Pis["Ask_inspread"] = Pis["Bid_inspread"]
Pis["Ask_L2"] = Pis["Bid_L2"]
Pi_Q0 = {k: [0., [(400, 1.)]] for k in ('Ask_L1', 'Ask_L2', 'Bid_L1', 'Bid_L2')}

kwargs = {
    "TradingAgent": [],
    # TWAP kept in the config but neutered by off_time=0: it wakes every second
    # (giving the sampling clock) and returns (12,0) every time.
    "GymTradingAgent": [{"cash": 1000000, "cashlimit": 100000000000, "strategy": "TWAP",
                         "on_trade": False, "total_order_size": 150, "order_target": "INTC",
                         "total_time": 150, "window_size": 25, "action_freq": 1,
                         "Inventory": {"INTC": 500}, "start_trading_lag": 0, "side": "buy",
                         "wake_on_MO": False, "wake_on_Spread": False, "off_time": 0}],
    "Exchange": {"symbol": "INTC", "ticksize": 0.01, "LOBlevels": 2,
                 "numOrdersPerLevel": 10, "PriceMid0": 100, "spread0": 0.03},
    "Arrival_model": {"name": "Hawkes",
                      "parameters": {"kernelparams": kernelparams, "tod": tod, "Pis": Pis,
                                     "beta": 0.941, "avgSpread": 0.0101, "Pi_Q0": Pi_Q0,
                                     "expApprox": False}},
}

COLS = ["t", "bid", "ask", "twap_cash", "twap_inv"]


def main():
    t0 = time.time()
    import contextlib, io
    rows = []
    actions = []
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):   # the sim is extremely chatty
        env = tradingEnv(stop_time=STOP, wall_time_limit=23400, seed=SEED, **kwargs)
        Simstate, obs, term, trunc = env.step(action=None)
        agent = None
        while Simstate["Done"] is False and term is not True:
            ids = [k for k, v in Simstate["Infos"].items() if v is True]
            for aid in ids:
                agent = env.getAgent(ID=aid)
                a = agent.get_action(data=env.getobservations(agentID=aid))
                actions.append(a[0] if a is not None else -1)
                lob = obs['LOB0']
                rows.append([Simstate['TimeCode'], lob['Bid_L1'][0], lob['Ask_L1'][0],
                             agent.cash, agent.Inventory['INTC']])
                Simstate, obs, term, trunc = env.step(action=(aid, a))

    # Per-dimension event counts from the arrival model. cols[0:6] are the Ask
    # dimensions and cols[6:12] the Bid dimensions, and cols[6:12] reversed is
    # the mirror of cols[0:6] -- so a symmetric generator must produce
    # counts[i] ~ counts[11-i] for every i.
    am = env.kernel.exchange.Arrival_model
    counts = np.array(am.n, dtype=float).reshape(-1)

    arr = np.array(rows, dtype=float)
    np.savez(os.path.join(OUT, f"path_seed{SEED}.npz"),
             path=arr, counts=counts, cols=np.array(cols),
             null_kernels=np.array([int(NULL_KERNELS)]))

    mid = (arr[:, 1] + arr[:, 2]) / 2.0

    def drift_bps(a, b):
        ia = min(np.searchsorted(arr[:, 0], a), len(mid) - 1)
        ib = min(np.searchsorted(arr[:, 0], b), len(mid) - 1)
        return (mid[ib] - mid[ia]) / mid[ia] * 1e4

    # Sanity: the neutered TWAP must not have traded at all.
    traded = abs(arr[-1, 4] - 500.0)
    non_noop = sum(1 for a in actions if a != 12)

    pairs = " ".join(f"{cols[i].replace('_Ask','')}:{counts[i]:.0f}/{counts[11-i]:.0f}"
                     for i in range(6))

    print(f"seed={SEED} nullk={int(NULL_KERNELS)} n={len(arr)} t=[{arr[0,0]:.0f},{arr[-1,0]:.0f}] "
          f"traded={traded:.0f} non_noop_actions={non_noop} "
          f"drift_0_100={drift_bps(0,100):+.2f} "
          f"drift_100_250={drift_bps(100,250):+.2f} "
          f"drift_250_400={drift_bps(250,400):+.2f} "
          f"drift_400_550={drift_bps(400,STOP-1):+.2f} "
          f"drift_total={drift_bps(0,STOP-1):+.2f} "
          f"[{time.time()-t0:.0f}s]", flush=True)
    print(f"seed={SEED} counts(Ask/Bid) {pairs} totAsk={counts[:6].sum():.0f} "
          f"totBid={counts[6:].sum():.0f}", flush=True)
    if traded != 0 or non_noop != 0:
        print(f"WARNING seed={SEED}: neutered TWAP was not inert "
              f"(traded={traded}, non-noop actions={non_noop})", flush=True)


if __name__ == "__main__":
    main()
