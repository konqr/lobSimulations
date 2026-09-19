"""End-to-end symmetry check with a dumb, exactly symmetric market maker.

A PeggedMMAgent quotes both sides at the touch and holds no view. On a
drift-free market its mark-to-market PnL should straddle zero; on a drifting
one it accumulates inventory on the losing side and bleeds. This tests the
whole stack -- generator, exchange, matching, agent bookkeeping -- without
relying on any theory of where an asymmetry originates.

The agent is symmetric by construction, so it is also its own control: run it
long and it is the market, not the strategy, that decides the sign.

Env vars: MM_SEED / SGE_TASK_ID, MM_SEED_BASE, MM_STOP, MM_NULL_KERNELS,
          MM_SIZE, MM_FREQ, MM_OUT
"""
import contextlib
import io
import logging
import os
import pickle
import sys
import time

import numpy as np

_CLUSTER_REPO = "/home/ajafree/lobSimulations"
_LOCAL_REPO = "/Users/alirazajafree/lobSimulations-1"
REPO = _CLUSTER_REPO if os.path.isdir(_CLUSTER_REPO) else _LOCAL_REPO
sys.path.append(os.path.abspath(REPO))

_PARAM_NAME = ("Symmetric_INTC.OQ_ParamsInferredWCutoffEyeMu_sparseInfer_"
               "2019-01-02_2019-12-31_CLSLogLin_10")
_C = os.path.join("/home/ajafree/researchprojects/otherdata", _PARAM_NAME)
_L = os.path.join("/Users/alirazajafree/researchprojects/otherdata", _PARAM_NAME)
PARAM_FILE = _C if os.path.exists(_C) else _L

logging.disable(logging.CRITICAL)
from HawkesRLTrading.src.Envs.HawkesRLTradingEnv import *  # noqa: E402,F401,F403

STOP = float(os.environ.get("MM_STOP", 550))
SEED = int(os.environ["MM_SEED"]) if os.environ.get("MM_SEED") else \
    int(os.environ.get("MM_SEED_BASE", 0)) + int(os.environ.get("SGE_TASK_ID", 1))
NULL_KERNELS = os.environ.get("MM_NULL_KERNELS", "0") == "1"
SIZE = int(os.environ.get("MM_SIZE", 100))
FREQ = float(os.environ.get("MM_FREQ", 1))
OUT = os.environ.get("MM_OUT", os.path.join(REPO, "HawkesRLTrading", "mm_probe_out"))
os.makedirs(OUT, exist_ok=True)

kernelparams = preprocessdata(pickle.load(open(PARAM_FILE, "rb")))
if NULL_KERNELS:
    kernelparams[0][0] = np.zeros_like(kernelparams[0][0])

cols = ["lo_deep_Ask", "co_deep_Ask", "lo_top_Ask", "co_top_Ask", "mo_Ask", "lo_inspread_Ask",
        "lo_inspread_Bid", "mo_Bid", "co_top_Bid", "lo_top_Bid", "co_deep_Bid", "lo_deep_Bid"]
tod = np.ones((12, 13))
Pis = {k: [0., [(40, 1.)]] for k in ('Bid_L2', 'Bid_inspread', 'Bid_L1', 'Bid_MO',
                                     'Ask_MO', 'Ask_L1', 'Ask_inspread', 'Ask_L2')}
Pi_Q0 = {k: [0., [(400, 1.)]] for k in ('Ask_L1', 'Ask_L2', 'Bid_L1', 'Bid_L2')}

START_CASH = 1000000
START_INV = 500

kwargs = {
    "TradingAgent": [],
    "GymTradingAgent": [{"cash": START_CASH, "cashlimit": 100000000000,
                         "strategy": "PeggedMM", "Inventory": {"INTC": START_INV},
                         "action_freq": FREQ, "wake_on_MO": False, "wake_on_Spread": False,
                         "inventorylimit": 100000, "order_size": SIZE,
                         "max_quotes_per_side": 1, "start_trading_lag": 0}],
    "Exchange": {"symbol": "INTC", "ticksize": 0.01, "LOBlevels": 2,
                 "numOrdersPerLevel": 10, "PriceMid0": 100, "spread0": 0.03},
    "Arrival_model": {"name": "Hawkes",
                      "parameters": {"kernelparams": kernelparams, "tod": tod, "Pis": Pis,
                                     "beta": 0.941, "avgSpread": 0.0101, "Pi_Q0": Pi_Q0,
                                     "expApprox": False}},
}


def main():
    t0 = time.time()
    rows, acts = [], []
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        env = tradingEnv(stop_time=STOP, wall_time_limit=23400, seed=SEED, **kwargs)
        Simstate, obs, term, trunc = env.step(action=None)
        agent = None
        while Simstate["Done"] is False and term is not True:
            for aid in [k for k, v in Simstate["Infos"].items() if v is True]:
                agent = env.getAgent(ID=aid)
                a = agent.get_action(data=env.getobservations(agentID=aid))
                acts.append(a[0] if a is not None else -1)
                lob = obs["LOB0"]
                rows.append([Simstate["TimeCode"], lob["Bid_L1"][0], lob["Ask_L1"][0],
                             agent.cash, agent.Inventory["INTC"]])
                Simstate, obs, term, trunc = env.step(action=(aid, a))

    arr = np.array(rows, dtype=float)
    mid = (arr[:, 1] + arr[:, 2]) / 2.0
    counts = np.array(env.kernel.exchange.Arrival_model.n, dtype=float).reshape(-1)

    d_cash = arr[-1, 3] - START_CASH
    d_inv = arr[-1, 4] - START_INV
    mtm = d_cash + d_inv * mid[-1]                 # marked at the final mid
    drift = 1e4 * (mid[-1] - mid[0]) / mid[0]
    # how much of the PnL is explained by holding inventory through the drift
    inv_pnl = d_inv * (mid[-1] - mid[0])

    n_ask = sum(1 for a in acts if a in (2, 3))
    n_bid = sum(1 for a in acts if a in (8, 9))

    np.savez(os.path.join(OUT, f"mm_seed{SEED}.npz"),
             path=arr, counts=counts, cols=np.array(cols),
             null_kernels=np.array([int(NULL_KERNELS)]),
             summary=np.array([mtm, d_cash, d_inv, drift, inv_pnl, n_ask, n_bid]))

    print(f"seed={SEED} nullk={int(NULL_KERNELS)} mtm={mtm:+.2f} d_cash={d_cash:+.2f} "
          f"d_inv={d_inv:+.0f} drift={drift:+.2f}bps inv_pnl={inv_pnl:+.2f} "
          f"askacts={n_ask} bidacts={n_bid} steps={len(acts)} [{time.time()-t0:.0f}s]",
          flush=True)


if __name__ == "__main__":
    main()
