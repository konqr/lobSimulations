"""TWAP-alone baseline matched EXACTLY to the explo_gae eval runs, WITH per-step observations.

Extends twap_alone_evalmatch.py (cluster copy, commit-matched to AR_RL_runner @4791868)
with a per-step record of the book, the TWAP's cash/inventory and its chosen action,
so the meta-order's execution can be decomposed (passive vs aggressive fills, price
path through the execution window) and compared against the RL-present eval runs.
Slippage/config logic is byte-identical to the original so the summary numbers
reproduce the existing baseline.

Every simulation/agent parameter is copied from AR_RL_runner.py as committed at
4791868, with the single difference that the RL MM agent is absent. This gives
the apples-to-apples "TWAP transaction cost alone vs. in the presence of the RL
market maker" comparison; the paper's rPOV_1 number (7.05/10.36 bps) is NOT
comparable because it used T=300s / window=50s.

Matched to AR_RL_runner.py:
  stop_time      550
  TWAP           Q=150, T=150, window=25, action_freq=1, side=<arg>,
                 start_trading_lag=250 (overridden from 100 at line 353),
                 off_time=400, Inventory 500, cash 1e6
  Exchange       tick 0.01, 2 levels, 10 orders/level, PriceMid0 100, spread0 0.03
  Arrival        Pis dirac 40, Pi_Q0 400, beta 0.941, avgSpread 0.0101,
                 expApprox=False
  slippage       benchmark = mid at TWAP's FIRST WAKE (t~251, verified to equal
                 mid@t=250), total_executed = abs(500 - inventory),
                 bps positive = worse than benchmark on BOTH sides
                 (AR_RL_runner.py:586-595)

Usage:  python3 twap_alone_evalmatch.py <buy|sell> [n_episodes]
"""
import os
import pickle
import sys

import numpy as np

REPO = os.environ.get("TWAP_REPO", "/Users/alirazajafree/lobSimulations-1")
sys.path.append(os.path.abspath(REPO))
PARAM_FILE = os.environ.get(
    "TWAP_PARAM_FILE",
    "/Users/alirazajafree/researchprojects/otherdata/"
    "Symmetric_INTC.OQ_ParamsInferredWCutoffEyeMu_sparseInfer_"
    "2019-01-02_2019-12-31_CLSLogLin_10")
OUT_DIR = os.environ.get(
    "TWAP_OUT_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "twap_alone_evalmatch_out"))
os.makedirs(OUT_DIR, exist_ok=True)

from HawkesRLTrading.src.Envs.HawkesRLTradingEnv import *  # noqa: F401,F403,E402

SIDE = sys.argv[1] if len(sys.argv) > 1 else "buy"
EPISODES = int(sys.argv[2]) if len(sys.argv) > 2 else 17
assert SIDE in ("buy", "sell")

STOP_TIME = int(os.environ.get("TWAP_STOP_TIME", 550))
SEED_BASE = int(os.environ.get("TWAP_SEED_BASE", 1))
VARY_SEED = os.environ.get("TWAP_VARY_SEED", "0") == "1"
TWAP_TIME = 250        # runner's twap_time; also the TWAP's start_trading_lag
TWAP_OFF_TIME = 400
STARTING_CASH = 1_000_000
STARTING_INV = 500

with open(PARAM_FILE, "rb") as f:
    kernelparams = pickle.load(f)
kernelparams = preprocessdata(kernelparams)

cols = ["lo_deep_Ask", "co_deep_Ask", "lo_top_Ask", "co_top_Ask", "mo_Ask", "lo_inspread_Ask",
        "lo_inspread_Bid", "mo_Bid", "co_top_Bid", "lo_top_Bid", "co_deep_Bid", "lo_deep_Bid"]
faketod = {k: {k1: 1.0 for k1 in np.arange(13)} for k in cols}
tod = np.zeros(shape=(len(cols), 13))
for i in range(len(cols)):
    tod[i] = [faketod[cols[i]][k] for k in range(13)]

# runner lines 50-70: Pis dirac 40, Pi_Q0 dirac 400
Pis = {'Bid_L2': [0., [(40, 1.)]],
       'Bid_inspread': [0., [(40, 1.)]],
       'Bid_L1': [0., [(40, 1.)]],
       'Bid_MO': [0., [(40, 1.)]]}
Pis["Ask_MO"] = Pis["Bid_MO"]
Pis["Ask_L1"] = Pis["Bid_L1"]
Pis["Ask_inspread"] = Pis["Bid_inspread"]
Pis["Ask_L2"] = Pis["Bid_L2"]
Pi_Q0 = {'Ask_L1': [0., [(400, 1.)]],
         'Ask_L2': [0., [(400, 1.)]],
         'Bid_L1': [0., [(400, 1.)]],
         'Bid_L2': [0., [(400, 1.)]]}

kwargs = {
    "TradingAgent": [],
    "GymTradingAgent": [{
        "cash": STARTING_CASH,
        "cashlimit": 100000000000,
        "strategy": "TWAP",
        "on_trade": False,
        "total_order_size": 150,
        "order_target": "INTC",
        "total_time": 150,
        "window_size": 25,
        "action_freq": 1,
        "Inventory": {"INTC": STARTING_INV},
        "start_trading_lag": TWAP_TIME,
        "side": SIDE,
        "wake_on_MO": False,
        "wake_on_Spread": False,
        "off_time": TWAP_OFF_TIME,
    }],
    "Exchange": {"symbol": "INTC", "ticksize": 0.01, "LOBlevels": 2,
                 "numOrdersPerLevel": 10, "PriceMid0": 100, "spread0": 0.03},
    "Arrival_model": {"name": "Hawkes",
                      "parameters": {"kernelparams": kernelparams, "tod": tod, "Pis": Pis,
                                     "beta": 0.941, "avgSpread": 0.0101, "Pi_Q0": Pi_Q0,
                                     "expApprox": False}},
}


def mid_of(obs):
    lob = obs.get('LOB0')
    return float((lob.get('Ask_L1')[0] + lob.get('Bid_L1')[0]) / 2)


def slippage_bps(final_cash, executed, benchmark_mid, side):
    """Identical to AR_RL_runner.py:586-595."""
    bench = benchmark_mid * executed
    if side == "sell":
        earned = final_cash - STARTING_CASH
        return (bench - earned) * 10000 / bench
    spent = STARTING_CASH - final_cash
    return (spent - bench) * 10000 / bench


# Per-step record layout (one row per TWAP wake). Kept as a plain float array
# rather than pickled observation dicts: 3 orders of magnitude smaller and it is
# everything the execution-mechanism analysis needs.
OBS_COLS = ["t", "bid", "ask", "bidQ", "askQ", "bidQ_L2", "askQ_L2",
            "cash", "inventory", "action", "n_own_bids", "n_own_asks"]


def obs_row(obs, t, action):
    lob = obs['LOB0']
    pos = obs.get('Positions', {}) or {}
    n_bid = sum(len(v) for k, v in pos.items() if str(k).startswith('Bid'))
    n_ask = sum(len(v) for k, v in pos.items() if str(k).startswith('Ask'))
    return [t, lob['Bid_L1'][0], lob['Ask_L1'][0], lob['Bid_L1'][1], lob['Ask_L1'][1],
            lob['Bid_L2'][1], lob['Ask_L2'][1], obs['Cash'], obs['Inventory'],
            action, n_bid, n_ask]


def run():
    print(f"[twap-alone-evalmatch+obs] side={SIDE} episodes={EPISODES} -> {OUT_DIR}", flush=True)
    slips, cashes, executeds, mids = [], [], [], []
    all_obs = []

    for episode in range(EPISODES):
        kwargs["GymTradingAgent"][0]["Inventory"] = {"INTC": STARTING_INV}
        kwargs["GymTradingAgent"][0]["cash"] = STARTING_CASH
        kwargs["GymTradingAgent"][0]["side"] = SIDE

        # Default (VARY_SEED=0) matches how every runner in this repo has always
        # called tradingEnv: no explicit seed, so __init__ does np.random.seed(1)
        # (HawkesRLTradingEnv.py:126) every episode. With no RL agent to perturb
        # the stream via market feedback, episodes are near-identical — this is
        # the established convention and is what makes the baseline comparable to
        # the eval runs. Set TWAP_VARY_SEED=1 for genuinely independent draws.
        if VARY_SEED:
            env = tradingEnv(stop_time=STOP_TIME, wall_time_limit=23400,
                             seed=SEED_BASE + episode, **kwargs)
        else:
            env = tradingEnv(stop_time=STOP_TIME, wall_time_limit=23400, **kwargs)
        Simstate, observations, termination, truncation = env.step(action=None)

        benchmark_mid = None
        final_cash = STARTING_CASH
        twap_agent = None
        ep_obs = []
        while Simstate["Done"] is False and termination is not True:
            AgentsIDs = [k for k, v in Simstate["Infos"].items() if v is True]
            agents = [env.getAgent(ID=aid) for aid in AgentsIDs]
            for agent in agents:
                if benchmark_mid is None:
                    # first TWAP wake — same capture point as the runner
                    benchmark_mid = mid_of(observations)
                agentAction = agent.get_action(data=env.getobservations(agentID=agent.id))
                pre_obs = env.getobservations(agentID=agent.id)
                ep_obs.append(obs_row(pre_obs, Simstate['TimeCode'],
                                      agentAction[0] if agentAction is not None else -1))
                Simstate, observations, termination, truncation = env.step(
                    action=(agent.id, agentAction))
                twap_agent = agent
                final_cash = agent.cash
                ep_obs.append(obs_row(observations, Simstate['TimeCode'], -2))

        executed = abs(STARTING_INV - twap_agent.Inventory["INTC"])
        slip = slippage_bps(final_cash, executed, benchmark_mid, SIDE)
        slips.append(slip)
        cashes.append(final_cash)
        executeds.append(executed)
        mids.append(benchmark_mid)
        all_obs.append(np.array(ep_obs, dtype=float))
        print(f"[ep {episode:2d}] executed={executed:4.0f}  benchmark_mid={benchmark_mid:.4f}  "
              f"cash={final_cash:.2f}  slippage={slip:+.2f} bps", flush=True)

    slips = np.array(slips)
    np.save(os.path.join(OUT_DIR, f"slippages_{SIDE}.npy"), slips)
    np.save(os.path.join(OUT_DIR, f"final_cash_{SIDE}.npy"), np.array(cashes))
    np.save(os.path.join(OUT_DIR, f"total_executed_{SIDE}.npy"), np.array(executeds))
    np.save(os.path.join(OUT_DIR, f"start_midprices_{SIDE}.npy"), np.array(mids))
    # Build the object array explicitly: np.array(list_of_2d, dtype=object) is
    # shape-dependent (3-D when episodes happen to match, ragged otherwise).
    obs_arr = np.empty(len(all_obs), dtype=object)
    for _i, _a in enumerate(all_obs):
        obs_arr[_i] = _a
    np.save(os.path.join(OUT_DIR, f"stepobs_{SIDE}.npy"), obs_arr, allow_pickle=True)
    with open(os.path.join(OUT_DIR, f"stepobs_{SIDE}_cols.txt"), "w") as fh:
        fh.write(",".join(OBS_COLS) + "\n")
    keep = np.abs(slips - np.median(slips)) < 20
    print(f"\n[{SIDE}] n={len(slips)}  mean {slips.mean():.3f}  median {np.median(slips):.3f}  "
          f"std {slips.std():.3f}")
    print(f"[{SIDE}] excl. {int((~keep).sum())} drift outlier(s): "
          f"mean {slips[keep].mean():.3f} +/- {slips[keep].std():.3f}")


if __name__ == "__main__":
    run()
