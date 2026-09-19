"""Runs must be bit-reproducible from their seed, and must differ across seeds.

Two defects broke this before Phase B:

1. `Exchange.py:473` picks which resting order to cancel with the stdlib
   `random`, which was never seeded. `tradingEnv` seeded numpy only.
2. `TradingAgent.__init__` stored the caller's `Inventory` dict by reference
   and then mutated it in place as the agent traded, so a caller reusing its
   kwargs carried one episode's closing inventory into the next.

Of the two, (2) is the load-bearing one: with the Inventory copy in place this
test passes even with the stdlib RNG left unseeded, in this configuration. The
stdlib seeding is kept because an unseeded global RNG in the simulation path is
a reproducibility hazard regardless -- a longer run, or one that exercises the
cancel path with more than one eligible order, may well depend on it -- but it
was not what broke reproducibility here.

This is an integration test on purpose: it runs the real simulator twice.

Needs the fitted-parameter pickle; skips cleanly if it is not present.
"""
import contextlib
import io
import logging
import os
import pickle
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

_NAME = ("Symmetric_INTC.OQ_ParamsInferredWCutoffEyeMu_sparseInfer_"
         "2019-01-02_2019-12-31_CLSLogLin_10")
_PATHS = [os.path.join("/home/ajafree/researchprojects/otherdata", _NAME),
          os.path.join(os.path.expanduser("~"), "researchprojects/otherdata", _NAME)]
PARAMS = next((p for p in _PATHS if os.path.exists(p)), None)

logging.disable(logging.CRITICAL)


def _kwargs():
    from HawkesRLTrading.src.Envs.HawkesRLTradingEnv import preprocessdata
    kp = preprocessdata(pickle.load(open(PARAMS, "rb")))
    tod = np.ones((12, 13))
    Pis = {k: [0., [(40, 1.)]] for k in ("Bid_L2", "Bid_inspread", "Bid_L1", "Bid_MO",
                                         "Ask_MO", "Ask_L1", "Ask_inspread", "Ask_L2")}
    PiQ0 = {k: [0., [(400, 1.)]] for k in ("Ask_L1", "Ask_L2", "Bid_L1", "Bid_L2")}
    return {"TradingAgent": [],
            "GymTradingAgent": [{"cash": 1000000, "cashlimit": 10 ** 11,
                                 "strategy": "PeggedMM", "Inventory": {"INTC": 500},
                                 "action_freq": 1, "wake_on_MO": False,
                                 "wake_on_Spread": False, "inventorylimit": 100000,
                                 "order_size": 50, "max_quotes_per_side": 1,
                                 "start_trading_lag": 0}],
            "Exchange": {"symbol": "INTC", "ticksize": 0.01, "LOBlevels": 2,
                         "numOrdersPerLevel": 10, "PriceMid0": 100, "spread0": 0.03},
            "Arrival_model": {"name": "Hawkes",
                              "parameters": {"kernelparams": kp, "tod": tod, "Pis": Pis,
                                             "beta": 0.941, "avgSpread": 0.0101,
                                             "Pi_Q0": PiQ0, "expApprox": False}}}


def _run(seed, kwargs, T=25):
    from HawkesRLTrading.src.Envs.HawkesRLTradingEnv import tradingEnv
    rows = []
    with contextlib.redirect_stdout(io.StringIO()):
        env = tradingEnv(stop_time=T, wall_time_limit=23400, seed=seed, **kwargs)
        S, o, term, _ = env.step(action=None)
        while S["Done"] is False and term is not True:
            for aid in [k for k, v in S["Infos"].items() if v is True]:
                ag = env.getAgent(ID=aid)
                a = ag.get_action(data=env.getobservations(agentID=aid))
                rows.append((S["TimeCode"], o["LOB0"]["Bid_L1"][0], o["LOB0"]["Ask_L1"][0],
                             ag.cash, ag.Inventory["INTC"], a[0]))
                S, o, term, _ = env.step(action=(aid, a))
    return rows


def test_same_seed_is_bit_reproducible():
    if PARAMS is None:
        print("SKIP (no parameter file)")
        return
    k = _kwargs()
    a, b = _run(7, k), _run(7, k)     # deliberately REUSES kwargs, as callers do
    assert a == b, "same seed produced different paths"


def test_different_seeds_differ():
    if PARAMS is None:
        print("SKIP (no parameter file)")
        return
    k = _kwargs()
    assert _run(7, k) != _run(8, k), "different seeds produced identical paths"


def test_inventory_kwargs_not_mutated():
    """The caller's dict must survive a run unchanged -- the aliasing bug."""
    if PARAMS is None:
        print("SKIP (no parameter file)")
        return
    k = _kwargs()
    before = dict(k["GymTradingAgent"][0]["Inventory"])
    _run(7, k)
    assert k["GymTradingAgent"][0]["Inventory"] == before, \
        f"kwargs Inventory mutated: {before} -> {k['GymTradingAgent'][0]['Inventory']}"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
    print("all tests passed")
