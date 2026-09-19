"""Phase A4 step 1b: run the Hawkes point process in ISOLATION from the exchange.

The full simulation shows a systematic Ask excess in the deep (L2) event types
that is present whether or not the Hawkes excitation is on. This probe removes
the exchange entirely and pins the spread, so the only thing left is the point
process. If the deep-level asymmetry survives, it is a defect in the arrival
model; if it vanishes, it requires the exchange feedback loop.

Env vars: GP_SEED / SGE_TASK_ID, GP_SEED_BASE, GP_T, GP_SPREAD, GP_NULL_KERNELS, GP_OUT
"""
import os
import sys
import pickle
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
_CLUSTER_REPO = "/home/ajafree/lobSimulations"
_LOCAL_REPO = "/Users/alirazajafree/lobSimulations-1"
REPO = _CLUSTER_REPO if os.path.isdir(_CLUSTER_REPO) else _LOCAL_REPO
sys.path.append(os.path.abspath(REPO))

_NAME = ("Symmetric_INTC.OQ_ParamsInferredWCutoffEyeMu_sparseInfer_"
         "2019-01-02_2019-12-31_CLSLogLin_10")
_C = os.path.join("/home/ajafree/researchprojects/otherdata", _NAME)
_L = os.path.join("/Users/alirazajafree/researchprojects/otherdata", _NAME)
PARAM_FILE = _C if os.path.exists(_C) else _L

from HawkesRLTrading.src.Envs.HawkesRLTradingEnv import preprocessdata  # noqa: E402
from HawkesRLTrading.src.Stochastic_Processes.Arrival_Models import HawkesArrival  # noqa: E402

T = float(os.environ.get("GP_T", 550))
SPREAD = float(os.environ.get("GP_SPREAD", 0.02))
NULL_KERNELS = os.environ.get("GP_NULL_KERNELS", "0") == "1"
SEED = int(os.environ["GP_SEED"]) if os.environ.get("GP_SEED") else \
    int(os.environ.get("GP_SEED_BASE", 0)) + int(os.environ.get("SGE_TASK_ID", 1))
OUT = os.environ.get("GP_OUT", os.path.join(REPO, "HawkesRLTrading", "generator_probe_out"))
os.makedirs(OUT, exist_ok=True)

cols = ["lo_deep_Ask", "co_deep_Ask", "lo_top_Ask", "co_top_Ask", "mo_Ask", "lo_inspread_Ask",
        "lo_inspread_Bid", "mo_Bid", "co_top_Bid", "lo_top_Bid", "co_deep_Bid", "lo_deep_Bid"]
kp = preprocessdata(pickle.load(open(PARAM_FILE, "rb")))
kp = [[a.copy() for a in kp[0]], kp[1].copy()]
if NULL_KERNELS:
    kp[0][0] = np.zeros_like(kp[0][0])

tod = np.ones((12, 13))
Pis = {'Bid_L2': [0., [(40, 1.)]], 'Bid_inspread': [0., [(40, 1.)]],
       'Bid_L1': [0., [(40, 1.)]], 'Bid_MO': [0., [(40, 1.)]]}
Pis.update({'Ask_MO': Pis['Bid_MO'], 'Ask_L1': Pis['Bid_L1'],
            'Ask_inspread': Pis['Bid_inspread'], 'Ask_L2': Pis['Bid_L2']})
PiQ0 = {k: [0., [(400, 1.)]] for k in ('Ask_L1', 'Ask_L2', 'Bid_L1', 'Bid_L2')}

t0 = time.time()
np.random.seed(SEED)
am = HawkesArrival(spread0=SPREAD, kernelparams=kp, tod=tod, Pis=Pis, beta=0.941,
                   avgSpread=0.0101, Pi_Q0=PiQ0, expApprox=False)
am.spread = SPREAD          # pinned: the exchange never writes back
guard = 0
while (am.s or 0) < T and guard < 5_000_000:
    am.thinningOgataIS2(T=T)
    guard += 1

counts = np.array(am.n, dtype=float)
np.savez(os.path.join(OUT, f"counts_seed{SEED}.npz"),
         counts=counts, cols=np.array(cols),
         null_kernels=np.array([int(NULL_KERNELS)]), T=np.array([T]))
pairs = " ".join(f"{cols[i].replace('_Ask','')}:{counts[i]:.0f}/{counts[11-i]:.0f}" for i in range(6))
print(f"seed={SEED} nullk={int(NULL_KERNELS)} T={T:.0f} spread={SPREAD} "
      f"totAsk={counts[:6].sum():.0f} totBid={counts[6:].sum():.0f} [{time.time()-t0:.0f}s]", flush=True)
print(f"seed={SEED} counts(Ask/Bid) {pairs}", flush=True)
