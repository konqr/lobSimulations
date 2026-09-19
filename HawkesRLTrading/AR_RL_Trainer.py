import sys
import os
import tempfile

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["TORCH_USE_CUDA_DSA"] = "1"  # Device-Side Assertions (even more detail)

sys.path.append(os.path.abspath('/home/ajafree/lobSimulations'))
# sys.path.append(os.path.abspath('/Users/alirazajafree/Documents/GitHub/lobSimulations1'))
from HawkesRLTrading.src.Envs.HawkesRLTradingEnv import *
from HawkesRLTrading.src.SimulationEntities.MetaOrderTradingAgents import TWAPGymTradingAgent
from HawkesRLTrading.src.Utils.action_flow import ActionFlowAudit

import torch
import time

# Run identity is env-overridable so a submission script fully determines its
# own config. Without this the config is whatever ~/lobSimulations happens to be
# checked out to when the job STARTS (not when it is queued), which means only
# one run can safely be queued at a time -- a queued job can dequeue after a
# later pull and silently run the wrong config. Defaults reproduce today's run.
_DEFAULT_RUN = '/home/ajafree/LSTM_fRL/wout_expo/new_value_function/gae_lambda/training/sell'
log_dir = os.environ.get('RUN_LOG_DIR', _DEFAULT_RUN + '/logs/')
model_dir = os.environ.get('RUN_MODEL_DIR', _DEFAULT_RUN + '/model')

start_trading_lag = 100
# reassigned below once TWAP_DURATION is known; kept here for import order
twap_off_time = 400

twap_side = "sell"

# TWAP_SIDE_MODE: "sell" | "buy" | "alternate" | "random".
#   "alternate" flips buy/sell across TWAP-PRESENT episodes only -- counting all
#   episodes would let the absent ones consume side-slots and skew the balance.
# TWAP_ON / TWAP_OFF: the presence cycle. TWAP_ON episodes with the meta-order
#   active, then TWAP_OFF episodes with it neutered (off_time=0, TWAPPresent
#   pinned to 0 throughout), repeating. Absent episodes force the agent to be a
#   viable market maker standalone rather than only a front-runner.
#     1/0 -> present every episode (today's behaviour, the default)
#     1/1 -> alternating
#     2/1 -> two on, one off (keeps time-against-TWAP at 2/3 rather than 1/2)
TWAP_SIDE_MODE = os.environ.get("TWAP_SIDE_MODE", "sell")
TWAP_ON = int(os.environ.get("TWAP_ON", 1))
TWAP_OFF = int(os.environ.get("TWAP_OFF", 0))
assert TWAP_ON >= 1 and TWAP_OFF >= 0, "TWAP_ON must be >=1 and TWAP_OFF >=0"

# SEED_MODE: "fixed" reproduces today's behaviour exactly (tradingEnv's default
#   seed=1 on every episode). "vary" gives each episode its own seed, so the
#   agent cannot overfit one realisation of the background order flow.
# The seed list is logged and saved so any run can be replayed exactly.
SEED_MODE = os.environ.get("SEED_MODE", "fixed")
SEED_BASE = int(os.environ.get("SEED_BASE", 1000))

# Number of training episodes; env-overridable so a short smoke run can
# validate a new config before committing a multi-day job to it.
N_EPISODES = int(os.environ.get("N_EPISODES", 80))
ON_POLICY_PPO = os.environ.get("ON_POLICY_PPO", "false").lower() == "true"
# Explicit frozen evaluation uses this same market/agent construction.
EVAL_ONLY = os.environ.get("EVAL_ONLY", "false").lower() == "true"
EVAL_CHECKPOINT = os.environ.get("EVAL_CHECKPOINT")
EVAL_PRESENCE = os.environ.get("EVAL_PRESENCE", "present")
eval_checkpoint_info = None
if EVAL_ONLY:
    assert EVAL_PRESENCE in ("present", "absent")
    assert not os.environ.get("RESUME_EPOCH"), "Evaluation is not training resume"
    assert int(os.environ.get("START_EPISODE", 0)) == 0
elif EVAL_CHECKPOINT:
    raise ValueError("EVAL_CHECKPOINT requires EVAL_ONLY=true")

# --- Arm parameters. Previously hardcoded, which is how all six of the
# 2026-09-07 runs silently launched on the gae_lambda arm (exploration_bonus=0)
# when the plan called for the combined explo_gae arm. Making them env-overridable
# means the arm is explicit per job and travels with the submission script.
#   EXPLORATION_BONUS: explo_gae arm = 0.1, gae_lambda-only arm = 0.
#   GAE_LAMBDA: 0.95 in every arm run to date.
#   EXP_APPROX: exponential-approximation simulator regime (TAU=10 vs 500).
#     ~13x faster but a DIFFERENT simulator regime -- fast iteration only,
#     never mix with expApprox=False results.
EXPLORATION_BONUS = float(os.environ.get("EXPLORATION_BONUS", 0.1))
GAE_LAMBDA = float(os.environ.get("GAE_LAMBDA", 0.95))
EXP_APPROX = os.environ.get("EXP_APPROX", "false").strip().lower() in ("1", "true", "yes")

# --- Reward-shaping arm. Measured on the 2026-09-07 runs: per-step PnL change
# has median EXACTLY 0 (85-88% of steps move no money), mean |dW| 0.006-0.018,
# and an episode terminal PnL of +-0.2 dollars over ~2,480 steps. Against that,
# ACTION_BONUS=0.5 per acted step is worth hundreds per episode and is
# deterministic and always positive, while PnL is near-zero-mean noise. The
# count-based exploration bonus (0.2 first visit, 0.1/sqrt(N) after) beats the
# mean PnL signal until a state-action pair has been seen ~132 times, which
# most never are. So the objective being maximised is "act often, visit novel
# states"; PnL is a rounding error, which is why no arm has ever produced
# positive standalone PnL.
#   ACTION_BONUS: 0.5 reproduces every run to date. ~1e-3 puts the episode
#     total near 0.25, an order of magnitude under the ~2.5 dollars a
#     successful front-run of a 500-share meta-order is worth at 10 bps.
#   RUNNING_INVPENALTY: per-step lambda*inv**2. 0 reproduces today's behaviour,
#     where the ONLY inventory control is a terminal penalty ~2,480 steps away.
ACTION_BONUS = float(os.environ.get("ACTION_BONUS", 0.5))
RUNNING_INVPENALTY = float(os.environ.get("RUNNING_INVPENALTY", 0.0))
ENTROPY_COEF = float(os.environ.get("ENTROPY_COEF", 0.0))

# --- Action space. 1 (the default, and every run to date) gives the agent FOUR
# actions: lo_top_Ask, co_top_Ask, co_top_Bid, lo_top_Bid -- post or cancel at
# the touch, on either side. There is no market order and no in-spread order in
# that set, so the policy cannot initiate a position at all; it can only choose
# which side to expose and then wait to be filled by someone else's aggression.
# That is a structural obstacle to criterion 1: front-running means taking a
# position AHEAD of the meta-order, but during a buying TWAP the aggression in
# the book is buying, so a resting ask gets lifted and the agent ends up SHORT
# into a rising market -- run over rather than in front. 0 opens the full
# 12-action space including mo_Ask/mo_Bid and the in-spread orders.
ACTION_SPACE_CONFIG = int(os.environ.get("ACTION_SPACE_CONFIG", 1))

# Market-order gating symmetry. Legacy (false, the default and every run to
# date) permits SELLING via market order only on inv in [1, limit-3] but BUYING
# on inv in [2-limit, +inf) -- long-biased. Dormant while ACTION_SPACE_CONFIG=1,
# which never emits a market order, but it must be on for any config-0 run.
SYMMETRIC_MO_GATING = os.environ.get("SYMMETRIC_MO_GATING", "false").strip().lower() in ("1", "true", "yes")

# --- TWAP-alone control. Criterion 3 ("the agent raises the TWAP's transaction
# costs") is a COMPARISON, and its denominator -- what the same meta-order costs
# with no RL agent interfering -- has never been measured on this simulator, on
# these seeds, with this timing. RL_DISABLED runs the identical episode loop
# with the RL agent never acting, so the control differs from the treatment in
# exactly one thing: whether the RL agent submits orders.
# get_action is skipped entirely rather than having its result overridden,
# because it draws from the stdlib RNG that the exchange also uses -- calling it
# and discarding the answer would desynchronise the background order flow and
# silently make the control a different market.
RL_DISABLED = os.environ.get("RL_DISABLED", "false").strip().lower() in ("1", "true", "yes")

# --- CEM self-imitation. `get_CEM_data` pools elite segments from EVERY episode
# in the buffer into one cross-entropy target with no split on side, so
# buy-episode behaviour is imitated on sell episodes and vice versa. That smears
# side-specific policy and shrinks the criterion-1 side contrast toward zero --
# and it fires on 3 of every 4 training sessions. Compounding it, the elite
# selector picks the segments with the highest summed reward, which under the
# default ACTION_BONUS=0.5 means the segments where the agent ACTED MOST, not
# where it traded well. USE_CEM=false turns it off so the contrast can be
# measured without that channel. Default true = today's behaviour.
USE_CEM = os.environ.get("USE_CEM", "true").strip().lower() in ("1", "true", "yes")
# CEM_ELITE_FLOOR: require an elite episode's total reward to clear the buffer
# mean by this many SDs, else take no elites that session. 0 = off = today's
# behaviour, where sorted(...)[:3] returns three episodes however bad they are.
# Matters once the reward is PnL-dominated: episode totals are then near
# zero-mean noise, so the top 3 of ~40 buffered episodes is mostly luck.
CEM_ELITE_FLOOR = float(os.environ.get("CEM_ELITE_FLOOR", 0.0))
# CEM_N_ELITES: total elite episodes per CEM session, held CONSTANT across arms.
# Per-pool counts would make the elite count -- and so the elite fraction of the
# ~40 episodes the buffer holds, i.e. the strength of the intervention -- depend
# on how many TWAP regimes an arm has (3 regimes -> 9, 2 -> 6, 1 -> 5), which
# would confound the very thing under test.
_CEM_N_ELITES_RAW = os.environ.get("CEM_N_ELITES")
# None = legacy: 6 when several regimes are present (top-3 of each of two
# pools, as before), 5 when only one is (the old global top-5). A flat default
# of 6 would silently raise the single-regime arms -- which includes the
# default configuration and buy_base/sell_base -- from 5 to 6.
CEM_N_ELITES = int(_CEM_N_ELITES_RAW) if _CEM_N_ELITES_RAW not in (None, "") else None

# --- Plot cadence. The per-episode inventory-distribution plot
# (graphInventories) was called at the end of every episode until 1f6f7f5
# ("LSTM training, first push"), which dropped the CALL but left the function
# behind as dead code -- the signature of an accidental deletion in a large
# commit rather than a deliberate removal. Restored, with the cadence explicit.
#   DIST_PLOT_EVERY: 1 = every episode, as before 1f6f7f5. 0 disables.
#   TRAJ_PLOT_EVERY: the avg-inventory-trajectory plot, which was tied to the
#     `episode % 4 == 0` checkpoint block and is now independent of it.
DIST_PLOT_EVERY = int(os.environ.get("DIST_PLOT_EVERY", 1))
# Wall-clock cadence, independent of DIST_PLOT_EVERY. Set both to 0 to
# disable all distribution plots. Completed episodes force a final live update.
LIVE_DIST_PLOT_SECONDS = float(os.environ.get("LIVE_DIST_PLOT_SECONDS", 60))
assert np.isfinite(LIVE_DIST_PLOT_SECONDS) and LIVE_DIST_PLOT_SECONDS >= 0
TRAJ_PLOT_EVERY = int(os.environ.get("TRAJ_PLOT_EVERY", 4))

# --- Resume. Models are checkpointed every 4 episodes, but nothing wired that
# to a restart: `checkpoint_params` was hardcoded to None, so a job could not
# ask to resume, and even with weights loaded the loop still ran
# `range(N_EPISODES)` from 0 and redid every episode.
#   RESUME_EPOCH: the checkpoint epoch to load. -1 = the final save.
#   RESUME_TIMESTAMP: which run's checkpoints. Unset = the newest for this
#     label in RUN_MODEL_DIR.
#   START_EPISODE: where the loop restarts. Defaults to RESUME_EPOCH + 1.
# What resume DOES restore: the d and u network weights, and the episode index,
# so seeds (SEED_BASE + episode), the TWAP presence cycle, the buy/sell
# alternation and the training/CEM cadences all continue in phase.
# What it does NOT restore: the trajectory buffer, the exploration visit
# counter, and the in-memory arrays behind the end-of-run .npy files -- those
# will cover post-resume episodes only. Weights are the expensive part; the
# rest is a known and accepted loss.
_RESUME_EPOCH_RAW = os.environ.get("RESUME_EPOCH")
RESUME_EPOCH = int(_RESUME_EPOCH_RAW) if _RESUME_EPOCH_RAW not in (None, "") else None
RESUME_TIMESTAMP = os.environ.get("RESUME_TIMESTAMP") or None
def _latest_checkpoint_epoch(mdir, lbl):
    """Highest epoch checkpointed for this label, or None."""
    import re as _re
    try:
        eps = [int(m.group(1)) for m in
               (_re.match(r"^model_metadata_epoch_(\d+)_\d{8}_\d{6}_"
                          + _re.escape(lbl) + r"\.json$", f)
                for f in os.listdir(mdir)) if m]
    except (IOError, OSError):
        return None
    return max(eps) if eps else None


if RESUME_EPOCH is None:
    START_EPISODE = int(os.environ.get("START_EPISODE", 0))
elif os.environ.get("START_EPISODE") not in (None, ""):
    START_EPISODE = int(os.environ["START_EPISODE"])
elif RESUME_EPOCH >= 0:
    START_EPISODE = RESUME_EPOCH + 1
else:
    # RESUME_EPOCH=-1 means "the newest checkpoint". `-1 + 1 == 0` would have
    # restarted the loop at episode 0 and redone the whole run -- defeating the
    # resume on the spelling a user is most likely to type. The trainer never
    # writes a `_final` save, so resolve the real epoch off disk instead.
    _resolved = _latest_checkpoint_epoch(model_dir, label)
    if _resolved is None:
        raise SystemExit(
            f"RESUME_EPOCH=-1 but no checkpoint found for label {label!r} in "
            f"{model_dir}. Set START_EPISODE explicitly, or check RUN_MODEL_DIR.")
    START_EPISODE = _resolved + 1
    print(f"resume: newest checkpoint for {label!r} is epoch {_resolved}; "
          f"starting at episode {START_EPISODE}", flush=True)

# --- The remaining two shaping terms, for the same reason as ACTION_BONUS.
# Scale reference, all in dollars per EPISODE (2,483 steps, 677 of them inside
# the 150s TWAP window), against the economic prize a successful front-run is
# worth -- 25 shares (the inventory limit) of a 150-share meta-order that moves
# the price ~3 bps, i.e. $0.75:
#
#   action bonus 0.5, 30% act        $372      497x the prize
#   exploration 0.1/sqrt(N), N=100    $25       33x
#   running inv penalty 1e-4 @inv=25  $42       56x
#   terminal inv penalty 25*inv^2   $15625   20833x  (undiscounted)
#   ---- at the prize's own scale ----
#   action bonus 1e-3                  $0.74     1x
#   running inv penalty 2e-7 @inv=25   $0.08     0.1x
#   terminal inv penalty 1.2e-3        $0.75     1x
#
# Note the meta-order is 150 shares; the TWAP's 500 is its starting INVENTORY,
# not its order size. Every shaping term is one to four orders of magnitude
# above the signal the agent is supposed to be learning.
# A running penalty of 1e-4 costs $42 to hold the limit through the window
# against a $0.75 prize, so it forbids precisely the inventory-holding that
# criterion 1 requires -- do not use it as an "inventory control" default.
TERMINAL_INVPENALTY = float(os.environ.get("TERMINAL_INVPENALTY", 5 * 5))
FIRST_VISIT_BONUS = float(os.environ.get("FIRST_VISIT_BONUS", 0.2))

# --- the meta-order being modelled ---------------------------------------
# These were hardcoded. They are the shape of the market this project studies,
# so changing them is a modelling decision, not a training knob -- but they
# have to be reachable to be studied at all. Every default below reproduces
# the historical values exactly, so an arm that sets none of them is unchanged.
#
# Why they matter for in-window front-running. The agent cannot anticipate the
# meta-order before it starts (TWAPPresent is 0 until twap_start_time), so the
# only front-running available is against the order that REMAINS after t=250.
# Two quantities bound the prize:
#   price move  ~ sqrt(TWAP_ORDER_SIZE)      (square-root law, see the paper)
#   shares held <= RL_INVENTORY_LIMIT        (a hard cap, linear)
# so the PRIZE scales as sqrt(Q) * limit.
#
# But prize is not detectability. Holding q shares through a move m earns q*m,
# and the episode-PnL noise of a market maker also scales roughly with q, so
# SNR ~ m and is largely INDEPENDENT of the limit. Raising RL_INVENTORY_LIMIT
# raises the stakes on both sides of the ledger; it does not make the edge
# easier to find. TWAP_ORDER_SIZE (and the participation rate Q/T it implies)
# is the lever that moves m, and therefore the lever that moves SNR.
# The limit's real job is to be large enough relative to Q that the agent can
# monetise a meaningful fraction of the move at all.
TWAP_ORDER_SIZE = float(os.environ.get("TWAP_ORDER_SIZE", 150))
TWAP_DURATION = float(os.environ.get("TWAP_DURATION", 150))
TWAP_WINDOW_SIZE = float(os.environ.get("TWAP_WINDOW_SIZE", 25))
TWAP_ACTION_FREQ = float(os.environ.get("TWAP_ACTION_FREQ", 1))
RL_INVENTORY_LIMIT = int(os.environ.get("RL_INVENTORY_LIMIT", 25))
STOP_TIME = float(os.environ.get("STOP_TIME", 550))

#the time that the TWAP agent will kick in:
twap_start_time = 150 + start_trading_lag

twap_end_time = twap_start_time + TWAP_DURATION

# The episode must outlast the meta-order, or there is no post-window stretch
# and `inventory_post_twap` is empty.
assert STOP_TIME > twap_end_time, (
    "STOP_TIME=%s does not outlast the meta-order, which ends at %s"
    % (STOP_TIME, twap_end_time))

twap_off_time = twap_end_time

label = os.environ.get('RUN_LABEL', 'train_new_vf_gae_lambda_sell')
layer_widths=100
n_layers=3
eta = 5

checkpoint_params = None if RESUME_EPOCH is None else (RESUME_TIMESTAMP, RESUME_EPOCH)

def graphInventories(beforetwap, withtwap_buy, withtwap_sell, episode_num,
                     live=False, simulation_time=None, single_episode=False):
    """Cumulative observations; episode N is not an evaluation of checkpoint N.

    The trainer saves checkpoint N after that episode's training update.
    Live snapshots include the current partial episode without mutating history.
    """
    fig = plt.figure(figsize=(12, 8))
    
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
        plt.hist(all_before, bins=30, alpha=0.7, label=f'Outside TWAP / absent (n={len(all_before)})',
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
                   label=f'Outside / absent median: {np.median(all_before):.1f}')
    if all_buy:
        plt.axvline(np.median(all_buy), color='green', linestyle='--', linewidth=2,
                   label=f'Buy Median: {np.median(all_buy):.1f}')
    if all_sell:
        plt.axvline(np.median(all_sell), color='red', linestyle='--', linewidth=2,
                   label=f'Sell Median: {np.median(all_sell):.1f}')
    
    plt.xlabel('RL Agent Inventory')
    plt.ylabel('Proportion') 
    progress = f'Episode {episode_num} only' if single_episode else f'Through episode {episode_num}'
    if simulation_time is not None:
        progress += f' (partial, simulation t={simulation_time:.1f}s)'
    plt.title('RL Agent Inventory Distributions — ' + progress)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = (os.path.join(log_dir, label + '_all_inventory_distributions_live.png')
            if live else log_dir + label + f'_all_inventory_distributions_episode_{episode_num}.png')
    if single_episode:
        path = os.path.join(log_dir, label + ('_inventory_current_live.png' if live
                              else f'_inventory_single_episode_{episode_num}.png'))
    # Readers copying the live image must never see a half-written PNG.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=log_dir, suffix='.png', delete=False) as fh:
            temporary = fh.name
        fig.savefig(temporary, dpi=150 if live else 300, bbox_inches='tight')
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)
        plt.close(fig)

def plot_avg_inventory_trajectories(buy_trajectories, sell_trajectories, episode_num, save_dir, label_prefix, twap_start, twap_end):
    """Plot avg inventory trajectories colored by training batch (every 4 episodes), separated by buy/sell."""
    import matplotlib.colors
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)

    cmap = plt.cm.viridis
    max_batch = max(episode_num // 4, 1)

    for ax, trajectories, side_label in zip(axes, [buy_trajectories, sell_trajectories], ['Buy TWAP', 'Sell TWAP']):
        if not trajectories:
            ax.set_title(f'{side_label} - No episodes')
            ax.grid(True, alpha=0.3)
            continue

        ax.axvspan(twap_start, twap_end, alpha=0.1, color='lightcoral', label='TWAP active')

        all_interp = []
        max_time = max(times[-1] for _, times, _ in trajectories if len(times) > 0)
        common_times = np.linspace(0, max_time, 500)

        for ep_num, times, invs in trajectories:
            batch_idx = ep_num // 4
            color = cmap(batch_idx / max_batch)
            ax.plot(times, invs, alpha=0.5, color=color, linewidth=1.5)
            all_interp.append(np.interp(common_times, times, invs))

        if all_interp:
            avg = np.mean(all_interp, axis=0)
            ax.plot(common_times, avg, color='black', linewidth=2.5, label=f'Avg (n={len(trajectories)})')

        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Inventory')
        ax.set_title(f'{side_label} Episodes (n={len(trajectories)})')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

    norm = matplotlib.colors.Normalize(vmin=0, vmax=max_batch)
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes, label='Training batch (every 4 episodes)', shrink=0.8)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{label_prefix}_avg_inv_trajectories_ep{episode_num}.png'), dpi=150)
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
                         "rewardpenalty": eta,
                         "Inventory": {"INTC": 0},
                         "log_to_file": True,
                         "cashlimit": 5000000,
                         "inventorylimit": RL_INVENTORY_LIMIT,
                         'start_trading_lag': start_trading_lag,
                         "wake_on_MO": True,
                         "wake_on_Spread": True}
                         ,
                         {"cash":1000000,
                          "cashlimit": 100000000000,
                          "strategy": "TWAP",
                          "on_trade":False,
                          "total_order_size":TWAP_ORDER_SIZE,
                          "order_target":"INTC",
                          "total_time":TWAP_DURATION,
                          "window_size":TWAP_WINDOW_SIZE, #window size, measured in seconds
                          "action_freq":TWAP_ACTION_FREQ,
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
                                     'expApprox' : EXP_APPROX}}
}

agents = kwargs['GymTradingAgent']
j = agents[0]
tc = 0.0001
RLagentInstance = AdversarialPPOAgent( seed=1, log_events=True, log_to_file=True, strategy=j["strategy"], Inventory=j["Inventory"], cash=j["cash"], action_freq=j["action_freq"],
                          wake_on_MO=j["wake_on_MO"], wake_on_Spread=j["wake_on_Spread"], cashlimit=j["cashlimit"],inventorylimit=j['inventorylimit'], batch_size=512,
                          layer_widths=layer_widths, n_layers =n_layers, buffer_capacity = 100000, rewardpenalty = j["rewardpenalty"], epochs = 100, transaction_cost=1e-4, start_trading_lag = j['start_trading_lag'],
                          gae_lambda=GAE_LAMBDA, gamma=0.999, truncation_enabled=False, action_space_config = ACTION_SPACE_CONFIG, alt_state=True, enhance_state=True, include_time=True, optim_type='ADAM',entropy_coef=ENTROPY_COEF, exploration_bonus = EXPLORATION_BONUS, hidden_activation='sigmoid',
                          typeNN = "LSTM", lr = 3e-4, chunk_length=64, TWAPPresent=0, cem_full_episode=True, terminal_invpenalty=TERMINAL_INVPENALTY, first_visit_bonus=FIRST_VISIT_BONUS, two_sided_reward=False,
                          action_bonus=ACTION_BONUS, running_invpenalty=RUNNING_INVPENALTY,
                          symmetric_mo_gating=SYMMETRIC_MO_GATING, cem_elite_floor=CEM_ELITE_FLOOR, cem_n_elites=CEM_N_ELITES)

RLagentInstance.on_policy = ON_POLICY_PPO
RLagentInstance.on_policy_epochs = int(os.environ.get('ON_POLICY_EPOCHS', 4))
RLagentInstance.on_policy_target_kl = float(os.environ.get('ON_POLICY_TARGET_KL', .02))
if ON_POLICY_PPO:
    assert not USE_CEM, 'On-policy arm explicitly disables CEM imitation'
    assert ACTION_SPACE_CONFIG in (0, 1), 'On-policy pilot uses scalar actions'

# Config banner. The six 2026-09-07 runs could not be told apart from their .o
# files because nothing recorded which arm they were on; this makes every job
# self-documenting.
print("=" * 72, flush=True)
print("RUN CONFIG", flush=True)
print(f"  ON_POLICY_PPO     = {ON_POLICY_PPO}; epochs={RLagentInstance.on_policy_epochs}; target_kl={RLagentInstance.on_policy_target_kl}", flush=True)
print(f"  MODEL_SEED        = {os.environ.get('MODEL_SEED')}", flush=True)
print(f"  label              = {label}", flush=True)
print(f"  log_dir            = {log_dir}", flush=True)
print(f"  model_dir          = {model_dir}", flush=True)
print(f"  N_EPISODES         = {N_EPISODES}", flush=True)
print(f"  TWAP_SIDE_MODE     = {TWAP_SIDE_MODE}", flush=True)
print(f"  TWAP_ON/TWAP_OFF   = {TWAP_ON}/{TWAP_OFF}", flush=True)
print(f"  SEED_MODE/BASE     = {SEED_MODE}/{SEED_BASE}", flush=True)
print(f"  exploration_bonus  = {EXPLORATION_BONUS}", flush=True)
print(f"  gae_lambda         = {GAE_LAMBDA}", flush=True)
print(f"  expApprox          = {EXP_APPROX}", flush=True)
print(f"  action_bonus       = {ACTION_BONUS}", flush=True)
print(f"  running_invpenalty = {RUNNING_INVPENALTY}", flush=True)
print(f"  entropy_coef       = {ENTROPY_COEF}", flush=True)
print(f"  action_space_config= {ACTION_SPACE_CONFIG}", flush=True)
print(f"  symmetric_mo_gating= {SYMMETRIC_MO_GATING}", flush=True)
print(f"  RL_DISABLED        = {RL_DISABLED}", flush=True)
print(f"  terminal_invpenalty= {TERMINAL_INVPENALTY}", flush=True)
print(f"  first_visit_bonus  = {FIRST_VISIT_BONUS}", flush=True)
# rewardpenalty/eta are INERT in the live objective: the running quadratic
# inventory penalty is commented out (ICRLAgent.py), and the only two other
# sites are guarded by `not alt_state` (we pass alt_state=True) and
# `two_sided_reward` (we pass False). eta reaches the reward ONLY through
# terminal_invpenalty. Printed with that label so nobody reads it as a knob.
print(f"  eta (INERT except via terminal_invpenalty) = {eta}", flush=True)
print(f"  rewardpenalty (INERT) = {j['rewardpenalty']}", flush=True)
print(f"  use_CEM            = {USE_CEM}", flush=True)
print(f"  cem_elite_floor    = {CEM_ELITE_FLOOR}", flush=True)
print(f"  cem_n_elites       = {CEM_N_ELITES}", flush=True)
print(f"  live_dist_seconds  = {LIVE_DIST_PLOT_SECONDS}", flush=True)
print(f"  dist/traj_plot_every = {DIST_PLOT_EVERY}/{TRAJ_PLOT_EVERY}", flush=True)
print(f"  START_EPISODE      = {START_EPISODE}", flush=True)
print(f"  resume             = {checkpoint_params}", flush=True)
print(f"  inventorylimit     = {j['inventorylimit']}", flush=True)
print(f"  meta-order         = {TWAP_ORDER_SIZE:g} shares over {TWAP_DURATION:g}s "
      f"[{twap_start_time:g},{twap_end_time:g}], window {TWAP_WINDOW_SIZE:g}s, "
      f"freq {TWAP_ACTION_FREQ:g}; episode ends {STOP_TIME:g}", flush=True)
print("=" * 72, flush=True)

inventories_with_twap_buy = []
inventories_with_twap_sell = []
inventories_without_twap = []

j['agent_instance'] = RLagentInstance
kwargs['GymTradingAgent'] = agents
i_eps=0
t, t_with_twap_buy, t_with_twap_sell, t_without_twap = [], [], [], []
avgEpisodicRewards, stdEpisodicRewards, finalcash, finalcash2, profit_with_twap_sell, profit_with_twap_buy, profit_without_twap = [], [], [], [], [], [], []
train_logger = TrainingLogger(layer_widths=layer_widths, n_layers=n_layers, log_dir=log_dir, label = label)
model_manager = ModelManager(model_dir = model_dir, label = label)
counter_profit = 0
episode_boundaries = [0]
cashs:Dict[int, List] = {}
inventories:Dict[int, List] = {}
actionss:Dict[int, List] = {}
RLagentID = 1

start_midprices = []
total_executeds = []

TWAP_obsv = []
RL_obsv = []

sides = []
twap_presents = []
episode_seeds = []
# Count the TWAP-present episodes we are skipping, so `alternate` resumes on
# the correct side. Left at 0, a resume would restart the cycle on buy.
n_twap_present = sum(1 for _e in range(START_EPISODE)
                     if (_e % (TWAP_ON + TWAP_OFF)) < TWAP_ON)

eps_with_buy = []
eps_with_sell = []

total_RL_obsv = []

final_cashs = []
total_executeds = []
episode_inv_trajectories_buy = []
episode_inv_trajectories_sell = []
# Per-episode scalar metrics, rewritten every episode. The arrays that carry the
# primary metric (inventory level inside the TWAP window) are only written when
# the run FINISHES, which is why every in-flight run to date could only be
# judged by eyeballing the avg_inv_trajectories PNGs. This file makes the three
# criteria readable numerically while a job is still running, and keeps
# TWAP-absent episodes separated from the out-of-window parts of present
# episodes -- `profit_without_twap` mixes the two and is not a clean readout.
episode_metrics = []
if START_EPISODE > 0:
    # Keep the episodes recorded before the interruption, or the live metrics
    # file is truncated to the post-resume tail and the run looks like it
    # started at START_EPISODE.
    try:
        import json as _json
        with open(log_dir + "episode_metrics_" + label + ".json") as _fh:
            episode_metrics = [e for e in _json.load(_fh).get("episodes", [])
                               if e.get("episode", 0) < START_EPISODE]
        print(f"resume: kept {len(episode_metrics)} episode metrics from before "
              f"episode {START_EPISODE}", flush=True)
    except (IOError, OSError, ValueError) as _e:
        print(f"resume: no prior episode metrics to keep ({_e})", flush=True)


def _save_episode_metrics():
    import json
    with open(log_dir + "episode_metrics_" + label + ".json", "w") as fh:
        json.dump({"label": label,
                   "model_seed": os.environ.get("MODEL_SEED"),
                   "on_policy_ppo": ON_POLICY_PPO,
                   "eval_only": EVAL_ONLY,
                   "eval_presence": EVAL_PRESENCE if EVAL_ONLY else None,
                   "eval_checkpoint": eval_checkpoint_info,
                   "twap_order_size": TWAP_ORDER_SIZE,
                   "twap_duration": TWAP_DURATION,
                   "twap_window_size": TWAP_WINDOW_SIZE,
                   "twap_action_freq": TWAP_ACTION_FREQ,
                   "stop_time": STOP_TIME,
                   "exploration_bonus": EXPLORATION_BONUS,
                   "gae_lambda": GAE_LAMBDA,
                   "expApprox": EXP_APPROX,
                   "action_bonus": ACTION_BONUS,
                   "running_invpenalty": RUNNING_INVPENALTY,
                   "entropy_coef": ENTROPY_COEF,
                   "action_space_config": ACTION_SPACE_CONFIG,
                   "symmetric_mo_gating": SYMMETRIC_MO_GATING,
                   "rl_disabled": RL_DISABLED,
                   "use_cem": USE_CEM,
                   "cem_elite_floor": CEM_ELITE_FLOOR,
                   "cem_n_elites": CEM_N_ELITES,
                   "terminal_invpenalty": TERMINAL_INVPENALTY,
                   "first_visit_bonus": FIRST_VISIT_BONUS,
                   "twap_on": TWAP_ON, "twap_off": TWAP_OFF,
                   "twap_side_mode": TWAP_SIDE_MODE,
                   "seed_mode": SEED_MODE, "seed_base": SEED_BASE,
                   "n_episodes_planned": N_EPISODES,
                   "inventorylimit": j["inventorylimit"],
                   "rewardpenalty": j["rewardpenalty"],
                   "episodes": episode_metrics}, fh, indent=1)


def _twap_slippage_bps(side, twap_final_cash, executed, start_mid):
    """TWAP execution cost in bps vs the arrival midprice. Positive = the
    meta-order paid more (buy) or received less (sell) than the arrival mid,
    i.e. higher transaction cost. Criterion 3 wants this to go UP versus the
    TWAP-alone baseline. Same convention as AR_RL_runner.py:605-613."""
    # np.nan is the sentinel total_executed carries on TWAP-absent episodes, and
    # `not nan` is False -- guard on non-finiteness explicitly or nan propagates
    # into every downstream mean.
    if executed is None or start_mid is None:
        return None
    if not (np.isfinite(executed) and np.isfinite(start_mid)):
        return None
    benchmark = start_mid * executed
    if benchmark <= 0:
        return None
    if side == "sell":
        return (benchmark - (twap_final_cash - 1000000)) * 10000 / benchmark
    return ((1000000 - twap_final_cash) - benchmark) * 10000 / benchmark


last_live_dist_plot = float("-inf")
for episode in range(START_EPISODE, N_EPISODES):
    inventory_with_twap_buy = []
    inventory_with_twap_sell = []
    inventory_without_twap = []
    # The CLOCK window (twap_start, twap_end), recorded on EVERY episode --
    # present or absent. `inventory_with_twap_*` is keyed on the meta-order
    # actually being there, so it is empty on an absent episode and the
    # (250,400) stretch has no window-level record at all. That makes the
    # present-vs-absent contrast -- the only front-running measure available to
    # a SINGLE-SIDE run -- impossible to compute. This list is the
    # counterfactual arm of that contrast. On a present episode it holds
    # exactly the same samples as `inventory_with_twap_*` (same bounds, same
    # strict inequalities), which is asserted below.
    inventory_window_clock = []
    inventory_pre_twap = []
    inventory_post_twap = []
    episode_times_rl = []
    episode_invs_rl = []
    action_flow = ActionFlowAudit()

    RL_agent_obsv = []
    TWAP_agent_obsv = []

    twap_diff = 0
    starting_midprice = 0
    new_midprice = True
    kwargs["GymTradingAgent"][1]["Inventory"] = {"INTC": 500}
    kwargs["GymTradingAgent"][1]["cash"] = 1000000

    # Is the TWAP meta-order present at all this episode?
    twap_present = (EVAL_PRESENCE == "present") if EVAL_ONLY else (episode % (TWAP_ON + TWAP_OFF)) < TWAP_ON
    if TWAP_SIDE_MODE == "alternate":
        # Indexed by how many TWAP-present episodes have already run, so the
        # buy/sell split stays exact regardless of the presence cycle.
        twap_side = "buy" if (n_twap_present % 2 == 0) else "sell"
    elif TWAP_SIDE_MODE == "random":
        twap_side = str(np.random.choice(["buy", "sell"]))
    else:
        twap_side = TWAP_SIDE_MODE
    if twap_present:
        n_twap_present += 1
    twap_presents.append(twap_present)
    sides.append(twap_side if twap_present else "none")
    if twap_present:
        eps_with_buy.append(episode) if twap_side == "buy" else eps_with_sell.append(episode)
    # off_time=0 neuters the TWAP: it still wakes on its 1s clock (so the
    # sampling cadence and starting_midprice capture are unchanged) but returns
    # (12,0) every time and never trades.
    kwargs["GymTradingAgent"][1]["off_time"] = twap_off_time if twap_present else 0
    kwargs["GymTradingAgent"][1]["start_trading_lag"] = twap_start_time
    #randomise buy or sell
    kwargs["GymTradingAgent"][1]["side"] = twap_side
    i = 0
    action_num = 0
    episode_seed = (SEED_BASE + episode) if SEED_MODE == "vary" else 1
    episode_seeds.append(episode_seed)
    env=tradingEnv(stop_time=STOP_TIME, wall_time_limit=23400, seed=episode_seed, **kwargs)
    if EVAL_ONLY:
        # Reproducible policy sampling per held-out seed, independent of task order.
        torch.manual_seed(episode_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(episode_seed)
    print(f"Start of episode {episode}. TWAP present: {twap_present}, side: "
          f"{twap_side if twap_present else 'none'}, seed: {episode_seed}")
    print("Initial Observations"+ str(env.getobservations()))
    Simstate, observations, termination, truncation =env.step(action=None) 
    AgentsIDs=[k for k,v in Simstate["Infos"].items() if v==True]
    agents:List[GymTradingAgent] = [env.getAgent(ID=agentid) for agentid in AgentsIDs]
    observationsDict:Dict[int, Dict] = {agentid: {"Inventory": agent.Inventory, "Positions": []} for agent, agentid in zip(agents, AgentsIDs)}
    if episode == START_EPISODE:
        if not EVAL_ONLY and os.environ.get('MODEL_SEED'):
            torch.manual_seed(int(os.environ['MODEL_SEED']))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(os.environ['MODEL_SEED']))
        for agent in agents:
            if isinstance(agent, PPOAgent):
                agent.setupNNs(observations)
        if EVAL_ONLY and not RL_DISABLED:
            from HawkesRLTrading.src.Utils.frozen_eval import load_frozen_checkpoint
            eval_checkpoint_info = load_frozen_checkpoint(RLagentInstance, EVAL_CHECKPOINT)
        if checkpoint_params is not None:
            loaded_models = model_manager.load_models(timestamp=checkpoint_params[0], epoch = checkpoint_params[1], d = agent.Actor_Critic_d, u = agent.Actor_Critic_u)
            # Fail loudly. A missing checkpoint used to surface as
            # `TypeError: list indices must be integers`, and assigning None
            # here would leave the agent with no networks at all.
            if not isinstance(loaded_models, dict) or loaded_models.get('d') is None \
                    or loaded_models.get('u') is None:
                raise SystemExit(
                    f"resume: could not load checkpoint {checkpoint_params} for "
                    f"label {label!r} from {model_dir}; got {loaded_models!r}")
            agent.Actor_Critic_d = loaded_models['d']
            agent.Actor_Critic_u = loaded_models['u']
    logger.debug(f"\nSimstate: {Simstate}\nObservations: {observations}\nTermination: {termination}")
    episode_start_time = time.time()
    while Simstate["Done"]==False and termination!=True:
        counter_profit +=1
        logger.debug(f"ENV TERMINATION: {termination}")
        AgentsIDs=[k for k,v in Simstate["Infos"].items() if v==True]
        print(f"Agents with IDs {AgentsIDs} have an action available")
        agents:List[GymTradingAgent] = [env.getAgent(ID=agentid) for agentid in AgentsIDs]
        # action:list[Tuple] = []
        if twap_present and (twap_end_time >= Simstate['TimeCode'] >= twap_start_time):
            if not RLagentInstance.TWAPPresent:
                RLagentInstance.TWAPPresent = -1 if twap_side == 'sell' else 1
        else:
            RLagentInstance.TWAPPresent = 0

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

                # On a TWAP-absent episode the meta-order never trades, so
                # total_executed is 0 and any slippage derived from it would
                # divide by zero. Record NaN so downstream analysis skips the
                # episode instead of silently producing inf/garbage.
                total_executed = abs(500 - agent.Inventory["INTC"]) if twap_present else np.nan
                final_cash = agent.cash if twap_present else np.nan
                   
            else:
                print(f"Twap present: {RLagentInstance.TWAPPresent}")
                action_num+=1
                RLagentID = agent.id
                action_time_rl = Simstate["TimeCode"]
                inventory_before_rl = agent.countInventory()
                if RL_DISABLED:
                    # Control arm: no policy call at all (see RL_DISABLED above).
                    agentAction = (12, (0, 0), 0, 0, 0, 0)
                    state_at_action = None
                    action = (agent.id, (12, 1))
                else:
                    agentAction:Tuple[int, int] = agent.get_action(data=env.getobservations(agentID=agent.id), epsilon = 0.5 if i_eps < 100 else 0.1, evaluation=EVAL_ONLY)
                    # Snapshot the state the action was actually chosen from (set inside get_action).
                    # Using this instead of prev_readData ensures the stored (s, a) pair is aligned —
                    # critical when other agents (TWAP) act between RL steps and shift env state.
                    state_at_action = agent.last_state.clone() if agent.last_state is not None else None
                    action = (agent.id, (agentAction[0],1))
                Simstate, observations, termination, truncation=env.step(action=action) #do not try and use this data before this line in the loop
                action_flow.record(
                    agentAction, action_time=action_time_rl, twap_present=twap_present,
                    twap_start=twap_start_time, twap_end=twap_end_time,
                    inventory_before=inventory_before_rl,
                    inventory_after=observations["Inventory"], disabled=RL_DISABLED)
                episode_times_rl.append(Simstate['TimeCode'])
                episode_invs_rl.append(observations["Inventory"])
                # Key on ACTUAL TWAP presence, not just the clock window: on a
                # TWAP-absent episode the [twap_start, twap_end] window is
                # ordinary market-making time and must not be filed as
                # "with TWAP", which would contaminate the very comparison this
                # design exists to make.
                if twap_end_time > Simstate['TimeCode'] > twap_start_time:
                    # Clock window only -- no twap_present condition. See the
                    # declaration above for why the counterfactual is needed.
                    inventory_window_clock.append(observations["Inventory"])
                if twap_present and (twap_end_time > Simstate['TimeCode'] > twap_start_time):
                    if twap_side == "sell":
                        inventory_with_twap_sell.append(observations["Inventory"])
                    else:
                        inventory_with_twap_buy.append(observations["Inventory"])
                else:
                    inventory_without_twap.append(observations["Inventory"])
                # Split the out-of-window samples by WHICH side of the window
                # they fall on. `inventory_without_twap` pools (100,250] with
                # [400,550], and the pre-TWAP stretch is exactly where
                # front-running happens -- a policy that builds its position
                # before t=250 and holds it through the window scores a paired
                # shift of ~0 against that pooled baseline, by construction.
                # The correct baseline for "did the agent position itself AHEAD
                # of the meta-order" is the PRE window alone.
                if Simstate['TimeCode'] <= twap_start_time:
                    inventory_pre_twap.append(observations["Inventory"])
                elif Simstate['TimeCode'] >= twap_end_time:
                    inventory_post_twap.append(observations["Inventory"])
                if LIVE_DIST_PLOT_SECONDS and (
                        time.monotonic() - last_live_dist_plot >= LIVE_DIST_PLOT_SECONDS):
                    graphInventories(
                        beforetwap=inventories_without_twap + [inventory_without_twap],
                        withtwap_buy=inventories_with_twap_buy + [inventory_with_twap_buy],
                        withtwap_sell=inventories_with_twap_sell + [inventory_with_twap_sell],
                        episode_num=episode, live=True, simulation_time=Simstate['TimeCode'])
                    graphInventories(beforetwap=[inventory_without_twap],
                                     withtwap_buy=[inventory_with_twap_buy],
                                     withtwap_sell=[inventory_with_twap_sell],
                                     episode_num=episode, live=True,
                                     simulation_time=Simstate['TimeCode'], single_episode=True)
                    last_live_dist_plot = time.monotonic()
                observationsDict.update({agent.id:observations})
                logger.debug(f"\n Agent: {agent.id}\n Simstate: {Simstate}\nObservations: {observations}\nTermination: {termination}\nTruncation: {truncation}")
                if len(t) > 0 and Simstate['TimeCode'] < t[-1]:
                    # Episode has reset - mark boundary BEFORE appending the new episode's first entry
                    episode_boundaries.append(len(cashs.get(agent.id, [])))
                cashs.update({agent.id:cashs.get(agent.id, [])+[observations['Cash']]})
                inventories.update({agent.id:inventories.get(agent.id, []) + [observations['Inventory']]})
                actionss.update({agent.id: actionss.get(agent.id, []) + [action[1][0]]})
                t += [Simstate['TimeCode']]
                current_readData = agent.readData(observations)
                if state_at_action is not None:
                    agent.store_transition(episode, state_at_action, agentAction[1], agent.calculaterewards(termination), current_readData, (termination or truncation), rollout=agentAction)
                print(f'Current reward: {agent.calculaterewards(termination):0.4f}')
                # print(f'Prev avg reward: {np.mean([r[2] for r in agent.experience_replay[-100:]]):0.4f}')
                i_eps+=1
                logger.debug(f"\nSimstate: {Simstate}\nObservations: {observations}\nTermination: {termination}\nTruncation: {truncation}")
                # Calculate current PnL (cash + inventory value)
                current_pnl = cashs[agent.id][-1] + inventories[agent.id][-1] * agent.mid * (1 - tc*np.sign(inventories[agent.id][-1]))
                finalcash2.append(current_pnl)

                # Same presence-vs-clock fix as the inventory buckets above.
                if twap_present and (twap_end_time >= Simstate['TimeCode'] >= twap_start_time):
                    if twap_side == "buy":
                        profit_with_twap_buy.append(current_pnl)
                        t_with_twap_buy += [Simstate['TimeCode']]
                    else:
                        profit_with_twap_sell.append(current_pnl)
                        t_with_twap_sell += [Simstate['TimeCode']]
                else:
                    profit_without_twap.append(current_pnl)
                    t_without_twap += [Simstate['TimeCode']]
                
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

                    plt.close()

                np.save(log_dir + "sharpe_" + label + '_profit', np.array([t, finalcash2]))
                if len(t_with_twap_buy) > 0:
                    np.save(log_dir + "sharpe_" + label + "_profit_w_twap_buy", np.array([t_with_twap_buy, profit_with_twap_buy]))
                if len(t_with_twap_sell) > 0:
                    np.save(log_dir + "sharpe_" + label + "_profit_w_twap_sell", np.array([t_with_twap_sell, profit_with_twap_sell]))
                np.save(log_dir + "sharpe_" + label + "_profit_wout_twap", np.array([t_without_twap, profit_without_twap]))

                RL_agent_obsv.append(observationsDict[RLagentID])
            
            print(agent.current_time)
            print(f"ACTION DONE{action_num}")
    
    if ON_POLICY_PPO and not EVAL_ONLY and not RL_DISABLED:
        from HawkesRLTrading.src.Utils.on_policy import finalize_episode
        tail_reward = finalize_episode(RLagentInstance, episode, termination)
        print('TERMINAL REWARD TAIL', tail_reward, flush=True)
    total_RL_obsv.append(RL_agent_obsv)

    if len(inventory_with_twap_buy) > 0:
        inventories_with_twap_buy.append(inventory_with_twap_buy)
        if len(episode_times_rl) > 0:
            episode_inv_trajectories_buy.append((episode, episode_times_rl, episode_invs_rl))
    if len(inventory_with_twap_sell) > 0:
        inventories_with_twap_sell.append(inventory_with_twap_sell)
        if len(episode_times_rl) > 0:
            episode_inv_trajectories_sell.append((episode, episode_times_rl, episode_invs_rl))
    inventories_without_twap.append(inventory_without_twap)
    if DIST_PLOT_EVERY and (episode % DIST_PLOT_EVERY == 0):
        # Restored from before 1f6f7f5. Note this plots the CUMULATIVE
        # distribution over every episode so far, not just this one -- the
        # episode number only names the file, giving a snapshot per episode.
        graphInventories(withtwap_buy=inventories_with_twap_buy,
                         withtwap_sell=inventories_with_twap_sell,
                         beforetwap=inventories_without_twap,
                         episode_num=episode)
        graphInventories(beforetwap=[inventory_without_twap],
                         withtwap_buy=[inventory_with_twap_buy],
                         withtwap_sell=[inventory_with_twap_sell],
                         episode_num=episode, single_episode=True)

    if LIVE_DIST_PLOT_SECONDS:
        graphInventories(beforetwap=inventories_without_twap,
                         withtwap_buy=inventories_with_twap_buy,
                         withtwap_sell=inventories_with_twap_sell,
                         episode_num=episode, live=True)
        last_live_dist_plot = time.monotonic()

    final_cashs.append(final_cash)
    total_executeds.append(total_executed)
    start_midprices.append(starting_midprice)

    # Terminal PnL straight off this episode's last sample, so no episode
    # boundary has to be reconstructed from non-monotonic time downstream.
    _term_pnl = (finalcash2[-1] - j["cash"]) if len(finalcash2) else None
    _inw = inventory_with_twap_sell if twap_side == "sell" else inventory_with_twap_buy
    # On a PRESENT episode the clock window and the with-TWAP window are the
    # same samples by construction (identical bounds, identical strict
    # inequalities). Assert it, so the counterfactual arm cannot silently drift
    # away from the arm it is compared against.
    if twap_present:
        assert inventory_window_clock == _inw, (
            "clock window (%d samples) and with-TWAP window (%d) diverged on a "
            "present episode -- the present-vs-absent contrast would compare "
            "two different windows" % (len(inventory_window_clock), len(_inw)))
    episode_metrics.append({
        "episode": episode,
        "twap_present": bool(twap_present),
        "side": twap_side if twap_present else "none",
        "seed": episode_seed,
        # PRIMARY metric for criterion 1: mean inventory LEVEL inside the TWAP
        # window. Want > 0 for a buying TWAP, < 0 for a selling TWAP. Level, not
        # the paired shift -- they disagree and level is the one that asks
        # whether the agent actually HOLDS the profitable position.
        "inv_level_in_window": float(np.mean(_inw)) if (twap_present and len(_inw)) else None,
        "n_in_window": int(len(_inw)) if twap_present else 0,
        "inv_level_out_window": float(np.mean(inventory_without_twap)) if len(inventory_without_twap) else None,
        "n_out_window": int(len(inventory_without_twap)),
        # PRE window (100, 250] -- the correct baseline for criterion 1. The
        # front-running response is inv_level_in_window - inv_level_pre_window.
        # Present-vs-absent contrast: mean(window_clock | present)
        # - mean(window_clock | absent). A per-run directional inventory bias
        # is in BOTH arms, so it differences out -- which is what lets a
        # single-side run be scored for front-running at all.
        "inv_level_window_clock": (float(np.mean(inventory_window_clock))
                                   if len(inventory_window_clock) else None),
        "n_window_clock": int(len(inventory_window_clock)),
        "inv_level_pre_window": float(np.mean(inventory_pre_twap)) if len(inventory_pre_twap) else None,
        "n_pre_window": int(len(inventory_pre_twap)),
        "inv_level_post_window": float(np.mean(inventory_post_twap)) if len(inventory_post_twap) else None,
        "n_post_window": int(len(inventory_post_twap)),
        "inv_terminal": float(inventory_without_twap[-1]) if len(inventory_without_twap) else None,
        # Fraction of steps sitting AT the inventory limit. Those are forced
        # liquidations: get_action returns (mo, (None, None)) before setting
        # last_state, so store_transition early-returns on `d is None` and the
        # transition never reaches the buffer -- while calculaterewards has
        # already run as its argument and advanced statelog. The liquidation's
        # cost is therefore deleted from the learning signal rather than merely
        # delayed. Measured on buy_base this grows 0.015 -> 0.089 over training
        # while its inventory level diverges, so it is worth tracking live.
        "frac_at_inventory_limit": (
            # `inventory_without_twap` holds pre+post on a PRESENT episode and
            # the WHOLE episode on an absent one, so `without + in_window` is
            # the complete episode either way. Using pre+in+post instead drops
            # the (250,400) stretch of every absent episode -- a third of it --
            # because nothing populates _inw when the meta-order is off.
            float(np.mean(np.abs(np.array(
                inventory_without_twap + _inw, dtype=float))
                >= j["inventorylimit"]))
            if (len(inventory_without_twap) + len(_inw)) else None),
        # Criterion 2: terminal PnL. On a TWAP-absent episode this is the clean
        # standalone-market-maker readout.
        "terminal_pnl": float(_term_pnl) if _term_pnl is not None else None,
        # Criterion 3: TWAP execution cost. None on absent episodes.
        "twap_slippage_bps": _twap_slippage_bps(twap_side, final_cash, total_executed, starting_midprice)
            if twap_present else None,
        "twap_total_executed": float(total_executed) if twap_present else None,
        "start_midprice": float(starting_midprice) if starting_midprice else None,
        "episode_seconds": None,
        "action_flow": action_flow.snapshot(),
    })

    if termination:
        print("Termination condition reached.")
    elif truncation:
        print("Truncation condition reached.")
    else:
        pass
    episode_total_time = time.time() - episode_start_time
    print(f"Episode took {episode_total_time} seconds to run")
    if episode_metrics:
        episode_metrics[-1]["episode_seconds"] = float(episode_total_time)
    _save_episode_metrics()

    if not EVAL_ONLY and ((episode) % 4 == 0):
        if (not RL_DISABLED) and ('test' not in label) and ((checkpoint_params is None) or (episode >= 0)):
            for epoch in range(1):
                start_time = time.time()
                d_policy_loss, d_value_loss, d_entropy_loss, u_policy_loss, u_value_loss, u_entropy_loss = agent.train(train_logger, use_CEM = USE_CEM and bool((episode) % 8) and (episode >= 10))
                train_time = time.time() - start_time
                # store timing on the train_logger (create list if necessary)
                print(f"Agent.train took {train_time:.4f}s for episode {episode}")
                train_logger.save_logs()
            train_logger.plot_losses(show=False, save=True)

        model_manager.save_models(epoch = episode, u = RLagentInstance.Actor_Critic_u, d= RLagentInstance.Actor_Critic_d)
    if TRAJ_PLOT_EVERY and (episode % TRAJ_PLOT_EVERY == 0):
        plot_avg_inventory_trajectories(episode_inv_trajectories_buy, episode_inv_trajectories_sell,
                                        episode, log_dir, label, twap_start_time, twap_end_time)
    for agent in agents:
        if isinstance(agent, PPOAgent):
            agent.current_time = 0
            agent.istruncated = False
            agent.cash = j['cash']
            agent.Inventory = {"INTC": 0}
            agent.positions = {'INTC':{}}
            agent.last_state = None  # Triggers LSTM reset on next get_action()
            agent.breach = False  # Clear stale breach flag; otherwise a prior episode that ended over the inventory limit forces an early-return MO on the new episode's first action (with last_state=None)
            agent.profit = 0
            agent.statelog = [(0, agent.cash, agent.profit, agent.Inventory.copy(), agent.positions.copy(), agent.mid)]
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
    # The control arm (RL_DISABLED) stores no transitions, so the buffer is
    # empty and there are no episodic rewards to summarise. Guard rather than
    # crash: this block is reporting only, and the criterion-3 numbers the
    # control exists for are already written to episode_metrics.
    if len(agent.trajectory_buffer) > 0:
        tmp = agent.trajectory_buffer[0][0]
        for ij in agent.trajectory_buffer:
            if ij[0] == tmp:
                r+=ij[1][3]
            else:
                episodic_rewards.append(r)
                r = ij[1][3]
                tmp=ij[0]
    avgEpisodicRewards.append(np.mean(episodic_rewards[-4:]) if episodic_rewards else np.nan)
    stdEpisodicRewards.append(np.std(episodic_rewards[-4:]) if episodic_rewards else np.nan)
    finalcash.append(cashs[(RLagentID)][-1] + inventories[(RLagentID)][-1]*agent.mid )
    pft = np.array(finalcash) - j["cash"]
    ma = np.convolve(pft, np.ones(5)/5, mode='valid')

    np.save(log_dir+label+"_trajectory_buffer", np.array(agent.trajectory_buffer, dtype=object), allow_pickle=True)

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

# On a resume the in-memory arrays cover post-resume episodes ONLY, so writing
# them to the usual filenames would overwrite -- and destroy -- the original
# run's outputs. Suffix them instead; the pre-resume files stay on disk and the
# two halves can be concatenated afterwards.
_tail = "" if START_EPISODE == 0 else f"_from{START_EPISODE}"
_out = label + _tail          # tail-save label only; `label` itself is unchanged
if _tail:
    print(f"resume: tail .npy outputs suffixed {_tail!r} so the pre-resume "
          f"files are preserved", flush=True)

np.save(log_dir + "inventorydists_" + _out+ "_inventory_without_twap.npy", np.array(inventories_without_twap, dtype=object), allow_pickle=True)

if len(inventories_with_twap_sell) > 0:
    np.save(log_dir + "inventorydists_" + _out+ "_inventory_with_twap_sell.npy", np.array(inventories_with_twap_sell, dtype=object), allow_pickle=True)
if len(inventories_with_twap_buy) > 0:
    np.save(log_dir + "inventorydists_" + _out+ "_inventory_with_twap_buy.npy", np.array(inventories_with_twap_buy, dtype=object), allow_pickle=True)

np.save(log_dir + "slippages_"+_out+"episode_seeds.npy", np.array(episode_seeds))
np.save(log_dir + "slippages_"+_out+"twap_present.npy", np.array(twap_presents))
np.save(log_dir + "slippages_"+_out+"sides.npy", np.array(sides))
np.save(log_dir + "slippages_"+_out+"total_executed.npy", np.array(total_executeds, dtype=float))
np.save(log_dir + "slippages_"+_out+"final_cash.npy", np.array(final_cashs))
np.save(log_dir + "slippages_"+_out+"start_midprice.npy", np.array(start_midprices, dtype=float))
np.save(log_dir+_out+"RL_observations.npy", np.array(total_RL_obsv, dtype=object), allow_pickle=True)

if EVAL_ONLY and not RL_DISABLED:
    from HawkesRLTrading.src.Utils.frozen_eval import assert_frozen
    assert_frozen(RLagentInstance, eval_checkpoint_info)
