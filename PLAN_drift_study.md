# Plan — seed drift diagnosis, randomised-seed retraining, and CIs for Table 8

Written 2026-08-29. Repo `lobSimulations-1`, branch `trying_to_avoid_localoptima`.
Paper: `/Users/alirazajafree/Tackling-Execution-paper/rQUFguide.tex`.

## 0. What is already established (code reading, done)

**Every episode in this study reruns the same Hawkes noise realisation.**

- `HawkesRLTradingEnv.py:126` calls `np.random.seed(self.seed)` inside
  `tradingEnv.__init__`, and `seed` defaults to `1`.
- No runner ever passes `seed`: `AR_RL_Trainer.py:288`, `AR_RL_runner.py:368`,
  `noRL_runner.py:155` all call `tradingEnv(stop_time=..., wall_time_limit=...,
  **kwargs)`. A fresh env is built per episode, so **`np.random` is reset to
  seed 1 at the top of every single episode**, in training and in evaluation.
- `HawkesArrival` draws from the *global* `np.random` stream (`Arrival_Models.py`
  lines 85, 201, 223, 297, 339) — it has no private RNG. So the background order
  flow is driven by the same stream that gets reset to 1.
- The only cross-episode variation comes from `Exchange.py:473`
  (`random.randrange`, Python stdlib `random`, **never seeded anywhere**) plus
  agent feedback. That is enough to make episodes non-identical, but the
  dominant driving noise is a single fixed realisation.
- Consequence: the pre-trading warm-up (t < 100) is essentially the same market
  in every episode, and the whole ensemble is a set of perturbations around one
  path rather than 17 independent draws. A directional drift baked into that
  path is a confound for exactly the buy/sell asymmetry we are trying to explain,
  and 17 "episodes" are not 17 independent samples, so the reported p-values in
  Table 8 are optimistic.

Two supporting observations from data already on disk:

- `twap_alone_out/pricepath_ep*.npy` (buy TWAP alone, 20 eps): drift over the run
  is **positive in all 20 of 20** episodes, mean +26 bps. Consistent with buy
  impact, but also consistent with a shared upward path; not diagnostic on its own.
- The sell matched baseline is raw mean **−9.43 bps, std 40.4** over 17 episodes,
  collapsing to **+0.36 ± 2.50** after the 20-bps-from-median trim (16 of 17 kept;
  buy keeps 14 of 17). Verified this reproduces Table 8's `0.37 ± 2.42` and
  `0.09 ± 1.70` — the `±` in Table 8 is a **population std (ddof=0)**, not a CI,
  and the trim discards up to 3 of 17 episodes. Both facts need stating in the paper.

## 0b. The generator has zero drift by construction — so any drift is a realisation effect

Checked the fitted Hawkes parameters in
`Symmetric_INTC.OQ_ParamsInferredWCutoffEyeMu_sparseInfer_2019-01-02_2019-12-31_CLSLogLin_10`:

- All 12 exogenous baseline intensities pair **exactly** under bid/ask reflection
  (e.g. `mo_Ask == mo_Bid == 0.4463458135512077`).
- All **74** kernels have an exact mirror under the same reflection; zero missing,
  zero differing.
- The initial book is symmetric too: `Exchange.py:60-69` puts ask at 100.01 and
  bid at 99.99 around `PriceMid0=100`, and `Pi_Q0` is 400 on all four levels.

So the data-generating process is exactly symmetric and its expected drift is
**zero**. That rules out the most boring explanation ("the calibrated market
genuinely trends") and leaves exactly two candidates:

- **(a) realisation effect** — seed 1's particular path happens to drift. This is
  the user's hypothesis, and randomising seeds fixes it.
- **(b) simulator-side asymmetry** — a code path that treats bid and ask
  differently (in-spread handling, queue depletion, tie-breaking). Randomising
  seeds would **not** fix this; it would make the study's problem worse by hiding
  a bug behind averaging.

**Phase A must distinguish (a) from (b), and this is now its primary job.** The
varied-seed ensemble is the discriminator: if mean drift across many independent
seeds is indistinguishable from zero, it is (a); if drift persists in expectation
across seeds, it is (b) and the correct response is to find the bug, not to
randomise. The plan previously treated the varied-seed run as merely a control;
it is in fact the decisive test.

## 0c. The paper's own data already indicates downward drift of roughly 4 bps

Table 11 (`tab:advopt t100`, line 837) reports TWAP-alone slippage against a
**common `t=100` benchmark**, struck before any agent trades:

- Sell TWAP alone: **+4.37** bps (seller does *worse* than the t=100 mid)
- Buy TWAP alone: **−3.91** bps (buyer does *better* than the t=100 mid)

Both signs are what a **falling price** produces: a buyer buying into a decline
beats the benchmark, a seller selling into it misses. Symmetric market impact
alone would push both slippages in the *same* direction. Half the gap,
**≈4.1 bps of downward drift**, is the drift term, and since section 0b shows the
generator cannot produce drift in expectation, that 4.1 bps is a property of
seed 1 (or of candidate (b)).

This is prior evidence for the user's hypothesis drawn from the paper's own
numbers, not new simulation. It also means the asymmetry is quantified before
Phase A runs; Phase A's job is to confirm it and settle (a) vs (b).

Note the paper at line 878 **already discloses** the shared Hawkes seed and the
unseeded cancellation generator, and measures run-to-run variation of 0.86 bps
from that source alone. The seeding issue is therefore not a new discovery to the
paper — but its consequence for the buy/sell asymmetry has not been drawn.

## 0d. There is a competing explanation already in the paper

Section "Slippage with a Common Benchmark" (line 833+) attributes the buy/sell
asymmetry to **benchmark contamination**: the arrival benchmark at `t=250` is
struck after the RL agent has already moved the price, so on the buy side the
benchmark rises with the execution price and hides a real +3.38 bps cost. Under
the common `t=100` benchmark the significance flips — buy becomes significant
(p=0.001), sell drops to +1.53 (p=0.19).

Drift and benchmark contamination are **not mutually exclusive, and they
interact**: the t=100 benchmark removes the contamination but is maximally
exposed to drift over the [100,250] window, while the t=250 benchmark is the
reverse. That is very likely why the two measures give opposite answers. Any
drift finding must be reconciled with this section rather than presented
alongside it, and the reconciliation is a paper-level deliverable, not just a
code change.

Nothing above proves drift exists. Phase A tests that, and settles (a) vs (b).

## 0e. RESULT (2026-08-29): the drift is real, but it is NOT the seed

15 independent seeds, agent-free market (`off_time=0`), `stop_time=255`,
completed locally before the local batch was stopped. Drift in bps:

| Window | n | Mean | 95% CI | t-test p | Wilcoxon p | neg/pos |
|---|---|---|---|---|---|---|
| `[0,100]` warm-up | 15 | **-2.67** | [-4.89, -0.45] | 0.022 | 0.027 | 12/3 |
| `[100,250]` | 15 | -1.90 | [-5.24, +1.44] | 0.243 | 0.378 | 9/6 |
| `[0,250]` total | 15 | **-4.57** | [-8.91, -0.22] | 0.041 | 0.033 | 9/5 |

Trimming the two most extreme seeds leaves it intact: -3.42, CI [-6.37, -0.48],
p=0.026.

**Seed 1 is entirely typical.** Its total drift is -2.5 bps, at the **47th
percentile** of the ensemble — essentially the median. Seed 1 is not an unlucky
draw.

**This is case (b), not case (a).** The generator is exactly symmetric
(section 0b), so its expected drift is zero; yet the mean drift over *independent*
seeds is negative and significant. A realisation effect would average away. This
one does not. The conclusion is that **the simulator has a bid/ask asymmetry**,
and therefore:

> **Randomising seeds will not remove this drift.** Phases B and C as originally
> conceived treat the wrong cause. Retraining on randomised seeds would average
> over the asymmetry and hide it rather than fix it.

**Two independent lines of evidence agree on the magnitude.** Section 0c inferred
about 4.1 bps of downward drift from the paper's own `t=100`-benchmark TWAP-alone
baselines. This probe measures -4.57 bps over `[0,250]`. That convergence, from
completely different data, is the strongest part of the result.

**Where to look.** The effect is strongest and most consistent in the `[0,100]`
warm-up window (p=0.022, 12 of 15 seeds negative) and is not significant on its
own over `[100,250]`. That points at the initial book construction and early
dynamics rather than steady state. Already checked and found symmetric: the
exogenous baselines and kernels (0b), the initial price levels
(`Exchange.py:60-69`, ask 100.01 / bid 99.99 about mid 100), `Pi_Q0` (400 on all
four levels), and the L1-depletion/level-promotion blocks
(`Exchange.py:381-394` bid vs `Exchange.py:431-444` ask, exact mirrors). Not yet
checked: order matching and partial fills, in-spread placement
(`Exchange.py:153-155`), the spread accounting, and `generate_queuesize` /
`generate_orders_in_queue` in `Arrival_Models.py`.

**Caveats.** n=15 and p ≈ 0.04 is suggestive, not settled; several windows were
examined, so there is some multiplicity; and these runs stop at t=255, so the
TWAP window `[250,400]` and the post window are not covered. **Cluster job 7305742
(48 seeds, stop_time=550) is running to settle all three** at roughly triple the
sample size and full episode length.

**The `off_time=0` construction is empirically verified.** All 15 seeds report
`actions={12: 254}` — every wake a no-op — with cash and inventory exactly
unchanged at (1000000, 500). The neutered TWAP is genuinely inert.

**Cost, measured.** ~700-770 s wall per 255-second episode under 16-way
contention on an 11-core laptop. A 550-second episode is materially worse, since
the `expApprox=False` Hawkes path costs more as history accumulates. This is
cluster work.

## Phase A4 (NEW, now the critical path) — locate the bid/ask asymmetry

If job 7305742 confirms 0e, the study's blocking problem is a simulator bug, not
a seed choice. Bisect it by construction rather than by reading:

1. **Null the Hawkes feedback.** Re-run with kernels zeroed so the process is
   pure Poisson at the symmetric baselines. Drift must vanish. If it does not,
   the asymmetry is in the exchange/matching layer, and the Hawkes code is
   exonerated.
2. **Symmetry-under-reflection test.** Add a debug mode that mirrors every event
   type (Ask<->Bid) at generation. Under exact symmetry the reflected market's
   drift distribution must match the original's with the sign flipped. Any
   residual is the bug's signature and localises which event type carries it.
3. **Per-event-type accounting.** Log the count and volume of each of the 12
   event types per episode and compare Ask/Bid pairs. A symmetric generator must
   produce statistically equal counts; whichever pair diverges names the
   mechanism.
4. **Instrument the mid.** Decompose the midprice change into contributions from
   inspread placement, L1 depletion, and level promotion, separately per side.

Only once the asymmetry is found and fixed do Phases B, C and D become
meaningful. Seed randomisation remains worth doing for statistical validity
(17 correlated episodes is a real problem regardless), but it is no longer the
fix for the buy/sell asymmetry.

## Phase A — Does seed 1 carry a directional drift? (local, no cluster)

Ground truth first; everything downstream is conditional on this.

**A1 IS RUNNING ON THE CLUSTER.** `HawkesRLTrading/drift_probe.py` +
`drift_probe.sh`, committed as `7f804be` and submitted from
`~/LSTM_fRL/drift_probe/` as job array **7305742**, tasks 1-48 (one seed each,
`-tc 24`), output to `~/LSTM_fRL/drift_probe/out/path_seed<N>.npy` with a
one-line drift summary per task in the `.o` files. Seed 1 is task 1, so the
status-quo seed is measured inside the same ensemble it is compared against.
An earlier local attempt was abandoned: 17 concurrent sims on an 11-core laptop
produced no completed episode in 13 minutes.

**A1 design.** New script `HawkesRLTrading/drift_probe.py`: run the
sim with the TWAP agent neutered and no RL agent, recording the midprice path.
Neutering: keep the TWAP agent in the config but set `off_time=0` — `get_action`
then returns `(12,0)` at every wake (`MetaOrderTradingAgents.py:54`), so agent
count, entity IDs, wake schedule and code path are unchanged while the agent
never trades. This is the minimal-perturbation "no meta-order" market, and it is
the same construction Phase C needs.

Run two ensembles, same config, 40 episodes each:
- `seed=1` every episode (status quo),
- `seed = 1000 + episode` (independent draws).

Report, per ensemble, the drift in bps over each phase — `[100,250]` pre-TWAP,
`[250,400]` TWAP window, `[400,550]` post — as mean, 95% CI, and sign counts.
**Drift is confirmed if the seed-1 ensemble's mean drift CI excludes 0 while the
varied-seed ensemble's straddles 0**, or if the two means differ materially.

**A2. Dumb pegged market maker.** New agent `PeggedMMAgent`
(`HawkesRLTrading/src/SimulationEntities/PeggedMM.py`, registered as
`strategy: "PeggedMM"` in the env dispatch): at each wake, cancel resting quotes
and post one lot at best bid and one at best ask (actions 9 and 2, the same
level-1 actions the TWAP uses), with a hard inventory cap and no signal of any
kind. Symmetric by construction, so in a driftless market its expected P&L is
non-negative (it earns the spread and pays adverse selection); a persistent
directional drift makes it bleed on the side it is leaning into.

Run 40 episodes at `seed=1` and 40 at varied seeds. Report per-episode terminal
mark-to-market P&L (`cash + q·S_t(1 − τ·sgn q)`, τ = 1 bps, matching the paper's
convention), mean with 95% CI, and a decomposition into spread capture vs.
inventory P&L so a loss can be attributed to drift rather than to fee drag.

**A3. Reuse existing evidence.** `HawkesRLTrading/twap_baseline_obs_out/{seed1,vary}/`
already holds a seed-1 vs varied-seed TWAP-alone comparison from earlier work.
Fold it in as a third data point rather than re-running it.

**Decision gate.** If A1+A2 show no drift, Phases B–D still improve statistical
validity (17 correlated episodes is a real problem regardless) but stop being
urgent, and Phase E proceeds either way. If drift is confirmed, B–D are mandatory
and the paper's asymmetry claim needs re-derivation, not just re-wording.

## Phase B — Seed randomisation (small, reviewable diff)

1. `tradingEnv.__init__`: seed Python's `random` alongside numpy, so
   `Exchange.py:473` stops being an unseeded leak and runs become genuinely
   reproducible. This is a behaviour change to *existing* seed-1 runs and must be
   called out in the paper's reproducibility note.
2. Trainer and runner: pass an explicit per-episode seed. Env-var controlled —
   `SEED_MODE=fixed|vary` (default `fixed`, preserving today's behaviour so no
   existing result silently changes) and `SEED_BASE` (default 1000). In `vary`
   mode, `seed = SEED_BASE + episode`. Log the seed per episode and save the seed
   list next to the results so any run can be replayed exactly.
3. Evaluation runs and their matched TWAP-alone baselines must use **the same
   seed list**, which turns the slippage comparison into a paired test (Phase D).

## Phase C — Retraining (cluster)

Read `.claude/skills/ucl-cluster.md` before touching the cluster. Its hard rules:
`qsub` only from inside the run's own directory, and **only one config queued or
running at a time** — the config baked into a job is whatever `~/lobSimulations`
is checked out to when the job *starts*.

All three runs use `SEED_MODE=vary` and the combined arm's hyperparameters
(exploration bonus + GAE fix), which the existing ablation established as the
only arm correct on both sides.

**Alternating TWAP presence.** Even episodes: TWAP active over `[250,400]` as
today. Odd episodes: TWAP neutered via `off_time=0` and `TWAPPresent` pinned to
`0` for the whole episode. This forces the agent to be a viable market maker
standalone rather than only a front-runner, which is the failure the current
results show. Implemented by toggling `kwargs["GymTradingAgent"][1]["off_time"]`
per episode plus a guard on the `TWAPPresent` assignment in the episode loop
(`AR_RL_Trainer.py:313-317`). The agent list is never mutated, so the
`kwargs[...][1]` / `RLagentID = 1` index assumptions elsewhere stay valid.

Three runs, sequential (one job at a time):
1. **buy** — `twap_side = "buy"` on TWAP-present episodes.
2. **sell** — `twap_side = "sell"`.
3. **two-sided** — side drawn at random per TWAP-present episode. This restores
   the randomisation that `AR_RL_Trainer.py:280` currently hard-codes to `"sell"`.

Diagnostics to check per run: inventory sign by TWAP side (the ablation's
criterion), and separately, **P&L on the TWAP-absent episodes** — the new
quantity this design is meant to move, and the one the current results fail.

**Bookkeeping hazards on TWAP-absent episodes** (found by reading the loop; these
must be handled or the new episodes will silently corrupt the recorded metrics):

- `AR_RL_Trainer.py:340` computes `total_executed = abs(500 - agent.Inventory["INTC"])`
  from the TWAP agent. On an absent episode this is `0`, and any slippage figure
  derived from it divides by zero. Slippage must be recorded as NaN / skipped for
  absent episodes, not computed.
- `AR_RL_Trainer.py:356-361` buckets RL inventory into
  `inventory_with_twap_sell` / `inventory_with_twap_buy` purely on the time window
  `(twap_start_time, twap_end_time)` and the run's `twap_side`. On an absent
  episode the RL agent's inventory during `[250,400]` would be filed under
  "with TWAP", contaminating exactly the comparison this study exists to make.
  The bucketing must key on actual TWAP presence, not on the clock.
- `AR_RL_Trainer.py:383-392` buckets `profit_with_twap_*` / `profit_without_twap`
  the same way and has the same defect.
- `starting_midprice` is captured inside the `not isinstance(agent, PPOAgent)`
  branch (`AR_RL_Trainer.py:324-327`), i.e. off the TWAP agent's first wake. A
  no-op TWAP still wakes, so this still fires — but confirm rather than assume.

While fixing this file, also fix the latent `np.save(..., np.array(total_RL_obsv))`
crash at `AR_RL_Trainer.py:570` (needs `dtype=object`), which killed the tail of
the last completed run.

## Phase D — Evaluation

For each new checkpoint, run 40+ episodes with `SEED_MODE=vary`, in three arms:
TWAP present, TWAP absent (RL alone), and pegged-MM control on the same seeds.
Matched TWAP-alone baselines re-run on the **same seed list**, giving paired
per-episode slippage differences. Report the paired test (Wilcoxon + paired t)
as primary and keep the unpaired Welch/MWU for continuity with the current table.

## Phase E — Confidence intervals for Table 8 (`tab:advopt slippage`)

Independent of A–D; the per-episode inputs are all on disk already
(`~/Downloads/new_value_function/ckpt_eval/{baseline_buy,baseline_sell,buy_ep16,
buy_ep40_lim25,buy_ep40_lim50,buy_ep64,sell_ep16}/`, n=17 each, with
`final_cash` / `total_executed` / `start_midprices` for the RL arms). So this can
be done now and does not block on retraining.

Changes:
1. State that `±` is a std, and add a **95% CI on the mean** for every row.
2. Add a **95% CI on Δ** (Welch–Satterthwaite for the difference of means) —
   this is the quantity the paper's claim rests on and currently has no interval.
3. Because the reported statistic is a **trimmed** mean, also give a BCa
   bootstrap CI of the trimmed mean, resampling the raw per-episode slippages and
   re-applying the 20-bps-from-median trim inside each resample, so the trimming
   rule's own variability is priced in.
4. Report **n after trimming** per row (16 sell baseline, 14 buy baseline, etc.),
   since it varies row to row and is currently invisible.
5. Add a footnote that these episodes share seed 1 and are therefore not
   independent, so the intervals are lower bounds on the true uncertainty —
   removable once Phase D's varied-seed runs replace the numbers.

**Already computed** (reconstruction validated — it reproduces every published
Table 8 figure exactly, so the pipeline is trustworthy):

| Row | n (post-trim) | Mean | 95% CI | Delta | 95% CI on Delta |
|---|---|---|---|---|---|
| Sell, matched baseline | 16/17 | +0.36 | [-0.97, +1.69] | --- | --- |
| Sell, combined ckpt 16 | 16/17 | +3.09 | [+1.52, +4.65] | +2.72 | [+0.75, +4.69] |
| Buy, matched baseline  | 14/17 | +0.09 | [-0.93, +1.10] | --- | --- |
| Buy, combined ckpt 16  | 12/17 | +0.22 | [-0.46, +0.89] | +0.13 | [-1.04, +1.29] |
| Buy, combined ckpt 64  | 16/17 | +0.38 | [-0.69, +1.46] | +0.30 | [-1.12, +1.71] |
| Buy, combined ckpt 40  | 16/17 | +0.61 | [-0.29, +1.50] | +0.52 | [-0.78, +1.81] |

The sell effect survives: its Delta CI `[+0.75, +4.69]` excludes zero. But it is
wide enough that the effect is bounded only between "marginal" and "large", which
is a weaker claim than the current text's bare `p = 0.008` implies. No buy row's
CI excludes zero, consistent with the paper.

**A new problem surfaced by this computation:** the bootstrap CI of the trimmed
mean for buy ckpt 16 is `[-0.37, +63.55]`. The 20-bps-from-median trim already
discards 5 of 17 episodes on that row, and under resampling the median moves
enough that the rule sometimes readmits a large outlier. The trim rule is not
stable on this data. That argues for replacing it with a fixed-fraction trimmed
mean or a Huber M-estimator rather than patching a CI onto it.

Deliverable: `HawkesRLTrading/table8_ci.py` (recomputes everything from the .npy
files, prints the LaTeX body) plus the edited table in `rQUFguide.tex`.

## Sequencing

- Phase E and Phase A are independent of each other and of the cluster; both can
  start immediately.
- Phase B is a prerequisite for C and D.
- Phase C is three sequential cluster jobs (~1 job at a time, hard constraint).
- Phase D follows each checkpoint.

## Independent review

A reviewer agent audits (i) the drift diagnosis and whether A1/A2 can actually
distinguish drift from impact, (ii) the Phase B/C diffs, (iii) the Phase E
statistics. Reviewer must not have written the code it reviews.

## Open questions for the user

1. Episode count for the retrains — keep 80, or raise it now that half the
   episodes are TWAP-absent (effectively halving TWAP-present experience)?
2. Alternation ratio — strict 50/50, or a different mix (e.g. 2:1 TWAP-present)?
3. Should the two-sided run also alternate, or is 3-way (buy / sell / none)
   randomisation per episode preferable to the even/odd scheme?


---

# Phase A4 bisection — results (2026-08-29)

## Step 1: null the Hawkes excitation

`DP_NULL_KERNELS=1` zeroes the excitation mask, leaving pure Poisson at the
exactly-symmetric baselines. Agent-free, `stop_time=255`, matched config.

| Arm | n | drift `[0,100]` | drift `[0,250]` |
|---|---|---|---|
| Excitation ON | 15 | **-2.67**, CI [-4.89,-0.45], p=0.022 | **-4.57**, CI [-8.91,-0.22], p=0.041 |
| Excitation OFF | 40 | -0.05, CI [-0.88,+0.78], p=0.90 | -0.75, CI [-2.16,+0.66], p=0.29 |

With the excitation off the drift is tightly bounded around zero, and the nulled
CI `[-2.16,+0.66]` excludes the excitation-on point estimate of -4.57. **But the
direct two-sample test is not significant**: difference -3.83 bps, 95% CI
[-8.34,+0.68], Welch p=0.091, MWU p=0.117 — the excitation-on arm is only n=15
with sd 7.85.

**Reading: the drift is in the excitation path, not the exchange/matching layer
— evidence consistent but underpowered.** Cluster arrays 7305748 (excitation on)
and 7305751 (excitation off), 48 seeds each at `stop_time=550`, will settle it
like-for-like.

## Step 3: per-event-type counts

Mirror pairs are `cols[i]` vs `cols[11-i]`; a symmetric generator must produce
equal counts in expectation. Ask minus Bid:

| Event pair | Excitation OFF (n=40) | Excitation ON (n=30) |
|---|---|---|
| `lo_deep` | **+12.32**, p<0.0001 | **+11.57**, p=0.0037 |
| `lo_top` | -1.93, p=0.76 | +35.87, p=0.067 |
| `co_deep` | +4.10, p=0.16 | +6.27, p=0.25 |
| `co_top` | -7.45, p=0.21 | -0.87, p=0.96 |
| `mo` | +0.97, p=0.67 | +2.27, p=0.47 |
| `lo_inspread` | -1.38, p=0.63 | +1.47, p=0.30 |

Two distinct things:

- **`lo_deep` carries a kernel-independent Ask excess of about +12**, essentially
  identical with and without excitation. This is a real generator-level asymmetry
  and a genuine bug, but `lo_deep` sits at L2, away from the touch, so it does
  not move the mid — consistent with the drift vanishing under nulling while this
  bias persists. It should be fixed, but it is probably not the drift mechanism.
- **`lo_top` shows an Ask excess only when the excitation is on** (+35.87 vs
  -1.93 with it off). `lo_top` *is* at the touch and does move the mid. This is
  the leading candidate for the drift mechanism, but at p=0.067 and n=30 it is
  not yet established.

## Refuted: the dimension-assignment walk

`lo_deep_Ask` is index 0 and `lo_deep_Bid` index 11 — the two ends of the
assignment walk at `Arrival_Models.py:346-349`, so a boundary bias there was the
obvious suspect. Tested directly: 2,000,000 draws against the real baseline
vector at three `lamb_bar/lamb` ratios (1.0, 1.5, 3.0). All 12 dimensions within
noise (|z| < 2.4), mirror-pair differences random in sign at every ratio.
**The walk is unbiased — REFUTED, not the bug.** The source of the `lo_deep`
excess is still unidentified.

## Still to do

- Generator-in-isolation run (no exchange, spread pinned) to establish whether
  the `lo_deep` excess originates in the point process or in the exchange
  interaction. A first attempt timed out and needs a cheaper configuration.
- Step 2 (reflection-symmetry test) and step 4 (mid-change decomposition by
  side) not yet started; both are better run once the cluster arrays land and
  the `lo_top` signal is either confirmed or dismissed at higher power.

---

# ROOT CAUSE FOUND (2026-08-30): thinning upper-bound violation biases dimension assignment

## Step 1b result — generator in isolation (n=48 seeds each, T=550, spread pinned)

`generator_probe.py` runs `HawkesArrival` with the exchange removed entirely, so
the *only* difference between arms is `kernelparams[0][0] = 0`.

| pair | excitation OFF: Ask−Bid | excitation ON: Ask−Bid |
|---|---|---|
| `lo_deep`      | +3.83  (+1.03%), p=0.35 | **+45.44 (+6.38%), p=4e-09** |
| `co_deep`      | −0.73  (−0.16%), p=0.86 | **+67.35 (+4.79%), p=8e-09** |
| `lo_top`       | −6.96  (−0.42%), p=0.32 | **+98.88 (+2.03%), p=0.0025** |
| `co_top`       | +0.29  (+0.02%), p=0.98 | +14.29 (+0.31%), p=0.52 |
| `mo`           | −2.62  (−1.07%), p=0.34 | −4.54 (−1.55%), p=0.29 |
| `lo_inspread`  | −1.71  (−0.53%), p=0.64 | −4.98 (−1.42%), p=0.23 |
| **TOTAL**      | −7.9 (−0.16%), p=0.57   | **+216.4 (+1.77%), p=7e-06** |

Two conclusions, one of which **overturns the earlier full-sim read**:

1. With excitation off the isolated generator is **exactly symmetric**. Baselines,
   thinning and the assignment walk are all clean.
2. With excitation on it is **strongly Ask-biased, with no exchange involved at
   all**. The bias is therefore inside the Hawkes machinery, not the LOB.

(The earlier full-sim finding that `lo_deep` showed an Ask excess *even with
kernels nulled* is a **separate, smaller, exchange-side** asymmetry. It does not
move the mid — consistent with drift ≈ 0 in the nulled full-sim arm.)

## The parameters are not at fault

All four 12×12 matrices in `kernelparams[0]` and the 12×1 baseline satisfy
`K[i][j] == K[11-i][11-j]` to **exactly 0.0** deviation. The process is
mirror-equivariant by construction, so any Ask bias is a code defect.

## The defect

`Arrival_Models.py:339-350`. One uniform `D` does double duty:

```python
D = np.random.uniform(0, 1)
if D*lamb_bar <= self.lamb:          # 342: acceptance test
    k = 0; total = decays[k]
    while D*lamb_bar >= total:       # 348: dimension assignment, walks index 0->11
        k += 1; total += decays[k]
```

Reusing `D` for the assignment is the standard Ogata trick and is **valid only
while `lamb_bar` is a true upper bound on the intensity**. `lamb_bar` is the
previous iteration's `self.lamb` inflated by a crude one-event bump (line 355-360),
which is *not* a valid bound: clustering routinely pushes the realised intensity
above it.

When `self.lamb > lamb_bar` the point is accepted with probability 1, and
`D*lamb_bar` is drawn from `U[0, lamb_bar]` — truncated **below** `sum(decays)`.
The walk starts at index 0, so the mass above `lamb_bar` is unreachable: the tail
of the index order can never be selected, and every dimension below it is
over-sampled by the factor `lamb/lamb_bar`.

The index order is Ask-first:

```
0 lo_deep_Ask  1 co_deep_Ask  2 lo_top_Ask  3 co_top_Ask  4 mo_Ask  5 lo_inspread_Ask
6 lo_inspread_Bid  7 mo_Bid  8 co_top_Bid  9 lo_top_Bid  10 co_deep_Bid  11 lo_deep_Bid
```

so truncation is **systematically anti-Bid**. The observed excess is monotone in
index — +6.38, +4.79, +2.03, +0.31, −1.55, −1.42 % for indices 0→5 — which is the
exact signature of this truncation, not of any economic mechanism.

## Direct measurement of the violation rate

Instrumented run (8 seeds, T=25, generator in isolation):

| | excitation ON | excitation OFF |
|---|---|---|
| candidates | 20028 | 3559 |
| bound violations | 538 (2.69%) | 8 (0.22%) |
| accepted while violated | 538 (**6.23% of all events**) | 8 (0.22%) |
| mean intensity mass unreachable | 6.36% | 3.09% |

Every violation is accepted, so **6.2% of all simulated events with excitation on
have their event type drawn from a truncated, Ask-favouring distribution**, and
0.2% with excitation off. This is the drift.

## Consequences

- **The drift is a simulator bug, not a seed artifact.** Seed 1 is at the 47th
  percentile of the 15-seed drift distribution. Randomising seeds averages over
  the bug, it does not remove it. **Phases B/C/D (seed randomisation + retrain)
  should not start until this is fixed.**
- Every result in the paper that depends on directional symmetry — the buy/sell
  frontrunning asymmetry above all — is contaminated by a −4.6 bps systematic
  downward drift.

## Fix

Decouple the assignment from the (possibly violated) bound. Given that a point
occurred at `self.s`, its type is proportional to `decays` regardless of `lamb_bar`:

```python
V = np.random.uniform(0, 1) * self.lamb     # fresh draw over the ACTUAL intensity
k = 0; total = decays[k]
while V >= total:
    k += 1; total += decays[k]
```

This is exactly correct whenever the bound holds, and removes the directional
bias when it does not. It does not repair the residual bias in event *timing*
from an invalid bound — that needs a genuinely valid `lamb_bar` (e.g. bump by
`max_j sum_i K[j][i]` rather than the realised `k`'s row) and should be a second,
separate change so the two effects can be measured apart.

## Validation plan

1. Apply the assignment fix; re-run `generator_probe` 48 seeds x2 arms. Expect
   all six mirror pairs to return to the excitation-OFF (symmetric) column.
2. Re-run `drift_probe` 48 seeds x2 arms. Expect total drift to come back to 0
   within CI in both arms (currently −4.61 bps ON / +0.61 OFF).
3. Then chase the smaller exchange-side `lo_deep` asymmetry separately.
4. Only then restart Phases B/C/D.

---

# FIX VALIDATED (2026-09-01): steps 1-4 complete

Commit `6e94941`. `sample_dimension()` draws the fired dimension from a fresh
uniform over the realised intensity instead of reusing the thinning variate.

## Step 1 — blast radius

`src/fit/` contains **no** thinning code: `ConditionalLeastSquaresLogLin`, `MLE`
and `PlainHawkes` fit from data and never simulate. **The fitted parameters are
not contaminated** — the bug is confined to simulation, so the fix does not
invalidate the calibration.

The pattern occurred in three places:

| file | status |
|---|---|
| `HawkesRLTrading/src/Stochastic_Processes/Arrival_Models.py:346-350` | fixed |
| `src/simulation/Simulate.py:478-482` | fixed (identical bug, live path) |
| `src/backup/hawkes/simulate_optimized.py:158-162` | left alone, archival |

## Step 2 — regression test

`HawkesRLTrading/tests/test_sample_dimension.py` (the repo had no tests). Keeps
a `legacy_assign` copy of the old rule so the bias is pinned, not just described.
Two of its four assertions fail against pre-fix behaviour.

## Step 3 — generator in isolation, kernels ON (n=48 each)

Events/seed 24688.2 -> 24667.2 (-0.08%, p=0.80): the fix reallocates types
without changing the arrival rate, as intended.

| pair | PRE diff | PRE % | PRE p | POST diff | POST % | POST 95% CI | POST p |
|---|---|---|---|---|---|---|---|
| `lo_deep`     | +45.44 | +6.38% | 4.1e-09 | +3.52 | +0.46% | [−13.42, +20.46] | 0.69 |
| `co_deep`     | +67.35 | +4.79% | 8e-09   | +21.54 | +1.49% | [−0.47, +43.56] | 0.061 |
| `lo_top`      | +98.88 | +2.03% | 0.0025  | +34.71 | +0.71% | [−32.90, +102.31] | 0.32 |
| `co_top`      | +14.29 | +0.31% | 0.52    | +6.54 | +0.14% | [−43.11, +56.20] | 0.80 |
| `mo`          | −4.54 | −1.55% | 0.29    | −6.40 | −2.22% | [−13.80, +1.01] | 0.097 |
| `lo_inspread` | −4.98 | −1.42% | 0.23    | −1.25 | −0.36% | [−9.68, +7.18] | 0.77 |
| **TOTAL**     | **+216.44** | **+1.77%** | **7.1e-06** | **+58.67** | **+0.48%** | [−54.09, +171.43] | **0.31** |

The **reduction** is itself significant, so this is not just loss of power:
`lo_deep` −41.92 (p=0.0002), `co_deep` −45.81 (p=0.0026), TOTAL −157.77 (p=0.031).

**Control** (kernels OFF, where the bound is rarely violated and the fix should
be inert): events/seed 9919.4 -> 9922.2 (p=0.89), max per-dimension rate change
0.99%, all six pairs symmetric before and after. This rules out a second error
cancelling the first.

## Step 4 — full-simulation midprice drift, agent-free (bps over 550s, n=48)

| arm | drift | 95% CI | p vs 0 |
|---|---|---|---|
| PRE-FIX  kernels ON  | **−4.48** | [−7.21, −1.74] | **0.0024** |
| PRE-FIX  kernels OFF | +0.61 | [−1.10, +2.33] | 0.49 |
| POST-FIX kernels ON  | **+0.53** | [−2.03, +3.09] | **0.69** |
| POST-FIX kernels OFF | −0.21 | [−1.77, +1.35] | 0.79 |

- PRE-FIX  ON−OFF = −5.09 [−8.32, −1.87], Welch p=0.0027, MWU p=0.0022
- POST-FIX ON−OFF = +0.74 [−2.26, +3.74], Welch p=0.63, MWU p=0.44
- fix effect on the ON arm = **+5.01 bps [+1.27, +8.76], Welch p=0.0102**

**The drift is gone.** All four pass criteria met.

## Residuals

- `co_deep` +1.49% (p=0.061) is the largest remaining generator asymmetry —
  not significant, but it is the one to watch. Plausibly the exchange-side
  effect (step 5) leaking in, or a second smaller defect.
- Event *timing* is still biased by the invalid bound (step 6); untouched here.
- `Exchange.py:473` still uses an unseeded stdlib `random.randrange`.

## Consequence for the paper

Every trained checkpoint predates this fix and was trained against a simulator
with a −4.5 bps systematic downward drift. The buy/sell frontrunning asymmetry
in particular cannot be trusted until models are retrained. Table 8 numbers
will change, so `table8_ci.py` can be built now but the LaTeX edit should wait.

---

# co_deep investigation (2026-09-02): a second bug — stale `self.left` after purge

Chasing the `co_deep` +1.49% (p=0.061) residual left by the sampling fix.

## Ruled out by reading

In the isolated generator the only possible asymmetry sources are baselines
(verified mirror-exact), kernels (verified mirror-exact), the dimension sampling
(fixed in `6e94941`), and the spread multiplier on dims 5/6 — which is applied
symmetrically to the `lo_inspread` mirror pair, with the spread pinned at 0.02
so the `<2 tick` cutoff at line 333 never fires asymmetrically. `self.baselines`
is re-copied from `kernelparams[1]` at line 263 on every call, so the spread
multiplier at 273-274 does not compound across calls.

## Found: `Arrival_Models.py:374-375`

```python
if self.timeseries[-1][0] - self.timeseries[0][0] > self.TAU:
    self.timeseries = self.timeseries[self.left:]   # re-bases the list
    # self.left NOT reset -- the bug
```

The slice re-bases the list: what was at index `self.left` becomes index 0. But
`self.left` keeps its old value, so the next window advance starts `self.left`
points too far in and **silently drops the kernel contributions of points that
are still inside TAU**. The effective excitation window shrinks with every purge.

Demonstrated deterministically (TAU=500, one point per second): the step after
the first purge uses **399 points instead of 499**, dropping 100 in-window points
spanning t=102..201.

**Why it was invisible until now:** the purge only fires once the history spans
more than TAU=500s. The T=120 diagnostic runs show `purges=0` — the bug is inert
there. It is active in every T=550 probe, which is exactly where the `co_deep`
residual was measured.

`src/simulation/Simulate.py` never truncates its history (it only advances `left`
over a hardcoded 10s window) and does not share this bug.

Fixed in `4294506`; `self.left = 0` after the slice. Two regression tests added,
one of which pins the legacy behaviour.

## Status

Whether this explains `co_deep` is an empirical question, not settled by finding
the bug. Arrays 7330866-9 (`p2_*`, 48 seeds x 4 arms, T=550) test it. Note the
purge bug is provably inert in the kernels-nulled arms — with `kernelparams[0][0]`
zeroed every kernel contribution is zero regardless of which points are in the
window — so those two arms serve as a null control: they should be unchanged.

Pass criterion: `co_deep` Ask-Bid excess drops toward zero in the kernels-ON
generator arm, and the drift stays within CI of zero.

---

# co_deep resolved (2026-09-03): the purge fix is a no-op here, and co_deep is not established

## The purge fix changes nothing at T=550

| pair | original | +sampling fix | +sampling+purge fix | % | 95% CI | p |
|---|---|---|---|---|---|---|
| `lo_deep`     | +45.44 | +3.52 | +3.62 | +0.47% | [−13.32, +20.57] | 0.68 |
| `co_deep`     | +67.35 | +21.54 | **+21.67** | +1.50% | [−0.30, +43.64] | 0.059 |
| `lo_top`      | +98.88 | +34.71 | +34.67 | +0.71% | [−32.92, +102.26] | 0.32 |
| `co_top`      | +14.29 | +6.54 | +6.38 | +0.14% | [−43.31, +56.06] | 0.80 |
| `mo`          | −4.54 | −6.40 | −6.35 | −2.21% | [−13.73, +1.03] | 0.098 |
| `lo_inspread` | −4.98 | −1.25 | −1.21 | −0.35% | [−9.64, +7.23] | 0.78 |
| **TOTAL**     | +216.44 | +58.67 | +58.77 | +0.48% | [−54.04, +171.59] | 0.31 |

co_deep reduction from the purge fix: **−0.12, Welch p=0.994**. Drift unchanged
(+0.53 -> +0.58 bps, p=0.977); ON−OFF = +0.79 [−2.16, +3.74], p=0.60.
Null control (kernels OFF) bit-identical as predicted, p=1.00.

**Why the purge fix is inert at this horizon:** the purge only starts firing once
the history spans TAU=500s, and episodes are 550s. It is therefore active for
only the last ~50s, giving the stale offset almost no time to accumulate. The
bug is real and worth having fixed — it makes the effective excitation window
shrink over a run, a calibration error in any longer simulation — but it does
**not** explain `co_deep` and does not affect 550s results.

## co_deep is not established as a real asymmetry

- **Multiple comparisons:** six mirror pairs are tested. Holm and BH both reject
  co_deep (crit 0.0083 vs raw p=0.059). P(min of 6 p-values < 0.059 under perfect
  symmetry) = **0.31** — seeing one pair look this marginal is expected by chance.
- **Omnibus:** Hotelling T² on the joint 6-vector of mirror differences,
  T²=6.21, F(6,42)=0.925, **p=0.487**. The generator is jointly symmetric.
- **Sign test:** only **26/48 seeds positive** (binomial p=0.665, Wilcoxon
  p=0.113). A systematic Ask excess would show a consistent sign; this does not.
- **Not outlier-driven:** skew +0.02, kurtosis −0.84, Shapiro p=0.31, mean +21.7
  vs 10%-trimmed +22.8. Clean, normal, symmetric — genuinely underpowered rather
  than contaminated.
- **Power:** n=101 for 80%, n=136 for 90% (sd=77.7). At n=48 the study cannot
  resolve an effect this size either way.

Array 7332435 runs 150 further seeds (49-198) to pool to n=198 and settle it.

## Bottom line

Two real bugs found and fixed (`6e94941` sampling, `4294506` purge). The drift is
gone and the generator is jointly symmetric on every test available. `co_deep` is
the largest of six residuals and is consistent with noise; it is **not** a reason
to hold retraining.

---

# co_deep settled at n=198 (2026-09-04): not real. A `mo` signal appears instead.

Pooling arrays 7332435 (150 new seeds) with the 48 existing = **n=198**, all on
the fully fixed simulator, kernels ON.

| pair | Ask−Bid | % | 95% CI | p | seeds positive |
|---|---|---|---|---|---|
| `lo_deep`     | +2.85 | +0.38% | [−4.57, +10.28] | 0.452 | 101/198 |
| `co_deep`     | **+4.83** | **+0.33%** | [−5.94, +15.61] | **0.380** | **98/198** |
| `lo_top`      | +15.51 | +0.32% | [−17.00, +48.02] | 0.351 | 105/198 |
| `co_top`      | +1.87 | +0.04% | [−21.40, +25.14] | 0.875 | 105/198 |
| `mo`          | −5.35 | −1.86% | [−9.52, −1.19] | 0.013 | 80/198 |
| `lo_inspread` | −2.02 | −0.59% | [−6.14, +2.10] | 0.338 | 97/198 |

**`co_deep` is resolved as noise.** It fell from +21.67 (n=48, p=0.059) to +4.83
(n=198, p=0.380) — an effect that shrinks by 4x as power quadruples is a null.
98/198 seeds positive is a coin flip. The n=48 marginal p-value was the largest
of six residuals, exactly as the multiplicity analysis predicted.

## The `mo` residual — honest status: unresolved, probably not real

It is the only pair that firmed up with more data (p=0.288 -> 0.097 -> 0.049 as
n grew), which is what a real effect does. Against it:

- **Fails multiplicity correction.** Holm critical value for the smallest of six
  p-values is 0.0083; `mo` is at 0.0125 pooled. Does not survive.
- **Omnibus is null:** Hotelling T² F(6,192)=1.552, **p=0.163**. The joint
  6-vector of mirror differences is not distinguishable from zero.
- **Unchanged by both fixes** (−4.54 buggy, −6.40 sampling-fixed, −6.35
  purge-fixed): whatever it is, it is not what those bugs caused.
- Sign test excluding 5 ties: 80/193, p=0.021 — same marginal territory.
- `mo` is the **smallest-count dimension** (282 vs `lo_top`'s 4926), so it is the
  most susceptible to small absolute biases showing up as large percentages.

It is also present but not significant in the kernels-OFF arm (−1.50%, p=0.167,
19/48), which is weak evidence it does not originate in the excitation path.

**Verdict: not established, and not a blocker.** It is −1.86% on the rarest
event type, with a null omnibus. It is worth one more look after the MM probe
settles the end-to-end question, but it does not justify holding retraining.

## Method note

Reporting `mo` at all is a multiplicity artifact risk in the other direction:
with six pairs re-tested at every code version, something will eventually sit
near p=0.05. The omnibus test is the guard against that, and it says nothing
is there.

---

# PeggedMM probe, round 1 (2026-09-05): the control failed — the probe perturbs the market

Three arms, n=48, 550s, `order_size=100`. The third arm runs the **pre-fix
simulator** (clone at `002bbf3`, verified to contain the legacy walk and no
`sample_dimension`) with identical agent and probe files, so only the simulator
differs. It exists to answer: can this probe detect a drift that is known to be
there?

| arm | MtM PnL | 95% CI | p vs 0 |
|---|---|---|---|
| BUGGY code, kernels ON  | +38.8 | [+33.6, +43.9] | 2.9e-19 |
| FIXED code, kernels ON  | +32.9 | [+27.7, +38.0] | 1.3e-16 |
| FIXED code, kernels OFF | +58.0 | [+53.9, +62.1] | 7.1e-31 |

**The control fails.** The buggy arm made money (+38.8), essentially as much as
the fixed arm; fixed − buggy = −5.9 [−13.2, +1.3], Welch p=0.11. 1/48 buggy seeds
lost money vs 4/48 fixed. **A null result from this probe would have been
meaningless**, which is exactly why the arm was run.

## Why: the MM damps the drift it is meant to measure

| arm | drift with MM present | agent-free drift |
|---|---|---|
| BUGGY ON  | **−0.63** [−1.56, +0.31] | **−4.48** |
| FIXED ON  | +0.84 [−0.22, +1.91] | +0.58 |
| FIXED OFF | −0.27 [−1.39, +0.85] | −0.21 |

In the buggy arm the drift collapses from −4.48 to −0.63 once the MM is quoting:
one-sample t against −4.48 gives **p=1.9e-10**. At `order_size=100` against an
initial queue of 400 the agent is **25% of the touch**, so its two-sided quoting
absorbs the order flow that would otherwise move the mid. The fixed arms are
unaffected because there is no drift there to damp.

Spread PnL (~+33 to +58, hugely significant in every arm) also swamps the
inventory PnL (~−10), so MtM PnL is the wrong endpoint at this size.

## The signal that did survive

**Inventory skew is the right endpoint.** A symmetric MM in a drifting market
accumulates on the losing side:

| arm | d_inv | 95% CI |
|---|---|---|
| BUGGY ON  | **+294.6** | [−8.0, +597.1] |
| FIXED ON  | −217.9 | [−483.5, +47.7] |
| FIXED OFF | −49.6 | [−176.0, +76.9] |

buggy vs fixed **Welch p=0.0144**, and in the predicted direction: long in the
falling (buggy) market. This survives even with the damping working against it.

## Round 2

Arrays 7336071-3: `order_size=10` (2.5% of the touch instead of 25%), n=150 per
arm. Validity check before reading anything else: the buggy arm's measured drift
must return to ≈−4.5, confirming the agent is no longer perturbing. Primary
endpoint is inventory skew, not PnL.

---

# PeggedMM probe, round 2 (2026-09-07): control passes, fix confirmed end-to-end

`order_size=10` (2.5% of the touch, down from 25%), n=150 per arm.

## Gate 1 — residual perturbation, partially failed

| arm | drift with MM | 95% CI | agent-free | p vs agent-free |
|---|---|---|---|---|
| BUGGY ON  | **−2.41** | [−3.32, −1.50] | **−4.48** | **<0.001** |
| FIXED ON  | +0.32 | [−0.55, +1.20] | +0.58 | 0.567 |
| FIXED OFF | +0.43 | [−0.37, +1.23] | −0.21 | 0.120 |

The agent still damps the drift, from −4.48 to −2.41 — about 46% suppression,
down from 86% at size 100. **The gate is not cleanly passed.** The bias is
conservative: damping makes the bug *harder* to detect, so a positive detection
still counts, but any effect size read off this probe is an underestimate.

## Endpoint — inventory skew. The control now works.

| arm | d_inv | 95% CI | p vs 0 |
|---|---|---|---|
| BUGGY ON  | **+35.1** | [+12.9, +57.2] | **0.0023** |
| FIXED ON  | −15.7 | [−36.7, +5.3] | 0.146 |
| FIXED OFF | −4.5 | [−17.9, +8.8] | 0.507 |

buggy − fixed = **+50.7 [+20.2, +81.3], Welch p=0.0013, MWU p=0.0006.**

The probe detects the known bug, so its reading on the fixed simulator is
meaningful: **no inventory skew on fixed code, clear skew on buggy code.**

## Secondary — MtM PnL

| arm | MtM | 95% CI | losing seeds |
|---|---|---|---|
| BUGGY ON  | +3.96 | [+2.96, +4.95] | 26/150 |
| FIXED ON  | +5.71 | [+4.96, +6.47] | 14/150 |
| FIXED OFF | +9.45 | [+8.93, +9.96] | 1/150 |

fixed − buggy = +1.76 [+0.51, +3.00], Welch **p=0.0061**. The bug costs the
symmetric MM money, which is what the original hypothesis predicted. At size 100
this comparison was null (p=0.11) purely because the agent damped the drift.

## Mechanism — confirmed per seed, not just in aggregate

corr(drift, d_inv): BUGGY −0.593 (p=1e-15), FIXED ON −0.312 (p=1e-4),
FIXED OFF −0.363 (p=5e-6), **pooled −0.455 (p=2.5e-24, n=450)**.
Slope **−10.0 shares of inventory per bps of drift**.

Falling prices leave the symmetric MM long, exactly as it must. The relationship
holds within every arm, so it is the drift driving the inventory and not an
artifact of comparing arms.

## Verdict

The end-to-end check passes. On the fixed simulator an exactly symmetric market
maker shows no inventory skew and no directional edge; on the pre-fix simulator
the same agent skews long and loses relative to it. This confirms the fix through
the whole stack — generator, exchange, matching, agent bookkeeping — without
relying on any theory of where the asymmetry originated.

**Caveat to carry forward:** the probe underestimates, because the agent still
damps the drift ~46% at size 10. A cleaner run would use size 1-2, at the cost
of needing more seeds for the same power.
