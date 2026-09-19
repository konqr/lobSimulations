# Plan — pass the three gates

Written 2026-09-13. Branch `trying_to_avoid_localoptima`.
This plan uses ASD-STE100 Simplified Technical English.
It replaces section 7 of `HANDOFF.md`.

---

## 1. Correction to the handoff

The handoff says gate 3 was "NEVER TESTED" and "the denominator did not exist".
This is wrong.

The paper `/Users/alirazajafree/Tackling-Execution-paper/rQUFguide.tex` tests
gate 3. Section "Matched TWAP Baseline" gives the method. Table
`tab:advopt slippage` gives the result.

| Arm | Slippage (bps) | Delta | Welch p | MWU p |
|---|---|---|---|---|
| Sell TWAP, matched baseline | 0.37 +- 2.42 | — | — | — |
| Sell TWAP, combined ckpt 16 | **3.09 +- 2.84** | **+2.72** | **0.008** | **0.006** |
| Buy TWAP, matched baseline | 0.09 +- 1.70 | — | — | — |
| Buy TWAP, combined ckpt 16/40/64 | 0.22 / 0.61 / 0.38 | +0.13 / +0.52 / +0.30 | 0.82 / 0.42 / 0.67 | n.s. |

The baseline uses the same simulation time, the same TWAP parameters and the
same Hawkes parameters as the test. Only the market maker is absent. The test
uses 17 episodes and a robust mean.

**Gate 3 has a PASS on the sell side.** Gate 3 fails on the buy side.

The code for the baseline is `HawkesRLTrading/twap_baseline_obs.py`. The results
are in `HawkesRLTrading/twap_baseline_obs_out/` and on the cluster at
`~/LSTM_fRL/wout_expo/new_value_function/explo_gae/baseline/`. The handoff lists
these files as "long-standing untracked and intentionally left alone". The
previous agent did not recognise them. It then started two new control jobs
(`ctl_n`, `ctl_f`) to make a denominator that already existed.

I will add these files to git. A gate metric must not live in untracked files.

---

## 2. What the new evidence shows

I read the live per-episode metrics from all 12 running jobs.

### 2a. Every run that alternates the TWAP side fails gate 1

The table gives the response contrast. The response contrast is the mean
in-window inventory minus the mean pre-window inventory, for buy episodes minus
sell episodes. A correct front-runner gives a large positive number.

| Arm | n | Reward | Action space | Buy response | Sell response | Contrast |
|---|---|---|---|---|---|---|
| `ps_all_scaled_fast` | 49 | PnL scale | full | -10.93 | -5.47 | **-5.46** |
| `ps_full_fast` | 48 | PnL scale | full | -8.07 | -2.53 | **-5.55** |
| `ps_no_cem_fast` | 48 | PnL scale | full | -4.30 | -0.59 | **-3.70** |
| `ps_legacy_space` | 21 | PnL scale | 4 actions | -8.19 | -1.54 | **-6.65** |
| `eg_both_onoff_fast` | 50 | historical | 4 actions | +0.10 | +0.32 | **-0.21** |
| `eg_both_onoff` | 13 | historical | 4 actions | -0.57 | +0.35 | **-0.91** |

All 12 arms fail. The sign is wrong or the value is zero in every case. The
agent takes the same side in buy episodes and in sell episodes.

This result is important. It tests the handoff's main hypothesis. That
hypothesis said the reward shaping hides the front-running signal. The
`pnl_scaled` arms correct the shaping. They still fail gate 1. They fail worse
than the historical arms. **Reward rescaling alone does not give front-running.**

### 2b. The single-side runs in the paper do front-run

The paper trains one agent for the buy meta-order and a second agent for the
sell meta-order. Both agents take the correct side:

* The buy-trained agent holds +15.1 inventory. The target is long.
* The sell-trained agent holds -4.65 inventory. The target is short.
* The sell-trained agent increases the TWAP cost by 2.72 bps, at p = 0.008.

The contrast is clear. Single-side agents front-run. Agents that alternate the
side do not front-run.

### 2c. Why side alternation prevents front-running

The policy has one signal for the side. That signal is the `TWAPPresent`
feature. Its value is -1, 0 or +1.

Every other reward term is the same on both sides. The action bonus is the same.
The exploration bonus is the same. The inventory penalty is the same.

One LSTM network must learn two opposite responses from one scalar feature. The
easy solution is a policy that ignores the feature. The agent then holds the
same inventory on both sides. This is what the data shows.

Two separate agents do not have this problem. Each agent learns one response.

**Your suspicion is correct, and the data supports it.** I recommend that we
stop the side alternation.

### 2d. The TWAP on/off cycle is a different question

The evidence does not show that the on/off cycle is bad. Every run that
alternates the side also uses the on/off cycle. I cannot separate the two
effects from the existing data.

The on/off cycle has a benefit. Gate 2 needs episodes with no meta-order. The
paper's agents never trained on such episodes. Their solo profit is -0.04 and
-0.07. Gate 2 fails.

The `pnl_scaled` arms do train on such episodes. Their solo profit is positive:

| Arm | Solo PnL (TWAP absent) | 95% CI |
|---|---|---|
| `ps_legacy_space` | +0.64 | [-0.50, +1.78] |
| `ps_no_cem_fast` | +0.49 | [-0.09, +1.07] |
| `ps_all_scaled_fast` | +0.45 | [-0.16, +1.06] |
| `eg_both_onoff_fast` (historical reward) | -0.05 | [-0.37, +0.27] |

The numbers are positive but the CI includes zero. This is the best gate-2
evidence in the project. **Reward rescaling helps gate 2. It does not help
gate 1.**

I will keep the on/off cycle and test it directly with one arm.

### 2e. Nobody has run the two good ideas together

Single-side training gives gate 1 and gate 3. PnL-scaled reward gives gate 2.
No run has both. **That is the experiment this plan runs.**

### 2f. Why gate 3 fails on the buy side

The paper gives the reason. The RL agent starts to trade at t = 100. The
meta-order starts at t = 250. The slippage benchmark is the mid price at t = 250.

The market price falls by 4.00 bps between t = 100 and t = 250 with no agent.
Both agents reduce this fall. The buy-trained agent reduces it most.

The benchmark therefore absorbs 150 seconds of the agent's own price impact.
The measure does not see this impact. A stronger long position gives a *lower*
measured slippage. The paper confirms this with a larger inventory limit.

I will not change the gate. I will add a second measure as a diagnostic. The
second measure uses the mid price at t = 100 as the benchmark. The agent has not
traded at t = 100. This measure is harder, not easier.

### 2g. The exponential approximation is a good screen

`EXP_APPROX=true` gives about 500 s per episode. The normal regime gives about
3800 s per episode. The speed gain is 7.6 times.

The approximation changes the market. It uses an exponential kernel and a
history window of 10 s. The normal regime uses a power-law kernel and a window
of 500 s.

The live data shows two things:

1. The fast arms and their normal twins agree on gate 1 and gate 2. Example:
   `ps_full_fast` gives a buy response of -8.07. `ps_full_space` gives -5.32.
   Both are negative. Both fail.
2. The fast arms and their normal twins **disagree** on gate 3. The control
   gives +19.02 bps buy slippage in the fast regime. It gives +2.08 bps in the
   normal regime.

**Rule: screen arms in the fast regime. Decide all three gates in the normal
regime.** Never quote a gate-3 number from a fast run.

---

## 3. The work

### Step 1 — free the GPU slots (needs your approval)

Twelve jobs run now. I must stop some jobs to start the new arms.

I recommend that you stop these four jobs:

| Job | Reason to stop |
|---|---|
| `dfx_bot_o` | Alternating side, historical reward. Section 2a shows this arm cannot pass gate 1. It has run since 7 September. |
| `eg_bot_f` | Alternating side, historical reward, fast. Same reason. n = 50 is already enough to show the failure. |
| `rw_b` | Action bonus only, 4 actions. Superseded by the `pnl_scaled` arms. |
| `ctl_f` | Gate-3 control in the fast regime. Section 2g shows a fast gate-3 number is not usable. |

I recommend that you keep these:

* `ps_as`, `ps_af`, `ps_ff`, `ps_nc`, `ps_fs`, `ps_ls` — the gate-2 evidence.
* `eg_bot_o` — the only normal-regime historical-reward arm.
* `ctl_n` — a seed-paired gate-3 control in the normal regime.

**Killing a job destroys its `inventordists_*` data.** The per-episode JSON
survives. I will copy every JSON file to the repo before I stop anything.

### Step 2 — wave 1, the fast screen (4 new arms, about 22 h each)

All four arms use one fixed side. All four use `EXP_APPROX=true`.

| Label | Side | Reward | Action space | TWAP cycle | Question it answers |
|---|---|---|---|---|---|
| `s1_buy_f` | buy | PnL scale | full (0) | 2 on / 1 off | Does a single-side agent front-run with the new reward? |
| `s1_sell_f` | sell | PnL scale | full (0) | 2 on / 1 off | Same, sell side. |
| `s1_sell_on_f` | sell | PnL scale | full (0) | always on | Does the on/off cycle hurt gate 1? |
| `s1_sell_leg_f` | sell | historical | 4 actions (1) | 2 on / 1 off | Does the paper's arm still work after the simulator fixes? |

`s1_sell_leg_f` is the control. It reproduces the paper's successful arm. If it
fails, a simulator fix broke the result and I must find which one.

### Step 3 — extend the analysis tool (while wave 1 runs)

`criteria_report.py` refuses to score gate 1 on a single-side run. This refusal
is too strong. A single-side run with on/off episodes gives a valid measure.

I will add the **present-absent contrast**:

```
contrast = mean(in-window inventory | TWAP present)
         - mean(same-window inventory | TWAP absent)
```

The two arms use matched seeds. A constant directional bias is in both arms, so
it cancels. This is the measure the paper uses. The paper reports +15.1 against
+10.9 for the buy agent.

I will also add the t = 100 benchmark from section 2f.

I will not change the side contrast. It stays for alternating runs.

### Step 4 — wave 2, the normal regime (2 arms, about 5 days each)

I will take the two best arms from wave 1. I will run them in the normal regime.
These runs decide gates 1 and 2.

### Step 5 — the gate-3 test

I will evaluate the wave-2 checkpoints with `AR_RL_runner.py`. I will use seeds
5000 and above, because seeds 1000 and above are training seeds.

I will compare against the matched baseline from `twap_baseline_obs.py`. I will
use a Welch t-test and a Mann-Whitney U test, as the paper does.

### Step 6 — report

I will give one table. The table has one row for each arm and one column for
each gate. Each cell has a value, a CI and PASS or fail.

---

## 4. Cluster rules

I will copy the directive block from a trainer that works:

```
#$ -l h_rt=<hours>:00:00
#$ -l tmem=128G
#$ -l gpu=true
#$ -j y
#$ -N <name>
#$ -cwd
source /share/apps/source_files/python/python-3.9.5.source
source ~/myenv/bin/activate
python3 /home/ajafree/lobSimulations/HawkesRLTrading/AR_RL_Trainer.py
```

* I will not set `h_vmem`. Every script on the cluster that sets `h_vmem` is a
  probe job, and those are the jobs that crashed. `h_vmem` limits *virtual*
  memory. NumPy and torch reserve a large virtual address space at import. A
  small `h_vmem` kills a job that uses little real memory.
* I will raise `tmem` from 64G to 128G.
* I will run a short smoke job first. I will read its `.o` file. I will submit
  the full job only after the smoke job prints the config banner and finishes
  one episode.
* I will run all training on the cluster. I will run only analysis on the Mac.

---

## 5. The review agent

You asked for a review agent. I will start one when I begin step 2.

Its task:

1. Check the claims in section 2 against the raw JSON files.
2. Check each new `trainer.sh` before I submit it.
3. Check the new metric in `criteria_report.py` against a hand calculation.
4. Report any gate that I call PASS without a CI that excludes zero.

I will report what it finds, including findings against my own work.

---

## 6. What I will not do

* I will not make the meta-order larger or faster. Section 2a of the handoff
  shows the signal is above the noise floor.
* I will not relax any gate.
* I will not restart the queue-depletion measurement. The handoff abandoned it
  for good reasons.
* I will not change the reward attribution on breach steps. The supervisor has
  marked that as by design.
* I will not quote a gate-3 number from a fast run.

---

## 7. Progress log

### 2026-09-14 — steps 1 to 3

**Step 1 done.** I stopped `dfx_bot_o`, `eg_bot_f` and `rw_b`. `ctl_f` had
already finished by itself. I copied 2.1 GB of logs to
`~/killed_backup_20260914/` on the cluster first. I copied the per-episode
metrics of all 16 runs into `HawkesRLTrading/episode_metrics_snapshots/20260914/`.

**The gate-3 baseline is now in git.** Commit `208dfb6`. See section 1.

**A blocker was found and corrected before the full jobs started.**
`inv_level_in_window` is `None` on every TWAP-absent episode, in every run that
exists. I measured this: 0 of 22 absent episodes in `ps_no_cem_fast` carry it.
The cause is in `AR_RL_Trainer.py`. The list `inventory_with_twap_<side>` is
keyed on the meta-order being present. On an absent episode nothing fills it.

This makes the present-absent contrast impossible. That contrast is step 3 of
this plan. It is also the only gate-1 measure that a single-side run can use.

The correction is commit `093bbfa`. A new list `inventory_window_clock` records
the window (250,400) on every episode. It is keyed on the clock only. Nothing
that exists changes meaning, because the new list is additive. On a present
episode the new list holds the same samples as the old one. The trainer now
asserts this on every present episode.

I tested the new measure on synthetic data. The test injects a front-running
response of +5.0 and a per-run directional bias of +20.0:

| Measure | Result |
|---|---|
| Old LEVEL metric | +25.00 (wrong: it contains the bias) |
| New present-absent contrast | **+5.00 +- 0.16 PASS** (correct) |

The hand calculation and the tool agree to 3 decimal places.

**A second error was found by the smoke test.** The first four smoke jobs failed
after 5 seconds with `ModuleNotFoundError: No module named 'gymnasium'`. I had
copied the environment block from the `pnl_scaled` scripts. Those scripts do not
activate the virtual environment. The `explo_gae` scripts do. The correct order
is:

```
source /share/apps/source_files/python/python-3.9.5.source
source ~/myenv/bin/activate
```

The first line must come first. The virtual environment's Python cannot load
`libpython3.9.so.1.0` without it. All eight scripts now have both lines. The
smoke test caught this at a cost of 5 seconds of compute. Without it, four
168-hour jobs would have failed the same way.

**Step 2 in progress.** The four arms are at
`~/LSTM_fRL/wout_expo/new_value_function/single_side/`. Each arm has an
`env.sh`. Both `smoke.sh` and `trainer.sh` read it, so a smoke test validates
the exact configuration of the full job.

**Step 3 done.** Commit `093bbfa`.

### New result from the live runs

`ps_no_cem_fast` has reached 67 episodes. It now **passes gate 2**:

| Arm | n absent | Solo PnL | 95% CI | Verdict |
|---|---|---|---|---|
| `ps_no_cem_fast` | 22 | **+0.45** | [+0.01, +0.89] | **PASS** |

This is the first time any arm has passed gate 2. The arm uses the PnL-scaled
reward. It still fails gate 1 (side contrast -4.27). This supports section 2e:
the two good ideas are in different runs, and no run has both.

---

## 8. Correction to section 2c — the real mechanism

The review agent found an error in my section 2c. I confirmed it in the code.
Section 2c stays in this document as written, because a record of a wrong
explanation is more useful than a silent edit. It is wrong. This section
replaces it.

### What I said

I said the agent has one weak signal for the side, and that one LSTM cannot
learn two opposite responses from one scalar.

### What is true

The signal is not weak in the window. It does not exist before the window.

`AR_RL_Trainer.py:668-672` sets `TWAPPresent` to -1 or +1 **only** when the
clock is inside `[twap_start_time, twap_end_time]` = [250, 400]. Outside that
interval it sets `TWAPPresent = 0`. `AR_RL_runner.py:404-409` does the same.
`ICRLAgent.py:2739` appends `TWAPPresent` to the observation as one scalar.

So between t = 100 and t = 250 the feature is 0. It is 0 on buy episodes. It is
0 on sell episodes. It is 0 on TWAP-absent episodes. The three cases are
identical to the policy.

**No agent in this simulator can position itself before t = 250 in a way that
depends on the meta-order.** The information is not there. The meta-order has
not traded, so no other feature carries it either.

There is a feature that would help, and it is not used. `_twap_phase()` at
`ICRLAgent.py:1662-1666` returns -1 before the meta-order, 0 during, and +1
after. It is used only by `_exploration_key_state()` at `ICRLAgent.py:1704-1708`,
which keys the exploration bonus. It never reaches the policy observation.

Note that `_twap_phase()` would not leak the future. It becomes +1 only after
`_twap_seen` is set, and `_twap_seen` is set only when `TWAPPresent` is non-zero.
Before the window it returns -1 on every episode, present or absent.

### What this means for gate 1

Gate 1 says "TWAP buying, agent goes long **ahead of it**". Two readings:

1. **Before the meta-order starts.** This is impossible here, for any agent and
   any reward. An agent can only appear to do it by carrying a fixed
   directional bias in its weights. A single-side agent does exactly that,
   because its side never changes. That is not detection.
2. **Ahead of the remaining child orders, inside the window.** The meta-order
   runs for 150 seconds and trades throughout. After the first child orders the
   side is visible, both through `TWAPPresent` and through the order flow. An
   agent that then positions itself takes the rest of the meta-order's impact.

Reading 2 is the only one that is achievable, and it is what the paper measures.
It is also normal usage: a high-frequency trader detects a meta-order from order
flow after it starts, and trades ahead of its remainder.

**This needs a decision from the user, because gate 1 belongs to the user.**
I recommend reading 2. I will not change the gate without an answer.

The measures in this plan are correct under reading 2, and no change is needed:

* Response against the PRE window = in-window level minus pre-window level.
  Under reading 2 this is exactly the quantity of interest.
* The present-absent contrast compares the same window with and without the
  meta-order. Under reading 2 this is exactly the quantity of interest.

### What this does NOT change

Wave 1 is still the right experiment. The reviewer says wave 1 does not isolate
side alternation, because the paper's runs differ in seeds, presence cycle,
action space, reward, and simulator version.

That is true of the paper. It is not true of my comparator. **I compare wave 1
against `ps_ff`, which is running now**, not against the paper:

| Setting | `ps_ff` (running) | `s1_sell_f` (new) |
|---|---|---|
| reward | PnL scale | PnL scale |
| action space | 0 | 0 |
| symmetric MO gating | true | true |
| expApprox | true | true |
| TWAP cycle | 2 on / 1 off | 2 on / 1 off |
| seeds | vary, base 1000 | vary, base 1000 |
| simulator | current | current |
| **TWAP side mode** | **alternate** | **sell** |

One variable differs. I have asked the reviewer to check this against the
`ps_full_fast` metrics header and confirm.

### An honest limit on the evidence

The reviewer is right about one thing that I must record. Every one of the 16
metrics files has `twap_side_mode = "alternate"`. There is no variance in the
variable at all. So section 2c was never supported by that data, whatever the
mechanism. Wave 1 creates the variance. Until it lands, "side alternation
prevents front-running" is a **hypothesis**, not a finding.
