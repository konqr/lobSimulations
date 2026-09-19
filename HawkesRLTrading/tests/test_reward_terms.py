"""Pin the reward function's term magnitudes in PPOAgent.calculaterewards.

Why this test exists. Measured on the six 2026-09-07 runs:

  per-step PnL change  median EXACTLY 0 (85-88% of steps move no money)
                       mean |dW| 0.006-0.018 dollars
  episode terminal PnL +-0.2 dollars over ~2,480 steps
  flat action bonus    0.5 per acted step, deterministic and always positive
  exploration bonus    0.2 first visit, then lambda/sqrt(N)

So the two shaping terms are 30-90x the typical PnL signal per step and worth
hundreds per episode against a terminal PnL of +-0.2. The objective actually
being maximised is "act often, visit novel states". These tests pin the term
structure so that ratio cannot drift silently again, and pin that the new
knobs default to the historical behaviour.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))

from HawkesRLTrading.src.SimulationEntities.ICRLAgent import PPOAgent


class _Stub(PPOAgent):
    """calculaterewards touches only a handful of attributes; bypass __init__
    (which builds torch networks and an exchange connection) and set them."""

    def __init__(self, inv, mid, cash, prev_inv, prev_mid, prev_cash,
                 action_bonus=0.5, running_invpenalty=0.0, last_action=12,
                 terminal_invpenalty=0.0, istruncated=False):
        self.Inventory = {"INTC": inv}
        self.mid = mid
        self.cash = cash
        self.current_time = 1.0
        self.positions = {}
        self.profit = 0.0
        self.transaction_cost = 1e-4
        self.rewardpenalty = 5
        self.istruncated = istruncated
        self.last_action = last_action
        self.terminal_invpenalty = terminal_invpenalty
        self.last_state = None          # skips the state-dependent shaping branches
        self.alt_state = True
        self.two_sided_reward = False
        self.exploration_bonus = False
        self.action_bonus = action_bonus
        self.running_invpenalty = running_invpenalty
        # statelog entries: (time, cash, profit, inventory, positions, mid)
        self.statelog = [(0.0, prev_cash, 0.0, {"INTC": prev_inv}, {}, prev_mid)]


def _reward(**kw):
    return _Stub(**kw).calculaterewards(termination=kw.pop("_term", False))


def test_reward_is_the_mtm_wealth_change_when_no_op():
    """Action 12 is the no-op, so no shaping applies and the reward must be
    exactly the change in cash plus the change in marked inventory value."""
    r = _Stub(inv=10, mid=100.01, cash=1000.0,
              prev_inv=10, prev_mid=100.00, prev_cash=1000.0).calculaterewards(False)
    tc = 1e-4
    expected = 10 * 100.01 * (1 - tc) - 10 * 100.00 * (1 - tc)
    assert abs(r - expected) < 1e-9, (r, expected)


def test_action_bonus_default_is_the_historical_half_dollar():
    """Default must reproduce every run to date, or new runs are not
    comparable with the six on the cluster."""
    kw = dict(inv=0, mid=100.0, cash=1000.0, prev_inv=0, prev_mid=100.0, prev_cash=1000.0)
    noop = _Stub(last_action=12, **kw).calculaterewards(False)
    acted = _Stub(last_action=3, **kw).calculaterewards(False)
    assert abs((acted - noop) - 0.5) < 1e-12, (acted, noop)


def test_action_bonus_dwarfs_a_typical_step_of_pnl():
    """The measured mean |dW| per step is 0.0087 on buy_base. A one-tick move
    on a single share is 0.01. The default bonus must be shown to be ~50x that
    -- this is the whole reason criterion 2 has never been met."""
    tick_pnl = _Stub(inv=1, mid=100.01, cash=1000.0,
                     prev_inv=1, prev_mid=100.00, prev_cash=1000.0,
                     last_action=12).calculaterewards(False)
    assert 0.0 < tick_pnl < 0.011, tick_pnl
    assert 0.5 / tick_pnl > 45, "action bonus is no longer dominant; update the analysis"


def test_action_bonus_is_tunable_to_the_pnl_scale():
    kw = dict(inv=0, mid=100.0, cash=1000.0, prev_inv=0, prev_mid=100.0, prev_cash=1000.0)
    for b in (0.0, 1e-3, 0.5):
        noop = _Stub(last_action=12, action_bonus=b, **kw).calculaterewards(False)
        acted = _Stub(last_action=3, action_bonus=b, **kw).calculaterewards(False)
        assert abs((acted - noop) - b) < 1e-12, (b, acted, noop)


def test_running_inventory_penalty_defaults_off():
    """Status quo: the running penalty is commented out, so the ONLY inventory
    control is a terminal penalty ~2,480 steps away."""
    kw = dict(mid=100.0, cash=1000.0, prev_mid=100.0, prev_cash=1000.0, last_action=12)
    flat = _Stub(inv=0, prev_inv=0, **kw).calculaterewards(False)
    heavy = _Stub(inv=20, prev_inv=20, **kw).calculaterewards(False)
    assert abs(heavy - flat) < 1e-9, (flat, heavy)


def test_running_inventory_penalty_is_quadratic_and_signed_correctly():
    """Holding inventory must REDUCE reward, and do so as inv**2."""
    kw = dict(mid=100.0, cash=1000.0, prev_mid=100.0, prev_cash=1000.0,
              last_action=12, running_invpenalty=1e-4)
    r0 = _Stub(inv=0, prev_inv=0, **kw).calculaterewards(False)
    r10 = _Stub(inv=10, prev_inv=10, **kw).calculaterewards(False)
    r20 = _Stub(inv=20, prev_inv=20, **kw).calculaterewards(False)
    assert r10 < r0 and r20 < r10, (r0, r10, r20)
    assert abs((r0 - r10) - 1e-4 * 100) < 1e-9
    assert abs((r0 - r20) - 1e-4 * 400) < 1e-9
    # and it must be symmetric in sign -- a short is as penalised as a long,
    # or the penalty itself would induce the directional bias it exists to stop
    rm10 = _Stub(inv=-10, prev_inv=-10, **kw).calculaterewards(False)
    assert abs(rm10 - r10) < 1e-9, (r10, rm10)


def test_terminal_inventory_penalty_only_fires_on_termination():
    kw = dict(inv=10, mid=100.0, cash=1000.0, prev_inv=10, prev_mid=100.0,
              prev_cash=1000.0, last_action=12, terminal_invpenalty=25.0)
    mid_ep = _Stub(**kw).calculaterewards(False)
    end_ep = _Stub(**kw).calculaterewards(True)
    assert abs((mid_ep - end_ep) - 25.0 * 100) < 1e-9, (mid_ep, end_ep)


def test_truncation_penalty_is_a_hundred():
    kw = dict(inv=0, mid=100.0, cash=1000.0, prev_inv=0, prev_mid=100.0,
              prev_cash=1000.0, last_action=12)
    assert abs(_Stub(istruncated=False, **kw).calculaterewards(False)
               - _Stub(istruncated=True, **kw).calculaterewards(False) - 100.0) < 1e-9


def test_no_running_inventory_penalty_is_hidden_in_the_source():
    """The old line is commented out. If someone uncomments it, eta=5 makes the
    penalty 5*inv**2 -- 500 per step at inv=10, which is 50,000x the typical
    PnL step and would dominate even harder than the action bonus does."""
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                            "src", "SimulationEntities", "ICRLAgent.py")).read()
    assert "# penalty = self.rewardpenalty * (self.countInventory()**2)" in src, \
        "the commented-out running penalty was changed; re-derive its scale before enabling it"


if __name__ == "__main__":
    mod = sys.modules[__name__]
    failures = 0
    for name in sorted(dir(mod)):
        if not name.startswith("test_"):
            continue
        try:
            getattr(mod, name)()
            print("PASS %s" % name)
        except AssertionError as e:
            failures += 1
            print("FAIL %s: %s" % (name, e))
    print("all tests passed" if not failures else "%d FAILURES" % failures)
    sys.exit(1 if failures else 0)
