"""Tests for TWAP-absent ("alternating") episodes in AR_RL_Trainer.

Two things must hold or the alternating design silently corrupts its own
metrics:

1. `off_time=0` really does neuter the TWAP agent -- it must never trade, at
   any time, on either side, including the t=0 boundary where the `>` in
   `current_time > off_time` does not yet fire.
2. Metrics must be bucketed on ACTUAL TWAP presence, not on the clock window.
   On an absent episode the [twap_start, twap_end] window is ordinary
   market-making time; filing it as "with TWAP" contaminates exactly the
   comparison the design exists to make.
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from HawkesRLTrading.src.SimulationEntities.MetaOrderTradingAgents import (  # noqa: E402
    TWAPGymTradingAgent,
)

TWAP_START, TWAP_END = 250, 400


def _twap(side, lag, off_time):
    return TWAPGymTradingAgent(
        seed=1, log_events=False, log_to_file=False, strategy="TWAP",
        Inventory={"INTC": 500}, cash=1000000, cashlimit=10 ** 11, action_freq=1,
        total_order_size=150, total_time=150, window_size=25, side=side,
        order_target="INTC", off_time=off_time, start_trading_lag=lag)


def _obs():
    return {"Inventory": 500, "Positions": {"Ask_L1": [], "Bid_L1": [],
                                            "Ask_L2": [], "Bid_L2": []}}


def test_off_time_zero_never_trades():
    """The neutering must be total, including at exactly t=0."""
    for side in ("sell", "buy"):
        for lag in (0, TWAP_START):
            a = _twap(side, lag, off_time=0)
            for t in (0, 0.5, 1, 2, 100, 250, 300, 400, 549):
                a.current_time = t
                act = a.get_action(_obs())
                assert act == (12, 0), f"side={side} lag={lag} t={t} -> {act}"


def test_off_time_nonzero_does_trade():
    """Control: the same agent with a live off_time must actually act,
    otherwise the test above passes for the wrong reason."""
    a = _twap("sell", TWAP_START, off_time=400)
    acted = False
    for t in range(TWAP_START + 1, TWAP_END):
        a.current_time = t
        if a.get_action(_obs()) != (12, 0):
            acted = True
            break
    assert acted, "TWAP with off_time=400 never acted; the neutering test is vacuous"


# --- bucketing predicate -----------------------------------------------------

def _with_twap(twap_present, t):
    """The predicate as it now stands in AR_RL_Trainer."""
    return twap_present and (TWAP_END >= t >= TWAP_START)


def test_absent_episode_never_buckets_as_with_twap():
    for t in (0, 100, 250, 300, 400, 549):
        assert not _with_twap(False, t), f"t={t} filed as with-TWAP on an absent episode"


def test_present_episode_buckets_on_the_window():
    assert _with_twap(True, 300)
    assert _with_twap(True, TWAP_START)
    assert _with_twap(True, TWAP_END)
    assert not _with_twap(True, TWAP_START - 1)
    assert not _with_twap(True, TWAP_END + 1)


# --- presence cycle and side alternation -------------------------------------

def _present(on, off, ep):
    """The presence rule as it now stands in AR_RL_Trainer."""
    return (ep % (on + off)) < on


def test_presence_cycle():
    assert [_present(1, 0, e) for e in range(6)] == [True] * 6, \
        "1/0 must preserve today's behaviour: present every episode"
    assert [_present(1, 1, e) for e in range(6)] == [True, False] * 3
    assert [_present(2, 1, e) for e in range(9)] == \
        [True, True, False, True, True, False, True, True, False]


def test_two_on_one_off_keeps_two_thirds_exposure():
    n = 160
    present = sum(_present(2, 1, e) for e in range(n))
    assert present == 107, present
    assert abs(present / n - 2 / 3) < 0.01


def _sides(on, off, n):
    """Side alternation must index on TWAP-PRESENT episodes, not all episodes."""
    out, k = [], 0
    for e in range(n):
        p = _present(on, off, e)
        out.append(("buy" if k % 2 == 0 else "sell") if p else "none")
        if p:
            k += 1
    return out


def test_alternate_side_splits_evenly_under_a_presence_cycle():
    s = _sides(2, 1, 9)
    assert s == ["buy", "sell", "none", "buy", "sell", "none", "buy", "sell", "none"], s
    for on, off, n in ((1, 0, 120), (2, 1, 160), (1, 1, 80)):
        got = _sides(on, off, n)
        b, sl = got.count("buy"), got.count("sell")
        assert abs(b - sl) <= 1, f"on={on} off={off}: {b} buy vs {sl} sell"


def test_side_counter_must_index_on_present_episodes_only():
    """Indexing side off the raw episode number is only safe for some cycles.

    At 1-on-1-off every present episode is even, so a naive `episode % 2` would
    make the entire run buy-side -- silently producing a single-sided model from
    a config that asked for both. (At 2-on-1-off the parities happen to balance,
    which is exactly why this has to be tested at 1/1 and not assumed.)
    """
    naive = [("buy" if e % 2 == 0 else "sell") if _present(1, 1, e) else "none"
             for e in range(80)]
    assert naive.count("sell") == 0, "expected the naive counter to be fully single-sided"

    fixed = _sides(1, 1, 80)
    assert abs(fixed.count("buy") - fixed.count("sell")) <= 1, fixed[:10]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
    print("all tests passed")
