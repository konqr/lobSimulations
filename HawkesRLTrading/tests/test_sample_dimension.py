"""Regression test for the thinning dimension-assignment bias.

The simulator used to reuse the thinning variate `D*lamb_bar` to choose which
of the 12 dimensions fired. That is valid only while `lamb_bar` upper-bounds
the realised intensity. Inhibitory kernels break the bound, and when it breaks
the reused variate is truncated below sum(decays); the walk starts at index 0,
so the tail becomes unreachable and low indices are over-sampled. `cols` is
Ask-first, so the bias was systematically anti-Bid.

These tests fail against the legacy rule and pass against sample_dimension().
"""
import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from HawkesRLTrading.src.Stochastic_Processes.Arrival_Models import sample_dimension  # noqa: E402


def legacy_assign(decays, lamb_bar):
    """The rule as it stood before the fix, kept only so the bias is pinned."""
    D = np.random.uniform(0, 1)
    if D * lamb_bar > float(np.sum(decays)):
        return None                      # candidate rejected
    d = np.asarray(decays, dtype=float).reshape(-1)
    k = 0
    total = d[0]
    while k < len(d) - 1 and D * lamb_bar >= total:
        k += 1
        total += d[k]
    return k


def empirical(sampler, decays, n=200000):
    counts = np.zeros(len(decays))
    for _ in range(n):
        k = sampler()
        if k is not None:
            counts[k] += 1
    return counts / counts.sum()


def test_fixed_rule_is_proportional_when_bound_is_violated():
    """The whole point: correct even when lamb_bar is not an upper bound."""
    np.random.seed(0)
    decays = np.array([3.0, 2.5, 2.0, 1.5, 1.0, 0.5,
                       0.5, 1.0, 1.5, 2.0, 2.5, 3.0]).reshape(12, 1)
    lamb = float(np.sum(decays))                 # 21.0
    lamb_bar = 0.80 * lamb                       # bound VIOLATED by 25%
    want = decays.reshape(-1) / lamb

    got = empirical(lambda: sample_dimension(decays, lamb), decays)
    assert np.abs(got - want).max() < 0.005, f"max dev {np.abs(got-want).max()}"

    # and the reused-variate rule is demonstrably not proportional
    np.random.seed(0)
    bad = empirical(lambda: legacy_assign(decays, lamb_bar), decays)
    assert bad[-1] == 0.0, "legacy rule should never reach the tail dimension"
    assert bad[0] > want[0] * 1.15, "legacy rule should over-sample index 0"


def test_ask_bid_bias_direction_matches_observed():
    """Legacy rule favours Ask (indices 0-5); fixed rule does not."""
    np.random.seed(1)
    decays = np.full((12, 1), 1.0)               # perfectly symmetric intensity
    lamb = float(np.sum(decays))
    lamb_bar = 0.90 * lamb

    got = empirical(lambda: sample_dimension(decays, lamb), decays)
    assert abs(got[:6].sum() - got[6:].sum()) < 0.01, "fixed rule must be side-neutral"

    np.random.seed(1)
    bad = empirical(lambda: legacy_assign(decays, lamb_bar), decays)
    assert bad[:6].sum() - bad[6:].sum() > 0.05, "legacy rule must show the Ask excess"


def test_fixed_rule_matches_legacy_when_bound_holds():
    """No behaviour change in the regime where the old trick was valid."""
    np.random.seed(2)
    decays = np.array([3.0, 2.5, 2.0, 1.5, 1.0, 0.5,
                       0.5, 1.0, 1.5, 2.0, 2.5, 3.0]).reshape(12, 1)
    lamb = float(np.sum(decays))
    want = decays.reshape(-1) / lamb

    got = empirical(lambda: sample_dimension(decays, lamb), decays)
    np.random.seed(2)
    old = empirical(lambda: legacy_assign(decays, 1.30 * lamb), decays)   # valid bound
    assert np.abs(got - want).max() < 0.005
    assert np.abs(old - want).max() < 0.005


def test_never_overruns_on_float_error():
    decays = np.full((12, 1), 1.0 / 3.0)
    lamb = float(np.sum(decays))
    for _ in range(10000):
        k = sample_dimension(decays, lamb)
        assert 0 <= k <= 11




# ---------------------------------------------------------------------------
# Regression test for the stale `self.left` purge bug (Arrival_Models.py:374).
# ---------------------------------------------------------------------------

def _window_after_purge(reset_left):
    """Replay the purge/advance logic; returns the kernel window on the step
    AFTER a purge fires. `reset_left=True` is the fixed behaviour."""
    TAU = 500
    ts = [(float(i), 0) for i in range(601)]
    left = 0
    s = 600.0
    while left < len(ts):
        if s - ts[left][0] >= TAU:
            left += 1
        else:
            break
    if ts[-1][0] - ts[0][0] > TAU:
        ts = ts[left:]
        if reset_left:
            left = 0
    s = 601.0
    while left < len(ts):
        if s - ts[left][0] >= TAU:
            left += 1
        else:
            break
    return ts[left:], [p for p in ts if s - p[0] < TAU]


def test_purge_resets_left_so_no_in_window_points_are_dropped():
    got, want = _window_after_purge(reset_left=True)
    assert got == want, f"fixed logic dropped {len(want)-len(got)} in-window points"


def test_stale_left_bug_is_pinned():
    """Without the reset the window silently loses points that are still valid."""
    got, want = _window_after_purge(reset_left=False)
    assert len(got) < len(want), "expected the legacy logic to drop points"
    assert len(want) - len(got) == 100


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
    print("all tests passed")
