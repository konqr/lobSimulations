"""Pin CEM's elite selection across the THREE TWAP regimes.

`cem_full_episode=True` (what the trainer passes), so the live selector is
top-3-per-regime whole episodes ranked by total episode reward; the
contiguous-subarray branch is dead code in every run to date.

The defect fixed here: the selector pooled only `episode_sides > 0` and
`< 0`. On a TWAP-ABSENT episode `TWAPPresent` is pinned to 0 throughout, so
`episode_sides[ep] == 0` and the episode fell into neither pool. In an
alternating run both buy and sell pools are always non-empty, so the global
top-5 fallback never fired either -- absent episodes could never be elites.
Self-imitation therefore never reinforced a single standalone market-making
episode, in precisely the on/off runs that exist to test criterion 2.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))

from HawkesRLTrading.src.SimulationEntities.ICRLAgent import PPOAgent

K = 10


class _Stub(PPOAgent):
    """Only the elite-selection branch is exercised."""

    def __init__(self, buffer, sides, full_episode=True, floor=0.0, n_elites=None):
        self.trajectory_buffer = buffer
        self.episode_sides = sides
        self.cem_full_episode = full_episode
        self.cem_elite_floor = floor
        self.cem_n_elites = n_elites


def _episode(ep, total, n=K + 5):
    """n transitions whose rewards sum to `total`."""
    r = total / n
    return [(ep, (None, 0, 0, r, None, 0)) for _ in range(n)]


def _build(spec):
    """spec: {ep: (side, total)} -> (buffer, sides)"""
    buf, sides = [], {}
    for ep, (side, total) in spec.items():
        buf.extend(_episode(ep, total))
        sides[ep] = side
    return buf, sides


def _elites(spec, floor=0.0, n_elites=None):
    buf, sides = _build(spec)
    res = _Stub(buf, sides, floor=floor, n_elites=n_elites).get_max_contiguous_rewards(K=K)
    return {ep for ep, v in res.items() if v is not None}


def test_absent_episodes_can_now_be_elites():
    """The regression. Absent episodes score highest here, and before the fix
    they were unreachable because both signed pools were non-empty."""
    spec = {0: (1, 1.0), 1: (-1, 1.0), 2: (0, 99.0),
            3: (1, 0.5), 4: (-1, 0.5), 5: (0, 98.0)}
    assert {2, 5} <= _elites(spec), _elites(spec)


def test_each_regime_gets_its_own_elites():
    spec = {}
    for i in range(5):
        spec[10 + i] = (1, 100.0 + i)     # buy, high totals
        spec[20 + i] = (-1, 50.0 + i)     # sell, middling
        spec[30 + i] = (0, 1.0 + i)       # absent, low totals
    e = _elites(spec)
    assert len({x for x in e if 10 <= x < 20}) == 2, e
    assert len({x for x in e if 20 <= x < 30}) == 2, e
    assert len({x for x in e if 30 <= x < 40}) == 2, e


def test_default_reproduces_the_legacy_elite_counts():
    """The DEFAULT must not silently change any arm. Legacy was 6 for a
    multi-regime buffer (top-3 of each of two pools) and 5 for a single-regime
    one (global top-5). A flat default of 6 would have raised every
    single-regime arm -- including the default configuration and
    buy_base/sell_base -- from 5 to 6."""
    three = {}
    for i in range(8):
        three[10 + i] = (1, 100.0 + i)
        three[20 + i] = (-1, 50.0 + i)
        three[30 + i] = (0, 1.0 + i)
    two = {k: v for k, v in three.items() if v[0] != 0}
    one = {k: v for k, v in three.items() if v[0] == 1}
    assert len(_elites(three)) == 6, _elites(three)
    assert len(_elites(two)) == 6, _elites(two)
    assert len(_elites(one)) == 5, _elites(one)


def test_explicit_total_is_constant_across_regime_counts():
    """With CEM_N_ELITES set, the count must NOT vary with the regime count --
    otherwise the elite FRACTION of the ~40 buffered episodes, i.e. how hard
    CEM pulls, is confounded with the arm under test."""
    three = {}
    for i in range(8):
        three[10 + i] = (1, 100.0 + i)
        three[20 + i] = (-1, 50.0 + i)
        three[30 + i] = (0, 1.0 + i)
    two = {k: v for k, v in three.items() if v[0] != 0}
    one = {k: v for k, v in three.items() if v[0] == 1}
    assert len(_elites(three, n_elites=6)) == 6
    assert len(_elites(two, n_elites=6)) == 6
    assert len(_elites(one, n_elites=6)) == 6


def test_elite_total_is_tunable():
    spec = {}
    for i in range(5):
        spec[10 + i] = (1, 100.0 + i)
        spec[20 + i] = (-1, 50.0 + i)
        spec[30 + i] = (0, 1.0 + i)
    assert len(_elites(spec, n_elites=3)) == 3
    assert len(_elites(spec, n_elites=9)) == 9


def test_remainder_goes_to_the_largest_pools():
    """With a total not divisible by the pool count, a small pool must not
    silently shrink the total below the target."""
    spec = {10 + i: (1, 100.0 + i) for i in range(5)}
    spec.update({20 + i: (-1, 50.0 + i) for i in range(5)})
    spec[30] = (0, 1.0)                       # absent pool has ONE episode
    e = _elites(spec, n_elites=6)
    assert len(e) == 6, e
    assert 30 in e, "the single absent episode must still be an elite"


def test_a_high_scoring_regime_cannot_monopolise_the_elite_set():
    """The reason to balance at all: absent episodes have no TWAP to trade
    against, so their totals are not comparable with present ones."""
    spec = {i: (1, 1000.0 + i) for i in range(6)}
    spec.update({10 + i: (0, 1.0 + i) for i in range(6)})
    e = _elites(spec)
    assert any(x >= 10 for x in e), "absent regime starved by a higher-scoring one"


def test_single_regime_falls_back_to_global_top_n():
    spec = {i: (1, float(i)) for i in range(9)}
    assert _elites(spec) == {8, 7, 6, 5, 4}   # legacy single-regime top-5


def test_untagged_buffer_falls_back_to_global_top_n():
    spec = {i: (0, float(i)) for i in range(9)}
    assert _elites(spec) == {8, 7, 6, 5, 4}   # legacy single-regime top-5


def test_short_episodes_are_ignored():
    buf, sides = _build({0: (1, 5.0), 1: (-1, 5.0)})
    buf.extend([(2, (None, 0, 0, 1e6, None, 0))] * (K - 1))   # too short
    sides[2] = 0
    res = _Stub(buf, sides).get_max_contiguous_rewards(K=K)
    assert res.get(2) is None, "an episode shorter than K must not be an elite"


def test_elite_floor_off_by_default_returns_elites_however_bad():
    """Status quo: sorted(...)[:3] takes three episodes regardless of quality."""
    spec = {i: (1, 0.0) for i in range(5)}
    spec.update({10 + i: (-1, 0.0) for i in range(5)})
    assert len(_elites(spec, floor=0.0)) > 0


def test_elite_floor_rejects_a_flat_buffer():
    """With every episode identical there is no signal, so a floor of any size
    must take nothing -- the guard against CEM imitating pure luck."""
    spec = {i: (1, 1.0) for i in range(5)}
    spec.update({10 + i: (-1, 1.0) for i in range(5)})
    assert _elites(spec, floor=1.0) == set()


def test_elite_floor_keeps_genuine_outliers():
    spec = {i: (1, 0.0) for i in range(9)}
    spec[100] = (1, 50.0)
    spec.update({10 + i: (-1, 0.0) for i in range(9)})
    spec[200] = (-1, 50.0)
    e = _elites(spec, floor=1.0)
    assert 100 in e and 200 in e, e
    assert len(e) == 2, e


def test_ranking_is_by_total_episode_reward():
    """Which makes the reward's term magnitudes the elite criterion. With
    terminal_invpenalty=25 a single unit of terminal inventory costs -25
    against an episode PnL of +-0.2, so elites become 'ended closest to flat'
    unless the terminal penalty is scaled down with the action bonus."""
    # enough episodes that the elite set is a strict subset, or "excluded"
    # is vacuous
    spec = {i: (1, float(i)) for i in range(6)}          # buy totals 0..5
    spec.update({10 + i: (-1, float(i)) for i in range(6)})  # sell totals 0..5
    e = _elites(spec)
    assert {5, 4, 15, 14} <= e, e            # the best of each side
    assert 0 not in e and 10 not in e, e     # the worst of each side


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
