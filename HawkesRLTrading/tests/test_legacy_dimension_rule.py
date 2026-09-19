"""Pin the HAWKES_LEGACY_DIMENSION_RULE control switch.

The drift study's own hardest-won lesson was that a buggy-code control arm is
mandatory: its round-1 null would have read as "the market is symmetric" and
been completely wrong. This switch lets the buggy arm be run from the CURRENT
code at matched seeds rather than by cloning commit 002bbf3, so the two arms
cannot differ in anything except the assignment rule.

Two things must hold and both are load-bearing:

  * With the switch OFF (the default, and what 16 running jobs use) the code
    path must be byte-for-byte the post-6e94941 behaviour, including the number
    of RNG draws -- otherwise enabling this switch silently perturbs every
    production run.
  * With it ON, the rule must reproduce the pre-6e94941 code exactly: reuse the
    thinning variate, walk with NO len-1 cap, and draw no random number.
"""
import importlib
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))

import numpy as np

MOD = "HawkesRLTrading.src.Stochastic_Processes.Arrival_Models"


def _load(legacy):
    saved = os.environ.get("HAWKES_LEGACY_DIMENSION_RULE")
    os.environ["HAWKES_LEGACY_DIMENSION_RULE"] = "1" if legacy else "0"
    try:
        m = importlib.import_module(MOD)
        return importlib.reload(m)
    finally:
        if saved is None:
            os.environ.pop("HAWKES_LEGACY_DIMENSION_RULE", None)
        else:
            os.environ["HAWKES_LEGACY_DIMENSION_RULE"] = saved


def _legacy_reference(decays, thinning_variate):
    """The pre-6e94941 code, transcribed verbatim from `git show 6e94941`."""
    k = 0
    total = decays[k]
    while thinning_variate >= total:
        k += 1
        total += decays[k]
    return k


def test_legacy_mode_matches_the_original_code_exactly():
    m = _load(legacy=True)
    rng = np.random.RandomState(0)
    for _ in range(500):
        d = rng.uniform(0.01, 2.0, size=12)
        lamb = float(d.sum())
        lamb_bar = lamb * rng.uniform(0.6, 1.4)      # bound sometimes violated
        tv = rng.uniform(0, 1) * lamb_bar
        if tv >= d.sum():                             # original would IndexError
            continue
        assert m.sample_dimension(d.reshape(12, 1), lamb, tv) == _legacy_reference(d, tv)


def test_legacy_mode_draws_no_random_number():
    """The original reused the thinning variate. If legacy mode consumed an
    extra uniform the two arms' RNG streams would diverge for a second reason."""
    m = _load(legacy=True)
    d = np.full((12, 1), 1.0)
    np.random.seed(7)
    before = np.random.get_state()[2]
    m.sample_dimension(d, 12.0, 3.5)
    assert np.random.get_state()[2] == before, "legacy mode consumed an RNG draw"


def test_default_mode_is_unchanged_and_proportional():
    m = _load(legacy=False)
    d = np.array([5.0] + [1.0] * 11).reshape(12, 1)
    lamb = float(d.sum())
    np.random.seed(1)
    counts = np.zeros(12)
    for _ in range(40000):
        counts[m.sample_dimension(d, lamb, thinning_variate=0.0)] += 1
    emp = counts / counts.sum()
    want = d.reshape(-1) / lamb
    assert np.abs(emp - want).max() < 0.01, (emp, want)


def test_default_mode_ignores_the_thinning_variate_entirely():
    """A thinning_variate of 0 would pin the legacy walk to index 0. The default
    path must be indifferent to it, or passing it at the call site would have
    changed production behaviour."""
    m = _load(legacy=False)
    d = np.full((12, 1), 1.0)
    out = []
    for tv in (0.0, 5.0, 11.9, None):
        np.random.seed(3)
        out.append([m.sample_dimension(d, 12.0, tv) for _ in range(200)])
    for o in out[1:]:
        assert o == out[0], "default path depends on the thinning variate"
    assert len(set(out[0])) > 6, "default path collapsed onto few dimensions"


def test_the_two_modes_actually_differ_when_the_bound_is_breached():
    """If they agreed, the control arm would be worthless."""
    leg = _load(legacy=True)
    d = np.full((12, 1), 1.0)
    lamb = 12.0
    lamb_bar = 0.75 * lamb                  # bound breached by 33%
    np.random.seed(11)
    kl = [leg.sample_dimension(d, lamb, np.random.uniform(0, 1) * lamb_bar)
          for _ in range(20000)]
    assert max(kl) < 11, "legacy rule reached the tail; the bound was not breached"
    fix = _load(legacy=False)
    np.random.seed(11)
    kf = [fix.sample_dimension(d, lamb) for _ in range(20000)]
    assert max(kf) == 11, "fixed rule must be able to reach the tail"
    # Ask is dimensions 0-5, Bid 6-11: the legacy rule must show the Ask excess
    ask_l = np.mean(np.array(kl) < 6)
    ask_f = np.mean(np.array(kf) < 6)
    assert ask_l - ask_f > 0.05, (ask_l, ask_f)


def test_switch_defaults_off():
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                            "src", "Stochastic_Processes", "Arrival_Models.py")).read()
    assert 'os.environ.get(\n    "HAWKES_LEGACY_DIMENSION_RULE", "0")' in src or \
           '"HAWKES_LEGACY_DIMENSION_RULE", "0"' in src, "default must be off"


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
    _load(legacy=False)
    print("all tests passed" if not failures else "%d FAILURES" % failures)
    sys.exit(1 if failures else 0)
