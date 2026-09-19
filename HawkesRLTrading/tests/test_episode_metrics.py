"""Pin the live per-episode metrics in AR_RL_Trainer.py.

Two things are being pinned:

1. `_twap_slippage_bps` must agree with AR_RL_runner.py's slippage formula
   (runner lines ~605-613) on both sides. Criterion 3 ("the agent raises the
   TWAP's transaction costs") is judged on this number, and a sign error would
   invert the conclusion rather than merely perturb it.

2. `start_midprices` must actually be appended to. It was initialised and
   assigned but never appended, so the arrival midprice -- the denominator of
   every slippage figure -- was absent from trainer output. That is why
   criterion 3 had never been measured from a training run.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINER = os.path.join(HERE, "..", "AR_RL_Trainer.py")
RUNNER = os.path.join(HERE, "..", "AR_RL_runner.py")


def _load_helper():
    """Pull `_twap_slippage_bps` out of the trainer source and exec it alone.

    Importing AR_RL_Trainer would launch a training run, so the function is
    extracted textually. If the def is renamed or moved this test fails loudly,
    which is the intent.
    """
    src = open(TRAINER).read()
    m = re.search(r"^def _twap_slippage_bps\(.*?(?=\n\n\n)", src, re.M | re.S)
    assert m is not None, "_twap_slippage_bps not found in AR_RL_Trainer.py"
    import numpy as np
    ns = {"np": np}
    exec(m.group(0), ns)
    return ns["_twap_slippage_bps"]


def _runner_slip(side, twap_final_cash, executed, start_mid):
    """The runner's formula, transcribed verbatim as the reference."""
    if side == "sell":
        cash_earned = twap_final_cash - 1000000
        benchmark_earned = start_mid * executed
        return (benchmark_earned - cash_earned) * 10000 / benchmark_earned
    cash_spent = 1000000 - twap_final_cash
    benchmark_spent = start_mid * executed
    return (cash_spent - benchmark_spent) * 10000 / benchmark_spent


def test_matches_runner_formula_both_sides():
    f = _load_helper()
    cases = [
        ("sell", 1000000 + 100.0 * 500 * (1 - 10e-4), 500, 100.0),
        ("sell", 1000000 + 99.5 * 480, 480, 100.02),
        ("buy", 1000000 - 100.0 * 500 * (1 + 7e-4), 500, 100.0),
        ("buy", 1000000 - 100.4 * 512, 512, 100.01),
    ]
    for side, cash, ex, mid in cases:
        got, want = f(side, cash, ex, mid), _runner_slip(side, cash, ex, mid)
        assert abs(got - want) < 1e-9, (side, got, want)


def test_sign_convention_positive_means_worse_for_the_twap():
    """Criterion 3 wants this number to go UP. A sign flip inverts the verdict."""
    f = _load_helper()
    # A seller who receives BELOW the arrival mid has suffered cost -> positive.
    assert f("sell", 1000000 + 99.0 * 500, 500, 100.0) > 0
    # A seller who receives ABOVE the arrival mid has done better -> negative.
    assert f("sell", 1000000 + 101.0 * 500, 500, 100.0) < 0
    # A buyer who pays ABOVE the arrival mid has suffered cost -> positive.
    assert f("buy", 1000000 - 101.0 * 500, 500, 100.0) > 0
    assert f("buy", 1000000 - 99.0 * 500, 500, 100.0) < 0


def test_executes_at_the_mid_is_zero_slippage():
    f = _load_helper()
    for side, sgn in (("sell", +1), ("buy", -1)):
        cash = 1000000 + sgn * 100.0 * 500
        assert abs(f(side, cash, 500, 100.0)) < 1e-9


def test_degenerate_inputs_return_none_not_inf():
    """TWAP-absent episodes execute nothing; dividing by zero would poison
    every downstream mean with inf/nan instead of being skipped."""
    f = _load_helper()
    assert f("sell", 1000000.0, 0, 100.0) is None
    assert f("sell", 1000000.0, float("nan"), 100.0) is None
    assert f("buy", 1000000.0, 500, 0) is None
    assert f("buy", 1000000.0, 500, None) is None


def test_known_magnitude():
    """A 10 bps adverse fill must read as ~10 bps, not 1 or 100."""
    f = _load_helper()
    got = f("sell", 1000000 + 100.0 * (1 - 10e-4) * 500, 500, 100.0)
    assert abs(got - 10.0) < 1e-6, got
    got = f("buy", 1000000 - 100.0 * (1 + 10e-4) * 500, 500, 100.0)
    assert abs(got - 10.0) < 1e-6, got


def test_start_midprices_is_no_longer_a_dead_list():
    src = open(TRAINER).read()
    assert "start_midprices.append(starting_midprice)" in src, \
        "start_midprices is assigned but never appended -- slippage has no denominator"
    assert 'start_midprice.npy' in src and "np.save" in src, \
        "the arrival midprice is never saved, so slippage cannot be recomputed post-hoc"


def test_pre_and_post_window_are_bucketed_separately():
    """`inventory_without_twap` pools (100,250] with [400,550]. The pre stretch
    is exactly where front-running happens, so subtracting the pooled baseline
    subtracts the signal: a policy that builds its position before t=250 and
    holds it through the window scores a paired shift of ~0 BY CONSTRUCTION.
    The pre window must therefore be recorded on its own."""
    src = open(TRAINER).read()
    assert "inventory_pre_twap.append" in src and "inventory_post_twap.append" in src
    assert '"inv_level_pre_window"' in src, "the correct criterion-1 baseline is not saved"
    assert '"inv_level_post_window"' in src
    # split must key on the window boundaries, not on twap_present
    assert "if Simstate['TimeCode'] <= twap_start_time:" in src
    assert "elif Simstate['TimeCode'] >= twap_end_time:" in src
    # and all of them must be reset per episode, or they accumulate across the
    # run. Check on the code with comments stripped, so that documenting one of
    # these lists cannot push another out of a fixed character window.
    code = "\n".join(l for l in src.splitlines()
                     if not l.strip().startswith("#"))
    i = code.index("inventory_without_twap = []")
    block = code[i:i + 300]
    for name in ("inventory_pre_twap", "inventory_post_twap",
                 "inventory_window_clock"):
        assert "%s = []" % name in block, "%s is not reset per episode" % name


def _bucket(t, twap_present, start=250, end=400):
    """The trainer's actual bucketing conditions, transcribed. Returns the set
    of destination lists a sample at time t lands in."""
    dests = set()
    if end > t > start:
        dests.add("clock")
    if twap_present and (end > t > start):
        dests.add("in")
    else:
        dests.add("without")
    if t <= start:
        dests.add("pre")
    elif t >= end:
        dests.add("post")
    return dests


def test_absent_episodes_still_record_the_clock_window():
    """The present-vs-absent contrast is the only front-running measure a
    SINGLE-SIDE run can be scored on, and it needs the (250,400) inventory
    level on TWAP-ABSENT episodes. `inventory_with_twap_*` is keyed on the
    meta-order actually being present, so it is empty there. Without a separate
    clock-window record the absent arm of the contrast does not exist."""
    for t in (260.0, 300.0, 399.0):
        assert "clock" in _bucket(t, twap_present=False), t
        assert "in" not in _bucket(t, twap_present=False), t
    # and on a present episode the two must be the same samples
    for t in (260.0, 300.0, 399.0):
        d = _bucket(t, twap_present=True)
        assert "clock" in d and "in" in d, t
    # outside the window, neither
    for t in (100.0, 250.0, 400.0, 550.0):
        assert "clock" not in _bucket(t, twap_present=True), t
        assert "clock" not in _bucket(t, twap_present=False), t


def test_the_report_scores_a_single_side_run_on_present_absent():
    """criteria_report refused to score criterion 1 on a single-side run. That
    refusal was too strong: the present-absent contrast differences out the
    per-run directional bias without needing both sides."""
    src = open(os.path.join(os.path.dirname(TRAINER), "criteria_report.py")).read()
    assert "inv_level_window_clock" in src
    assert "present-absent contrast" in src


def test_bucketing_conditions_match_the_source():
    """Guard the transcription above against drift."""
    src = open(TRAINER).read()
    assert "if twap_present and (twap_end_time > Simstate['TimeCode'] > twap_start_time):" in src
    assert "if twap_end_time > Simstate['TimeCode'] > twap_start_time:" in src
    assert "if Simstate['TimeCode'] <= twap_start_time:" in src
    assert "elif Simstate['TimeCode'] >= twap_end_time:" in src


def test_present_episode_partitions_into_pre_in_post():
    """Exactly one of pre/in/post per sample, including at the boundaries."""
    for t in (100.1, 249.9, 250.0, 250.1, 300, 399.9, 400.0, 400.1, 550):
        d = _bucket(t, True) & {"pre", "in", "post"}
        assert len(d) == 1, (t, d)


def test_absent_episode_leaves_the_middle_third_in_neither_pre_nor_post():
    """The bug: nothing populates the in-window list when the meta-order is
    off, so pre+in+post covers only two thirds of an absent episode."""
    middle = [t for t in (250.1, 300, 399.9) if not (_bucket(t, False) & {"pre", "post"})]
    assert middle == [250.1, 300, 399.9], middle


def test_the_metric_uses_a_union_that_covers_the_whole_episode():
    """`without + in` is complete for BOTH episode kinds, which is why
    frac_at_inventory_limit must be computed from those and not pre+in+post."""
    for present in (True, False):
        for t in (100.1, 249.9, 250.0, 250.1, 300, 399.9, 400.0, 550):
            assert _bucket(t, present) & {"without", "in"}, (present, t)
    src = open(TRAINER).read()
    i = src.index('"frac_at_inventory_limit"')
    block = src[i:i + 700]
    assert "inventory_without_twap + _inw" in block, \
        "metric is not computed from the complete-episode union"
    assert "inventory_pre_twap + _inw + inventory_post_twap" not in block


def _default_times():
    """Evaluate the trainer's timing block at its DEFAULTS, without importing
    the simulator. Behavioural, so parameterising a line cannot silently break
    the check the way a source-text match does."""
    import os as _os
    src = open(TRAINER).read()
    keep = [l for l in src.splitlines()
            if l.startswith(("TWAP_ORDER_SIZE", "TWAP_DURATION", "TWAP_WINDOW_SIZE",
                             "TWAP_ACTION_FREQ", "RL_INVENTORY_LIMIT", "STOP_TIME",
                             "start_trading_lag", "twap_start_time", "twap_end_time",
                             "twap_off_time"))]
    ns = {"os": _os}
    exec("\n".join(keep), ns)
    return ns


def test_the_three_windows_are_equal_thirds():
    """At the defaults the episode is three equal 150s thirds. HANDOFF §1a's
    post-hoc recovery of the pre window (`pre = out[:len(inw)]`, validated by
    `len(out) ~ 2*len(inw)`) depends on this, so it is pinned."""
    ns = _default_times()
    start, end, stop = ns["twap_start_time"], ns["twap_end_time"], ns["STOP_TIME"]
    lag = ns["start_trading_lag"]
    assert (start, end, stop, lag) == (250, 400, 550, 100), (start, end, stop, lag)
    assert end - start == start - lag == stop - end == 150
    assert ns["twap_off_time"] == end, "the TWAP must switch off when it ends"


def test_meta_order_knobs_default_to_the_historical_values():
    """The meta-order shape is now settable, because it is what decides whether
    in-window front-running is possible at all. Every default must reproduce
    the hardcoded values, or every existing arm silently changes market."""
    ns = _default_times()
    assert ns["TWAP_ORDER_SIZE"] == 150
    assert ns["TWAP_DURATION"] == 150
    assert ns["TWAP_WINDOW_SIZE"] == 25
    assert ns["TWAP_ACTION_FREQ"] == 1
    assert ns["RL_INVENTORY_LIMIT"] == 25
    assert ns["STOP_TIME"] == 550


def test_a_longer_meta_order_moves_the_window_and_keeps_a_post_stretch():
    import os as _os
    _os.environ["TWAP_DURATION"] = "300"
    _os.environ["STOP_TIME"] = "700"
    try:
        ns = _default_times()
        assert ns["twap_start_time"] == 250
        assert ns["twap_end_time"] == 550
        assert ns["twap_off_time"] == 550
        assert ns["STOP_TIME"] > ns["twap_end_time"], "no post-window stretch"
        # and the equal-thirds assumption is now FALSE -- any analysis relying
        # on it must not be pointed at such a run
        assert not (ns["twap_end_time"] - ns["twap_start_time"]
                    == ns["STOP_TIME"] - ns["twap_end_time"])
    finally:
        del _os.environ["TWAP_DURATION"], _os.environ["STOP_TIME"]


def test_live_metrics_cover_all_three_criteria():
    src = open(TRAINER).read()
    for key in ("inv_level_in_window", "inv_level_pre_window",   # criterion 1
                "terminal_pnl",          # criterion 2
                "twap_slippage_bps",     # criterion 3
                "twap_present", "side", "seed"):
        assert '"%s"' % key in src, "%s missing from episode_metrics" % key
    # and it must be flushed every episode, not only at the end
    assert "_save_episode_metrics()" in src


def test_in_window_bucketing_keys_on_actual_presence():
    """A TWAP-absent episode's [twap_start, twap_end] stretch is ordinary
    market-making time. Filing it as "with TWAP" would contaminate the exact
    comparison the metric exists to make."""
    src = open(TRAINER).read()
    assert re.search(r"if twap_present and \(twap_end_time > Simstate\['TimeCode'\] > twap_start_time\)", src), \
        "inventory bucketing no longer gates on twap_present"


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
