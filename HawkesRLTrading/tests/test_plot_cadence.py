"""Pin that the per-episode inventory plots are actually called.

`graphInventories` was called at the end of EVERY episode until 1f6f7f5
("LSTM training, first push"), which removed the call but left the function
defined -- dead code, and the reason the per-episode distribution PNGs stopped
appearing. Nothing caught it because a function that is merely never called
raises nothing and fails no test.

These tests pin the call sites, so a future refactor that drops one fails
loudly instead of silently removing the output.
"""
import os
import re
import sys

TRAINER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "AR_RL_Trainer.py")


def _src():
    return open(TRAINER).read()


def test_graph_inventories_is_actually_called():
    """The regression. It is not enough for the function to exist."""
    src = _src()
    assert "def graphInventories(" in src
    calls = re.findall(r"^\s+graphInventories\(", src, re.M)
    assert calls, "graphInventories is defined but never called -- this is the 1f6f7f5 regression"


def test_distribution_plot_defaults_to_every_episode():
    src = _src()
    assert 'DIST_PLOT_EVERY = int(os.environ.get("DIST_PLOT_EVERY", 1))' in src, \
        "default must be 1 = every episode, matching the behaviour before 1f6f7f5"


def test_distribution_plot_is_not_inside_the_checkpoint_block():
    """It was per-episode originally. The checkpoint block runs every 4
    episodes and also trains and saves models, so nesting the plot inside it
    would quietly quarter the output rate."""
    src = _src()
    call = src.index("graphInventories(withtwap_buy")
    guard = src.rindex("if DIST_PLOT_EVERY", 0, call)
    between = src[guard:call]
    assert "% 4" not in between, "distribution plot is gated on the checkpoint cadence"
    # and the guard must be the nearest enclosing condition, not far above it
    assert between.count("\n") < 12, between


def test_trajectory_plot_has_its_own_cadence():
    """It used to be tied to `episode % 4 == 0` by nesting, so it could not be
    changed without also changing how often models are saved and trained."""
    src = _src()
    assert 'TRAJ_PLOT_EVERY = int(os.environ.get("TRAJ_PLOT_EVERY", 4))' in src
    i = src.index("plot_avg_inventory_trajectories(episode_inv_trajectories_buy")
    guard = src[src.rindex("\n", 0, i) + 1:i]
    # the call is on the line after its own `if`, so look at the preceding lines
    window = src[max(0, i - 200):i]
    assert "TRAJ_PLOT_EVERY" in window, window


def test_both_cadences_can_be_disabled():
    """0 must disable rather than raise ZeroDivisionError from the modulo."""
    for name in ("DIST_PLOT_EVERY", "TRAJ_PLOT_EVERY"):
        m = re.search(r"if %s and \(episode %% %s == 0\)" % (name, name), _src())
        assert m, "%s is not guarded against 0 before the modulo" % name


def test_distribution_plot_filename_carries_the_label():
    """It wrote `<log_dir>_all_inventory_distributions_episode_N.png` with no
    label, unlike every other plot in the file."""
    src = _src()
    assert "log_dir + label + f'_all_inventory_distributions_episode_" in src


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
