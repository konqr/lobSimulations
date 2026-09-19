"""Pin the market-order inventory gates, legacy and symmetric.

The legacy gates in get_action are:

    mo_Ask (SELL) blocked when  inv < 1  OR  inv >= inventorylimit - 2
    mo_Bid (BUY)  blocked when  inv <= 2 - inventorylimit

At inventorylimit=25 that permits SELLING only on inv in [1, 22] but BUYING on
inv in [-22, +inf). Two things are wrong with it:

  * `inv >= limit - 2` blocks SELLING while the agent is very LONG, which is
    backwards -- a position limit should stop the trade that makes the position
    worse, and selling makes a long smaller.
  * its mirror, blocking BUYING while very long, is absent entirely, so nothing
    in this gate bounds inventory from above.

The exploration branch is looser still: no limit clause on the sell, and no bid
clause at all.

All of this is DORMANT in every run to date, because action_space_config=1
emits only u in {2,3,8,9} and never u=4 or u=7. It would corrupt any
action_space_config=0 run, which is why the symmetric mode exists. Legacy stays
the default so runs in flight and historical results are unaffected.

The separate "no shorting via market order" rule (inv < 1) is a modelling
choice, not a bug, and is deliberately kept in BOTH modes.
"""
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))

from HawkesRLTrading.src.SimulationEntities.ICRLAgent import PPOAgent

LIMIT = 25


class _Stub(PPOAgent):
    def __init__(self, inv, symmetric):
        self.Inventory = {"INTC": inv}
        self.inventorylimit = LIMIT
        self.symmetric_mo_gating = symmetric


def _allowed(symmetric, exploration=False):
    sell, buy = set(), set()
    for inv in range(-40, 41):
        a = _Stub(inv, symmetric)
        if not a._mo_ask_blocked(exploration=exploration):
            sell.add(inv)
        if not a._mo_bid_blocked(exploration=exploration):
            buy.add(inv)
    return sell, buy


def test_legacy_exploitation_gates_match_the_original_conditions():
    sell, buy = _allowed(symmetric=False)
    assert sell == set(range(1, LIMIT - 2)), sorted(sell)[:5]
    assert buy == set(range(2 - LIMIT + 1, 41)), (min(buy), max(buy))


def test_legacy_is_long_biased():
    """The finding, stated as a test: the buy region is far larger than the
    sell region, and nothing bounds buying from above."""
    sell, buy = _allowed(symmetric=False)
    assert len(buy) > 2 * len(sell), (len(sell), len(buy))
    assert 40 in buy, "legacy bid gate does not bound inventory from above"
    assert LIMIT - 1 not in sell, "legacy ask gate blocks selling while very long"


def test_legacy_exploration_branch_is_looser_still():
    sell, buy = _allowed(symmetric=False, exploration=True)
    assert sell == set(range(1, 41)), "exploration sell gate should have no limit clause"
    assert buy == set(range(-40, 41)), "exploration branch had no bid clause at all"


def test_symmetric_gates_are_mirror_images_about_the_no_short_rule():
    sell, buy = _allowed(symmetric=True)
    # Selling is bounded below by the short limit, buying above by the long limit.
    assert sell == set(range(1, 41)), (min(sell), max(sell))
    assert buy == set(range(-40, LIMIT - 2)), (min(buy), max(buy))
    # The long limit now stops BUYING, not selling -- the point of the fix.
    assert LIMIT - 1 not in buy and LIMIT - 1 in sell


def test_symmetric_mode_bounds_inventory_from_above():
    """Legacy leaves buying unbounded above; symmetric must not."""
    _, buy = _allowed(symmetric=True)
    assert max(buy) == LIMIT - 3, max(buy)
    assert all(i < LIMIT - 2 for i in buy)


def test_symmetric_mode_is_identical_in_both_branches():
    """Exploration and exploitation must not disagree, or the behaviour the
    policy is trained on differs from the one it is evaluated on."""
    assert _allowed(symmetric=True, exploration=False) == _allowed(symmetric=True, exploration=True)


def test_no_shorting_rule_survives_in_both_modes():
    for sym in (False, True):
        sell, _ = _allowed(symmetric=sym)
        assert not any(i < 1 for i in sell), (sym, sorted(sell)[:3])


def test_default_is_legacy():
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                            "src", "SimulationEntities", "ICRLAgent.py")).read()
    # match the parameter default, not its position in the signature -- keying
    # on a trailing "):" breaks the moment another kwarg is appended
    assert re.search(r"symmetric_mo_gating\s*=\s*False\s*[,)]", src), \
        "default must stay legacy so runs in flight stay comparable"
    tr = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                           "AR_RL_Trainer.py")).read()
    assert 'os.environ.get("SYMMETRIC_MO_GATING", "false")' in tr
    assert "{SYMMETRIC_MO_GATING}" in tr, "not reported in the config banner"


def test_gates_are_dormant_under_the_default_action_space():
    """Why the legacy asymmetry has not affected any run.

    Two separate things have to hold, and only the first is about convert_dict:

      1. The POLICY path under config 1 maps u through {0:2, 1:3, 2:8, 3:9}, so
         it can never emit u=4 (mo_Ask) or u=7 (mo_Bid), and the gates are never
         consulted.
      2. Market orders ARE still submitted under config 1 -- the inventory-breach
         path issues `mo = 4 if inv > 0 else 7` when |inv| >= inventorylimit --
         but it `return`s BEFORE any gating, so those orders do not pass through
         the gates either. They are forced liquidations, symmetric by
         construction (sell when long, buy when short), not position-taking.

    If the breach path is ever moved below the gates, the legacy asymmetry stops
    being dormant, so this test pins the ordering too."""
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                            "src", "SimulationEntities", "ICRLAgent.py")).read()
    assert "self.convert_dict = {0:2, 1:3, 2:8, 3:9}" in src
    assert 4 not in {2, 3, 8, 9} and 7 not in {2, 3, 8, 9}
    # the breach early-return must precede the first gate call
    i = src.index("mo = 4 if self.countInventory() > 0 else 7")
    j = src.index("_mo_ask_blocked", i)
    between = src[i:j]
    assert "return" in between, "breach path no longer returns before the gates"


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
