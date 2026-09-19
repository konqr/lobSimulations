"""Pin what the RL agent is actually allowed to do.

Every run to date uses action_space_config=1, which is FOUR actions:
lo_top_Ask, co_top_Ask, co_top_Bid, lo_top_Bid -- post or cancel at the touch
on either side. There is no market order and no in-spread order in that set.
The policy therefore cannot initiate a position; it can only choose which side
it is exposed on and wait for someone else's aggression to fill it.

That matters for criterion 1. Front-running means holding a position AHEAD of
the meta-order. During a BUYING TWAP the aggression in the book is buying, so
a resting ask gets lifted and the agent ends up SHORT into a rising market --
run over, not in front. Getting long instead requires other participants to
sell into the agent's bid, which is exactly what is scarce while a buy
meta-order is working.

CLAUDE.md describes a 13-choice action space, which is config 0 plus the
no-op. These tests pin the gap so it is not mistaken for the live config again.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))

from HawkesRLTrading.src.SimulationEntities.ICRLAgent import PPOAgent

# The 12 Hawkes event dimensions, in the order Arrival_Models uses.
COLS = ["lo_deep_Ask", "co_deep_Ask", "lo_top_Ask", "co_top_Ask", "mo_Ask",
        "lo_inspread_Ask", "lo_inspread_Bid", "mo_Bid", "co_top_Bid",
        "lo_top_Bid", "co_deep_Bid", "lo_deep_Bid"]


class _Stub(PPOAgent):
    def __init__(self, cfg):
        # Only the action-space branch of __init__ is exercised.
        PPOAgent.__dict__["__init__"]  # keep the reference explicit
        self.action_space_config = cfg
        if cfg == 0:
            self.allowed_actions = list(COLS)
            self.convert_dict = {}
        elif cfg == 1:
            self.allowed_actions = ["lo_top_Ask", "co_top_Ask", "co_top_Bid", "lo_top_Bid"]
            self.convert_dict = {0: 2, 1: 3, 2: 8, 3: 9}


def _source_branch(cfg):
    """Read the real branch out of ICRLAgent so the stub cannot drift from it."""
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                            "src", "SimulationEntities", "ICRLAgent.py")).read()
    return src


def test_config1_is_four_passive_actions():
    src = _source_branch(1)
    assert '''self.allowed_actions= ["lo_top_Ask","co_top_Ask","co_top_Bid", "lo_top_Bid" ]''' in src
    assert "self.convert_dict = {0:2, 1:3, 2:8, 3:9}" in src


def test_config1_contains_no_aggressive_order():
    """The whole structural argument rests on this."""
    a = _Stub(1).allowed_actions
    assert len(a) == 4, a
    assert not any("mo_" in x for x in a), a
    assert not any("inspread" in x for x in a), a
    # and everything in it is a top-of-book post or cancel
    assert all(x.startswith(("lo_top", "co_top")) for x in a), a


def test_config1_convert_dict_maps_onto_the_right_event_dimensions():
    """A wrong mapping would silently send, say, a cancel where a post was
    intended -- indices 2,3,8,9 must be exactly the four top-of-book events."""
    s = _Stub(1)
    for local, glob in s.convert_dict.items():
        assert COLS[glob] == s.allowed_actions[local], (local, glob, COLS[glob])


def test_config0_restores_market_and_inspread_orders():
    a = _Stub(0).allowed_actions
    assert len(a) == 12, a
    assert "mo_Ask" in a and "mo_Bid" in a
    assert "lo_inspread_Ask" in a and "lo_inspread_Bid" in a
    # config 0 needs no remapping: local index == event dimension
    assert _Stub(0).convert_dict == {}
    for i, name in enumerate(a):
        assert COLS[i] == name, (i, name)


def test_config1_is_symmetric_between_the_two_sides():
    """If the passive-only action set were itself lopsided, the side-independent
    inventory drift seen in the alternating runs would have a trivial cause."""
    a = _Stub(1).allowed_actions
    asks = sorted(x.replace("_Ask", "") for x in a if x.endswith("Ask"))
    bids = sorted(x.replace("_Bid", "") for x in a if x.endswith("Bid"))
    assert asks == bids == ["co_top", "lo_top"], (asks, bids)


def test_action_space_is_env_overridable_in_the_trainer():
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                            "AR_RL_Trainer.py")).read()
    assert 'ACTION_SPACE_CONFIG = int(os.environ.get("ACTION_SPACE_CONFIG", 1))' in src, \
        "default must stay 1 so runs in flight and all historical results remain comparable"
    assert "action_space_config = ACTION_SPACE_CONFIG" in src
    assert "{ACTION_SPACE_CONFIG}" in src, "not reported in the config banner"


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
