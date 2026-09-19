"""Pin the Ask/Bid symmetry of the exchange's book maintenance.

Motivation: the drift study found a `lo_deep` Ask excess of +12.32 (p<0.0001)
in the FULL simulation with the Hawkes kernels nulled, but only +3.83 (p=0.35)
in the isolated generator. With the kernels off the arrivals are Poisson at
exactly mirror-symmetric baselines, so a residual imbalance that appears only
when the exchange is attached has to come from the exchange's order handling.
Separately, a 200-path experiment showed the thinning fix accounts for only
about +0.52% of the +1.29% Ask-excess improvement credited to it, leaving ~0.77%
attributed to nothing -- and this is the leading suspect.

The concrete defect these tests pin is in `regeneratequeuedepletion`: the Ask
branch assigns `self.askprice = self.askprices["Ask_L1"]` (a scalar) while the
Bid branch assigns `self.bidprice = [self.bidprices["Bid_L1"]]` -- a LIST. It
is silent -- it never raises -- and the reason is worth stating precisely,
because it is a coincidence rather than a design. Every operation the corrupted
value flows through happens to be numpy-compatible: `askprice - bidprice` is
`np.float64 - list`, which broadcasts to `array([0.03])`; `np.round(bidprice, 2)`
gives `array([99.98])`; and `self.bidprice + self.ticksize`
(`Exchange.py:299`, validating a bid in-spread order) is `list + np.float64`,
which also broadcasts. Size-1 arrays are truthy, so every downstream guard
still works. Had `ticksize` been a plain Python float rather than
`np.float64`, that last one would be list concatenation and would raise
`TypeError` -- so the silence is one type-declaration away from a crash.

It propagates: `Exchange` assigns `askprice - bidprice` to
`Arrival_model.spread`, which the generator uses to scale the in-spread
intensities. But it scales dimensions 5 and 6 identically, so the array-ness
alone does not obviously produce a SIDE bias, and this defect is therefore not
yet established as the source of the `lo_deep` excess. It is a genuine Ask/Bid
asymmetry in the code and should be fixed regardless.

These tests DOCUMENT current behaviour rather than assert the fix, because 15
training jobs are mid-flight against this simulator and changing it would break
comparability. `test_bid_branch_wraps_bidprice_in_a_list` is expected to pass
while the defect is present and to FAIL once it is fixed -- at which point flip
it to the symmetric assertion below it.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))

import numpy as np

from HawkesRLTrading.src.SimulationEntities.Exchange import Exchange


class _FakeArrivalModel:
    """`regeneratequeuedepletion` refills a level through the arrival model."""

    def generate_orders_in_queue(self, loblevel, numorders=10):
        return [40] * 5

    def generate_queuesize(self, loblevel):
        return 400


def _exchange():
    e = Exchange(symbol="INTC", ticksize=0.01, LOBlevels=2, numOrdersPerLevel=10,
                 PriceMid0=100, spread0=0.03)
    e.Arrival_model = _FakeArrivalModel()
    return e


def _deplete(side_level):
    """Build a book with exactly one level emptied."""
    e = _exchange()
    e.asks = {p: ([] if k == side_level else [1] * 5) for k, p in e.askprices.items()}
    e.bids = {p: ([] if k == side_level else [1] * 5) for k, p in e.bidprices.items()}
    return e


def test_initial_book_is_symmetric_about_the_mid():
    e = _exchange()
    up = e.askprices["Ask_L1"] - 100
    dn = 100 - e.bidprices["Bid_L1"]
    assert abs(up - dn) < 1e-12, (up, dn)
    up2 = e.askprices["Ask_L2"] - 100
    dn2 = 100 - e.bidprices["Bid_L2"]
    assert abs(up2 - dn2) < 1e-12, (up2, dn2)


def test_initial_spread_is_two_ticks_not_three():
    """`spread0=0.03` with `//` on floats gives 0.03 // 0.01 == 2.0, so the book
    opens two ticks wide, not three. Symmetric, but not what the config says."""
    e = _exchange()
    assert abs((e.askprices["Ask_L1"] - e.bidprices["Bid_L1"]) - 0.02) < 1e-12


def test_ask_branch_leaves_askprice_a_scalar():
    e = _deplete("Ask_L1")
    e.regeneratequeuedepletion()
    assert np.isscalar(e.askprice) or isinstance(e.askprice, np.floating), type(e.askprice)


def test_bid_branch_wraps_bidprice_in_a_list():
    """CURRENT BEHAVIOUR, and the defect. Exchange.py's bid branch assigns
    `self.bidprice = [self.bidprices["Bid_L1"]]` where the ask branch assigns a
    scalar. Flip this to the assertion in the test below once it is fixed."""
    e = _deplete("Bid_L1")
    e.regeneratequeuedepletion()
    assert isinstance(e.bidprice, list), (
        "bidprice is no longer a list -- the defect appears to be FIXED. "
        "Delete this test and enable test_bid_branch_should_leave_bidprice_a_scalar.")


def _bid_branch_should_leave_bidprice_a_scalar():
    """Enable this (rename without the leading underscore) when the fix lands."""
    e = _deplete("Bid_L1")
    e.regeneratequeuedepletion()
    assert np.isscalar(e.bidprice) or isinstance(e.bidprice, np.floating), type(e.bidprice)


def test_the_list_silently_turns_the_spread_into_an_array():
    """One consequence is silent: the spread becomes a 1-element array, which is
    truthy, roundable and comparable, and Exchange assigns it straight to
    Arrival_model.spread."""
    e = _deplete("Bid_L1")
    e.regeneratequeuedepletion()
    spread = e.askprice - e.bidprice
    assert isinstance(spread, np.ndarray) and spread.shape == (1,), (type(spread), spread)
    # and every guard it flows through still works, which is why it is silent
    assert bool(spread < 2) in (True, False)
    assert float(np.round(spread, 2)) == float(np.round(spread[0], 2))


def test_book_prices_themselves_stay_symmetric_through_depletion():
    """The corruption is confined to the cached touch price. If the price
    LEVELS were also asymmetric the effect would be far larger, so pin that."""
    ea = _deplete("Ask_L1")
    ea.regeneratequeuedepletion()
    eb = _deplete("Bid_L1")
    eb.regeneratequeuedepletion()
    # ask side moved one tick up; bid side one tick down; mirror images
    assert abs((ea.askprices["Ask_L1"] - 100) - (100 - eb.bidprices["Bid_L1"])) < 1e-12
    assert abs((ea.askprices["Ask_L2"] - 100) - (100 - eb.bidprices["Bid_L2"])) < 1e-12


def test_l2_branches_are_symmetric():
    for lvl in ("Ask_L2", "Bid_L2"):
        e = _deplete(lvl)
        e.regeneratequeuedepletion()
        assert len(e.asks[e.askprices["Ask_L2"]]) > 0
        assert len(e.bids[e.bidprices["Bid_L2"]]) > 0
    # neither L2 branch touches the cached touch prices
    e = _deplete("Ask_L2")
    e.regeneratequeuedepletion()
    assert not isinstance(e.bidprice, list)


def test_the_bid_inspread_path_is_silent_only_because_ticksize_is_numpy():
    """Exchange.py:299 validates a bid in-spread order with
    `self.bidprice + self.ticksize`. With the list on the left of the `+` that
    would be list concatenation and would raise -- but `ticksize` is
    `np.float64`, so numpy broadcasts instead and it returns an array. The
    defect is silent by coincidence of types, one declaration away from a
    crash."""
    e = _deplete("Bid_L1")
    e.regeneratequeuedepletion()
    out = e.bidprice + e.ticksize          # list + np.float64 -> broadcasts
    assert isinstance(out, np.ndarray), type(out)
    # ...but only because ticksize is a numpy scalar. With a Python float this
    # is list concatenation and raises, which is the fragility worth pinning.
    try:
        e.bidprice + float(e.ticksize)
    except TypeError:
        return
    raise AssertionError("list + python float no longer raises; re-derive this note")


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
