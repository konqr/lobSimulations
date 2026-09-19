"""A deliberately dumb, exactly symmetric market maker.

Purpose is diagnostic, not economic. The agent quotes both sides at the touch
and does nothing else: no signal, no inventory skew, no directional view. Its
only asymmetry is whatever the market gives it.

On a drift-free simulator such an agent should earn roughly the spread and show
no directional edge, so its mark-to-market PnL should straddle zero. On a
simulator with a latent drift it accumulates inventory on the losing side --
falling prices fill its bids -- and loses money in proportion to the drift.
That makes it an end-to-end check on market symmetry that does not depend on
any particular theory of where an asymmetry comes from.

Symmetry is a correctness property here, so it is enforced rather than assumed:
the quote/cancel logic is a side-independent table, the agent alternates which
side it touches first on a fixed parity, and `assert_symmetric_config()`
verifies no side-specific parameter has been introduced.
"""
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from HawkesRLTrading.src.SimulationEntities.GymTradingAgent import GymTradingAgent

# Action indices, from TradingAgent.actions
LO_TOP_ASK, CO_TOP_ASK = 2, 3
CO_TOP_BID, LO_TOP_BID = 8, 9
NO_OP = 12

# (post, cancel) action index per side. The single place a side maps to actions.
_SIDE_ACTIONS = {"Ask": (LO_TOP_ASK, CO_TOP_ASK),
                 "Bid": (LO_TOP_BID, CO_TOP_BID)}
_SIDE_LEVEL = {"Ask": "Ask_L1", "Bid": "Bid_L1"}


class PeggedMMAgent(GymTradingAgent):
    def __init__(self, seed=1, log_events: bool = True, log_to_file: bool = False,
                 strategy: str = "PeggedMM", Inventory: Optional[Dict[str, Any]] = None,
                 cash: int = 1000000, action_freq: float = 1.0,
                 wake_on_MO: bool = False, wake_on_Spread: bool = False,
                 cashlimit: int = 1000000000, inventorylimit: int = 10000,
                 order_size: int = 100, max_quotes_per_side: int = 1,
                 start_trading_lag: int = 0, off_time: Optional[float] = None,
                 rewardpenalty: float = 0.0):
        super().__init__(seed=seed, log_events=log_events, log_to_file=log_to_file,
                         strategy=strategy, Inventory=Inventory, cash=cash,
                         action_freq=action_freq, wake_on_MO=wake_on_MO,
                         wake_on_Spread=wake_on_Spread, cashlimit=cashlimit,
                         inventorylimit=inventorylimit,
                         start_trading_lag=start_trading_lag)
        self.order_size = int(order_size)
        self.max_quotes_per_side = int(max_quotes_per_side)
        self.off_time = off_time
        self.rewardpenalty = abs(rewardpenalty)
        self.start_trading_lag = start_trading_lag
        self._step = 0
        self.mid_history: List[Tuple[float, float]] = []
        self.assert_symmetric_config()
        self.resetseed(seed)

    # -- symmetry guard -----------------------------------------------------
    def assert_symmetric_config(self):
        """No parameter may privilege a side. Guards against future edits."""
        assert set(_SIDE_ACTIONS) == {"Ask", "Bid"}
        assert len(set(len(v) for v in _SIDE_ACTIONS.values())) == 1
        # the two sides' action indices must be mirror images: k <-> 11-k
        post_a, canc_a = _SIDE_ACTIONS["Ask"]
        post_b, canc_b = _SIDE_ACTIONS["Bid"]
        assert post_a + post_b == 11, (post_a, post_b)
        assert canc_a + canc_b == 11, (canc_a, canc_b)

    def resetseed(self, seed):
        np.random.seed(seed)

    # -- helpers ------------------------------------------------------------
    def _positions_on(self, data, side) -> int:
        pos = data.get("Positions") or {}
        lvl = _SIDE_LEVEL[side]
        try:
            return len(pos[lvl])
        except (KeyError, TypeError):
            return 0

    def _mid(self, data) -> Optional[float]:
        lob = data.get("LOB0") or {}
        try:
            return float(lob["Ask_L1"][0] + lob["Bid_L1"][0]) / 2.0
        except (KeyError, TypeError, IndexError):
            return None

    # -- policy -------------------------------------------------------------
    def get_action(self, data=None) -> Optional[Tuple[int, int]]:
        """Quote both sides at the touch, one side per wakeup.

        Only one action fits in a wakeup, so the agent alternates sides on a
        fixed parity. The parity is deterministic and side-agnostic: over an
        even number of steps each side gets exactly the same number of turns.
        """
        if data is not None:
            m = self._mid(data)
            if m is not None:
                self.mid_history.append((self.current_time, m))

        if self.current_time < self.start_trading_lag:
            return (NO_OP, 0)
        if self.off_time is not None and self.current_time > self.off_time:
            return (NO_OP, 0)

        # strict alternation -- neither side is ever favoured
        side = "Ask" if (self._step % 2 == 0) else "Bid"
        self._step += 1
        post, cancel = _SIDE_ACTIONS[side]

        n = self._positions_on(data, side) if data is not None else 0
        if n < self.max_quotes_per_side:
            return (post, self.order_size)
        if n > self.max_quotes_per_side:
            return (cancel, 0)
        return (NO_OP, 0)

    # -- reward (unused for the diagnostic, present for the interface) ------
    def calculaterewards(self) -> Any:
        self.profit = self.cash - self.statelog[0][1]
        self.updatestatelog()
        d = self.statelog[-1][2] - self.statelog[-2][2]
        return d - self.rewardpenalty * self.countInventory()

    # -- diagnostics --------------------------------------------------------
    def mark_to_market(self, mid: Optional[float] = None) -> float:
        """Cash change plus inventory change marked at `mid`.

        Inventory is marked at the CURRENT mid, so a drifting market shows up
        directly: an agent left long into a falling market books the loss.
        """
        if mid is None:
            mid = self.mid_history[-1][1] if self.mid_history else self.mid
        inv0 = self.statelog[0][3]
        sym = self.exchange.symbol if self.exchange is not None else list(inv0)[0]
        d_cash = self.cash - self.statelog[0][1]
        d_inv = self.Inventory[sym] - inv0[sym]
        return float(d_cash + d_inv * mid)
