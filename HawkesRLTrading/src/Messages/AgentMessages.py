from dataclasses import dataclass
from HawkesRLTrading.src.Messages.Message import Message
from HawkesRLTrading.src.Orders import Order, LimitOrder,MarketOrder, CancelOrder

@dataclass
class AgentMsg(Message):
    pass

#These are agents issuing an order to an exchange
@dataclass
class LimitOrderMsg(AgentMsg):
    order: LimitOrder

@dataclass
class MarketOrderMsg(AgentMsg):
    order: MarketOrder


@dataclass
class CancelOrderMsg(AgentMsg):
    order: CancelOrder

@dataclass
class DoNothing(AgentMsg):
    time_placed: float

