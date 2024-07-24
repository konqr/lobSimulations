from dataclasses import dataclass
from RLenv.Messages.Message import Message
from RLenv.Orders import Order, LimitOrder,MarketOrder, CancelOrder

@dataclass
class AgentMsg(Message):
    pass

@dataclass
class LimitOrderMsg(AgentMsg):
    order: LimitOrder

@dataclass
class MarketOrderMsg(AgentMsg):
    order: MarketOrder


@dataclass
class CancelOrderMsg(AgentMsg):
    order: CancelOrder

