from dataclasses import dataclass
from RLenv.Messages.Message import Message
from RLenv.Orders import Order, LimitOrder,MarketOrder, CancelOrder

@dataclass
class OrderMsg(Message):
    pass

@dataclass
class LimitOrderMsg(OrderMsg):
    order: LimitOrder

@dataclass
class MarketOrderMsg(OrderMsg):
    order: MarketOrder


@dataclass
class CancelOrderMsg(OrderMsg):
    order: CancelOrder