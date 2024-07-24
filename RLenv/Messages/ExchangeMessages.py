from dataclasses import dataclass
from RLenv.Messages.Message import Message
from RLenv.Orders import Order, CancelOrder
@dataclass
class ExchangeMsg(Message):
    #Messages that an exchange sends to an agent
    pass

@dataclass
class OrderAcceptedMsg(ExchangeMsg):
    order: Order
@dataclass
class OrderCancelledMsg(ExchangeMsg):
    order: CancelOrder

@dataclass
class OrderExecutedMsg(ExchangeMsg):
    order: Order
    


@dataclass
class TradeNotificationMsg(ExchangeMsg):
    time: int


