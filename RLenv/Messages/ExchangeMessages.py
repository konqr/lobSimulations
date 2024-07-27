from dataclasses import dataclass
from RLenv.Messages.Message import Message
from RLenv.Orders import Order, CancelOrder
@dataclass
class ExchangeMsg(Message):
    #Messages that an exchange sends to an agent or the kernel
    pass

@dataclass
class OrderPending(ExchangeMsg):
    order: Order
@dataclass
class OrderAcceptedMsg(ExchangeMsg):
    order: Order
@dataclass
class OrderCancelledMsg(ExchangeMsg):
    order: CancelOrder

@dataclass
class OrderExecutedMsg(ExchangeMsg):
    order: Order
    

class WakeAgentMsg(ExchangeMsg):
    pass
class TradeNotificationMsg(WakeAgentMsg):
    pass

class SpreadNotificationMsg(WakeAgentMsg):
    pass
class BeginTradingMsg(WakeAgentMsg):
    pass
