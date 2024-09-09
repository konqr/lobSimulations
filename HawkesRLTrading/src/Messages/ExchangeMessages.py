from dataclasses import dataclass
from HawkesRLTrading.src.Messages.Message import Message
from HawkesRLTrading.src.Orders import Order
@dataclass
class ExchangeMsg(Message):
    #Messages that an exchange sends to an agent or the kernel
    pass

@dataclass
class PartialOrderFill(ExchangeMsg): #Exchange to agent
    order: Order
    consumed: int
    
@dataclass
class OrderAutoCancelledMsg(ExchangeMsg):#exchange to agent 
    order: Order

@dataclass
class OrderExecutedMsg(ExchangeMsg):#exchange to agent
    order: Order
    
@dataclass
class LimitOrderAcceptedMsg(ExchangeMsg):
    order: Order
@dataclass
class WakeAgentMsg(ExchangeMsg):
    time: float
    pass
@dataclass
class TradeNotificationMsg(WakeAgentMsg):
    time: float
    pass
@dataclass
class SpreadNotificationMsg(WakeAgentMsg):
    time: float
    pass
@dataclass
class BeginTradingMsg(WakeAgentMsg):
    time: float
    pass
