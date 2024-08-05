from dataclasses import dataclass
from RLenv.Messages.Message import Message
from RLenv.Orders import Order, CancelOrder
@dataclass
class ExchangeMsg(Message):
    #Messages that an exchange sends to an agent or the kernel
    pass

@dataclass
class OrderPending(ExchangeMsg): #this one ensures that a random market simulated order is made at a valid time. It is sent from the exchange to the kernel
    order: Order
@dataclass
class PartialOrderFill(ExchangeMsg): #Exchange to agent
    order: Order
    newsize: int
    
@dataclass
class OrderAutoCancelledMsg(ExchangeMsg):#exchange to agent 
    order: Order

@dataclass
class OrderExecutedMsg(ExchangeMsg):#exchange to agent
    order: Order
    

class WakeAgentMsg(ExchangeMsg):
    pass
class TradeNotificationMsg(WakeAgentMsg):
    pass

class SpreadNotificationMsg(WakeAgentMsg):
    pass
class BeginTradingMsg(WakeAgentMsg):
    pass
