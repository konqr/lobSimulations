from dataclasses import dataclass
from src.Messages.Message import Message
from src.Orders import Order, LimitOrder,MarketOrder, CancelOrder

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
    pass

@dataclass
class WakeUpRequestMsg(AgentMsg):
    agentID: int
    pass