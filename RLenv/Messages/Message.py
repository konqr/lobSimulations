from dataclasses import dataclass, field
from typing import ClassVar, List
@dataclass
class Message:
    """The base message class is responsible for delivering messages between the orderbook, trading agents, and the kernel. The post_init method here implements an autoincrementing counter of messages"""
    __message_counter: ClassVar[int] =1 
    message_id: int=field(init=False)
    def __post_init__(self):
        self.message_id: int = Message.__message_counter
        Message.__message_counter += 1
    def __lt__(self, other: "Message") -> bool:
        return self.message_id < other.message_id

    def __le__(self, other: "Message") -> bool:
        return self.message_id <= other.message_id

    def __eq__(self, other: "Message") -> bool:
        return self.message_id == other.message_id

    def __ne__(self, other: "Message") -> bool:
        return self.message_id != other.message_id

    def __gt__(self, other: "Message") -> bool:
        return self.message_id > other.message_id

    def __ge__(self, other: "Message") -> bool:
        return self.message_id >= other.message_id

    def messagetype(self) -> str:
        return self.__class__.__name__

    