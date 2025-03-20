import numpy as np
from abc import ABC, abstractmethod 
from typing import Any, List, Optional, Tuple, ClassVar
import pandas as pd
import logging
from HawkesRLTrading.src.Messages.Message import Message

logger = logging.getLogger(__name__)
class Entity:
    """
    Base Entity class of an object in the simulation. An Entity can either be a trading agent or an exchange

    Attributes:
        id: Must be a unique number (usually autoincremented) belonging to the subclass.
        type: is it a trading agent or exchange?
        seed: Every entity is given a random seed for stochastic purposes.
        log_events: flag to log or not the events during the simulation 
            Logging format:
                time: time of event
                event_type: label of the event (e.g., Order submitted, order accepted, last trade etc....)
                event: actual event to be logged (co_deep_Ask, lo_deep_Ask...)
                size: size of the order (if applicable)
        log_to_file: flag to write on disk or not the logged events
    """
    _entity_counter: ClassVar[int]=1
    _registry={}
    def __init__(self, type: str = None, seed=1, log_events: bool = True, log_to_file: bool = False) -> None:
        self.id=Entity._entity_counter
        Entity._entity_counter+=1
        self.type=type
        self.name="Entity"+str(self.id)+"_"+str(self.type)
        self.log_events=log_events
        self.log_to_file=log_events & log_to_file
        self.seed=seed
        # Simulation attributes
        
        self.kernel=None     
        #What time does the entity think it is?
        self.current_time: int = 0 
        
        #Entities will maintain a log of their activities, events will likely be stored in the format of (time, event_type, eventname, size)
        self.log: List[Any]=[]
        
        if self.log_to_file:
            self.filename=None
        
        #Finally, add the entity object to the registry
        Entity._registry[self.id]=self
    
    @classmethod
    def get_entity_by_id(cls, ID: int):
        return cls._registry.get(ID)   
    
    """Methods for communication with the exchange, kernel, or other entities"""
    def sendmessage(self, recipientID: int, message: Message) -> None:
        """
        Sends a message to a specific recipient. If the recipient is -1 then it is aimed at the kernel
        """
        assert self.kernel is not None, f"Kernel not linked to {type(self).__name__} {self.id}"
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"At {self.current_time}, {type(self).__name__} {self.id} sent: {message} to {type(Entity.get_entity_by_id(recipientID)).__name__} {recipientID}")
        self.kernel.sendmessage(self.current_time, senderID=self.id, recipientID=recipientID, message=message)
        
    def sendbatchmessage(self, recipientIDs: List[int], message: Message) -> None:
        """
        Sends a batch message to multiple recipients at the same time
        """
        assert self.kernel is not None, f"Kernel not linked to {type(self).__name__} {self.id}"
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("At {}, entity {} sent batch message: {} to entities {}".format(self.current_time, self.id, message, recipientIDs))
        self.kernel.sendbatchmessage(self.current_time, self.id, recipientIDs, message)
            
        
        
    def receivemessage(self, current_time, senderID: int, message: Message):
        """
        Called each time a message destined for this entity reaches the front of the
        kernel's priority queue.
        Arguments:
            current_time: The simulation time at which the kernel is delivering this message -- the entity should treat this as "now".
            sender: The entity that is sending the message
            message: An object guaranteed to inherit from the Message.Message class.
        """
        assert self.kernel is not None, f"Kernel not linked to Entity: {self.id}"
        self.current_time=current_time
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("At {}, entity {} received from entity {}: {}".format(self.current_time, self.id, senderID, message))
        
        
    def wakeup(self, current_time: float, delay: float=0) -> None:
        """
        Entities can request a wakeup call at a future simulation time using
        the Entity.set_wakeup(time) method

        This is the method called when the wakeup time arrives.

        Arguments:
            current_time: The simulation time at which the kernel is delivering this
                message -- the entity should treat this as "now".
        """

        assert self.kernel is not None

        self.current_time = current_time+delay

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "At {}, entity {} received wakeup.".format(
                    self.current_time, self.id
                )
            )
    
    def set_wakeup(self, requested_time: float) -> None:
        """
        Called to receive a future call from the kernel at the point of requested_time
        Returns:
            True - if a new wakeup time was set
            False - if the new wakeuptime is beyond simulation limits
        """
        assert self.kernel is not None, f"Kernel is not linked to Entity {self.id}"
        return self.kernel.set_wakeup(self.id, requested_time)
    
           
    """Book-keeping Methods for Internal use"""   
    def _logevent(self, event: List[Any]):
        """Adds an event to this entities log: events are stored differently for different entities."""
        if not self.log_events:
            return
        self.log.append(event)
        return

    def writelog(self, df_log: pd.DataFrame, filename: Optional[str]=None) -> None:
        """
        Called by the entity, usually at the very end of the simulation just before Kernel shuts down to log down any data. The Kernel places the log in a unique
        directory per run, with one filename per agent, also decided by the Kernel using
        agent type, id, etc. If filename is None the Kernel will construct a filename based on the name of
        the Agent requesting log archival.

        Arguments:
            df_log: dataframe that contains all the logged events during the simulation
            filename: Location on disk to write the log to.
        """
        assert self.kernel is not None
        self.kernel.write_log(self.id, df_log, filename)
    
    

    @abstractmethod
    def reset(self):
        pass
    @abstractmethod
    def kernel_start(self, start_time: int) -> None:
        pass
    @abstractmethod
    def kernel_terminate(self) -> None:
        pass
    
        
